# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    --wavenet_model=<path> wavenet checkpoint path
    --wavenet_preset=<path> wavenet checkpoint path
    --conditional=<file>    wavenet input mel-spec file
    --symmetric-mels                  Symmetric mel.
    --max-abs-value=<N>               Max abs value [default: -1].
    --length=<T>                      Steps to generate [default: 32000].
    --initial-value=<n>               Initial value for the WaveNet decoder.

    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
from os.path import dirname, join, basename, splitext

import audio
import wavenet_audio
import torch
from torch.autograd import Variable
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams
import wavenet_hparams

from tqdm import tqdm

from keras.utils import np_utils
from wavenet_vocoder.util import is_mulaw_quantize, is_mulaw, is_raw
import librosa
from pathlib import Path
import uuid

use_cuda = torch.cuda.is_available()
_frontend = None  # to be set later

def _to_numpy(x):
    # this is ugly
    if x is None:
        return None
    if isinstance(x, np.ndarray) or np.isscalar(x):
        return x
    # remove batch axis
    if x.dim() == 3:
        x = x.squeeze(0)
    return x.numpy()


def wavegen(model, length=None, c=None, g=None, initial_value=None,
            fast=False, tqdm=tqdm):
    """Generate waveform samples by WaveNet.

    Args:
        model (nn.Module) : WaveNet decoder
        length (int): Time steps to generate. If conditinlal features are given,
          then this is determined by the feature size.
        c (numpy.ndarray): Conditional features, of shape T x C
        g (scaler): Speaker ID
        initial_value (int) : initial_value for the WaveNet decoder.
        fast (Bool): Whether to remove weight normalization or not.
        tqdm (lambda): tqdm

    Returns:
        numpy.ndarray : Generated waveform samples
    """
    from wavenet_train import sanity_check
    sanity_check(model, c, g)

    c = _to_numpy(c)
    g = _to_numpy(g)

    model.eval()
    if fast:
        model.make_generation_fast_()

    if c is None:
        assert length is not None
    else:
        # (Tc, D)
        if c.ndim != 2:
            raise RuntimeError(
                "Expected 2-dim shape (T, {}) for the conditional feature, but {} was actually given.".format(wavenet_hparams.hparams.cin_channels, c.shape))
            assert c.ndim == 2
        Tc = c.shape[0]
        upsample_factor = audio.get_hop_size()
        # Overwrite length according to feature size
        length = Tc * upsample_factor
        # (Tc, D) -> (Tc', D)
        # Repeat features before feeding it to the network
        if not wavenet_hparams.hparams.upsample_conditional_features:
            c = np.repeat(c, upsample_factor, axis=0)

        # B x C x T
        c = torch.FloatTensor(c.T).unsqueeze(0)

    if initial_value is None:
        if is_mulaw_quantize(wavenet_hparams.hparams.input_type):
            initial_value = P.mulaw_quantize(0, wavenet_hparams.hparams.quantize_channels)
        else:
            initial_value = 0.0

    if is_mulaw_quantize(wavenet_hparams.hparams.input_type):
        assert initial_value >= 0 and initial_value < wavenet_hparams.hparams.quantize_channels
        initial_input = np_utils.to_categorical(
            initial_value, num_classes=wavenet_hparams.hparams.quantize_channels).astype(np.float32)
        initial_input = torch.from_numpy(initial_input).view(
            1, 1, wavenet_hparams.hparams.quantize_channels)
    else:
        initial_input = torch.zeros(1, 1, 1).fill_(initial_value)

    g = None if g is None else torch.LongTensor([g])

    # Transform data to GPU
    initial_input = initial_input.to(device)
    g = None if g is None else g.to(device)
    c = None if c is None else c.to(device)

    with torch.no_grad():
        y_hat = model.incremental_forward(
            initial_input, c=c, g=g, T=length, tqdm=tqdm, softmax=True, quantize=True,
            log_scale_min=wavenet_hparams.hparams.log_scale_min)

    if is_mulaw_quantize(wavenet_hparams.hparams.input_type):
        y_hat = y_hat.max(1)[1].view(-1).long().cpu().data.numpy()
        y_hat = P.inv_mulaw_quantize(y_hat, wavenet_hparams.hparams.quantize_channels)
    elif is_mulaw(wavenet_hparams.hparams.input_type):
        y_hat = P.inv_mulaw(y_hat.view(-1).cpu().data.numpy(), wavenet_hparams.hparams.quantize_channels)
    else:
        y_hat = y_hat.view(-1).cpu().data.numpy()

    return y_hat


def tts(model, text, p=0, speaker_id=None, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.
    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    if use_cuda:
        model = model.cuda()
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = Variable(torch.from_numpy(sequence)).unsqueeze(0)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long()
    text_positions = Variable(text_positions)
    speaker_ids = None if speaker_id is None else Variable(torch.LongTensor([speaker_id]))
    if use_cuda:
        sequence = sequence.cuda()
        text_positions = text_positions.cuda()
        speaker_ids = None if speaker_ids is None else speaker_ids.cuda()

    # Greedy decoding
    mel_outputs, linear_outputs, alignments, done = model(
        sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
    output_html = args["--output-html"]
    speaker_id = args["--speaker_id"]
    if speaker_id is not None:
        speaker_id = int(speaker_id)

    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    # Presets
    if hparams.preset is not None and hparams.preset != "":
        preset = hparams.presets[hparams.preset]
        import json
        hparams.parse_json(json.dumps(preset))
        print("Override hyper parameters with preset \"{}\": {}".format(
            hparams.preset, json.dumps(preset, indent=4)))

    _frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = _frontend
    from train import plot_alignment, build_model

    # Model
    model = build_model()

    # Load checkpoints separately
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    waveformx = np.array([])
    melspecx = list()
    sec2numpy = lambda sec: np.zeros(int(hparams.sample_rate * sec))

    
    taco2wavnet = lambda x:  x / np.abs(x).max() * wavenet_hparams.hparams.rescaling_max

    ## TTS： txt --> mel list
    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            #waveform, alignment, _, _ = tts(
            waveform, alignment, _, _ = tts(
                model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True)
            # concatenate ecah wave/mel  
            wavenefwav = taco2wavnet(waveform)
            mel_spectrogram = wavenet_audio.melspectrogram(wavenefwav).astype(np.float32).T
            # generate wav by wavenet model
            waveformx = np.array(wavenefwav,copy=True)  if idx == 0 else np.concatenate((waveformx,sec2numpy(0.5), wavenefwav),axis=0)
            #melspecx = np.array(mel_spectrogram,copy=True)  if idx == 0 else np.array([melspecx,mel_spectrogram])
            melspecx.append(mel_spectrogram)

    ## VOCODER
    
    preset = args["--wavenet_preset"]
    wavenet_model_path = args["--wavenet_model"]
    conditional_path = args["--conditional"]
    symmetric_mels = args["--symmetric-mels"]
    max_abs_value = float(args["--max-abs-value"])
    length = int(args["--length"])
    initial_value = args["--initial-value"]
    initial_value = None if initial_value is None else float(initial_value)

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            wavenet_hparams.hparams.parse_json(f.read())
    # Override hyper parameters
    #params_wavenet.hparams.parse(args["--hparam"])
    assert wavenet_hparams.hparams.name == "wavenet_vocoder"

    from wavenet_train import build_model as wavenet_build_model


 
    torch.set_num_threads(4)
    device = torch.device("cuda" if use_cuda else "cpu")
    # Model
    model = wavenet_build_model().to(device)



    # Load checkpoint
    print("Load checkpoint from {}".format(wavenet_model_path))
    if use_cuda:
        checkpoint = torch.load(wavenet_model_path)
    else:
        checkpoint = torch.load(wavenet_model_path, map_location=lambda storage, loc: storage)
        
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint_name = splitext(basename(wavenet_model_path))[0]

    # Load conditional features
    '''
    if conditional_path is not None:
        c = np.load(conditional_path)
        gg = melspecx[0]
    '''
    wavenet_wav = np.array([])
    for idx, mel in enumerate(melspecx):
        waveform = wavegen(model, length, c=mel, g=speaker_id, initial_value=initial_value, fast=True)
        #wavenet_wav = np.array(waveform,copy=True)  if idx == 0 else np.concatenate((wavenet_wav,sec2numpy(0.5), waveform),axis=0)
        wavenet_wav = np.array(waveform,copy=True)  if idx == 0 else np.concatenate((wavenet_wav, waveform),axis=0)
        

    #mel_filename = 'r9y9-tts-mel.npy' 
    #np.save(os.path.join(dst_dir, mel_filename), mel_spectrogram.astype(np.float32), allow_pickle=False)
    uuid = str( uuid.uuid4().hex)
    dst_wav_path = Path(dst_dir,Path(checkpoint_path).stem+'_'+uuid+'.wav')
    librosa.output.write_wav(dst_wav_path, wavenet_wav, sr=wavenet_hparams.hparams.sample_rate)
    
    print('done, check ' +Path(dst_wav_path).name)

    ##################################
    #    waveglow VOCODER part
    ##################################
    # save nvidia_waveglowpyt_fp32_20190427 to ~/.cache/torch/checkpoint/ 
    # in current project, NVIDIA waveglow frame size seems didn't equeal to r9y9 TTS output mel frame size
    # disable this function first
    '''
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval()
    with torch.no_grad():
        tensor_mel = torch.tensor(melx.T).unsqueeze(0).to('cuda') # [m,80] --> [1,80,m] nvidia waveglow input format
        glow_audio = waveglow.infer(tensor_mel)
    audio.save_wav(glow_audio[0].data.cpu().numpy(), dst_dir+'r9y9_waveglow_output.wav')        
    '''
    sys.exit(0)
