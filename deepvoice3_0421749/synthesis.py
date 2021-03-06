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

use_cuda = torch.cuda.is_available()
_frontend = None  # to be set later


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
    if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
        checkpoint = torch.load(checkpoint_seq2seq_path)
        model.seq2seq.load_state_dict(checkpoint["state_dict"])
        checkpoint = torch.load(checkpoint_postnet_path)
        model.postnet.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
    else:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)
    melx = np.array([])
    #specx = np.array([])
    waveformx = np.array([])
    sec2numpy = lambda sec: np.zeros(int(hparams.sample_rate * sec))

    with open(text_list_file_path, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            text = line.decode("utf-8")[:-1]
            words = nltk.word_tokenize(text)
            #waveform, alignment, _, _ = tts(
            waveform, alignment, _, mel = tts(
                model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True)
            # concatenate ecah wave/mel  
            melx = np.array(mel,copy=True) if idx == 0 else np.concatenate((melx,mel),axis=0)
            waveformx = np.array(waveform,copy=True)  if idx == 0 else np.concatenate((waveformx,sec2numpy(0.5), waveform),axis=0)
            # save each utterance
            dst_wav_path = join(dst_dir, "{}_{}{}.wav".format(idx, checkpoint_name, file_name_suffix))
            dst_alignment_path = join(dst_dir, "{}_{}{}_alignment.png".format(idx, checkpoint_name,file_name_suffix))
            plot_alignment(alignment.T, dst_alignment_path,info="{}, {}".format(hparams.builder, basename(checkpoint_path)))
            audio.save_wav(waveform, dst_wav_path)
            from os.path import basename, splitext
            name = splitext(basename(text_list_file_path))[0]
            if output_html:
                print("""
{}

({} chars, {} words)

<audio controls="controls" >
<source src="/audio/{}/{}/{}" autoplay/>
Your browser does not support the audio element.
</audio>

<div align="center"><img src="/audio/{}/{}/{}" /></div>
                  """.format(text, len(text), len(words),
                             hparams.builder, name, basename(dst_wav_path),
                             hparams.builder, name, basename(dst_alignment_path)))
            else:
                print(idx, ": {}\n ({} chars, {} words)".format(text, len(text), len(words)))

    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    audio.save_wav(waveformx, dst_dir+'r9y9_output.wav')

    ##################################
    #    wavenet VOCODER part
    ##################################
    #rescaling_max = 0.999
    waveformx = waveformx / np.abs(waveformx).max() * wavenet_hparams.hparams.rescaling_max
    out=waveformx
    constant_values = 0.0
    out_dtype = np.float32
    mel_spectrogram = wavenet_audio.melspectrogram(waveformx).astype(np.float32).T
    l, r = wavenet_audio.lws_pad_lr(waveformx, wavenet_hparams.hparams.fft_size, wavenet_audio.get_hop_size())
    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * wavenet_audio.get_hop_size()
    out = out[:N * wavenet_audio.get_hop_size()]
    assert len(out) % wavenet_audio.get_hop_size() == 0
    timesteps = len(out)

    # Write the spectrograms to disk:
    audio_filename = 'r9y9-tts-audio.npy' 
    mel_filename = 'r9y9-tts-mel.npy' 
    np.save(os.path.join(dst_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
    np.save(os.path.join(dst_dir, mel_filename), mel_spectrogram.astype(np.float32), allow_pickle=False)


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
