import numpy as np
from scipy.io.wavfile import write
import torch
from IPython.display import Audio

tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

text = "We are in the business of collaborative innovation and making those ideas happen. Building an innovative product can be challenging."

# preprocessing
sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

# run the models
with torch.no_grad():
    _, mel, _, _ = tacotron2.infer(sequence)
    audio = waveglow.infer(mel) # mel.shape = torch.Size([1,80,854])
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050

write("/eval_output/audio.wav", rate, audio_numpy)

Audio(audio_numpy, rate=rate)
print("mission completed")
