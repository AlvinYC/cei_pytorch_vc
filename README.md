# cei_pytorch_vc


# docker build/run
```
    docker build -t r9y9tts:torch0.3 .

     docker run -it 
        --rm 
        --gpus=1 
        --shm-size=256m 
        -v /home/ailab/Documents/alvin_doc/VoiceConv/LJSpeech-1.1:/home/docker/LJSpeech
        -v /media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min:/home/docker/VCTK_r9y9 
        -v /media/alvin/HD/eval_output:/home/docker/cei_pytorch_vc/deepvoice3_0421749/eval_output
        -p 7788:22 
        -p 8888:8888
        --name deepvoice 
        r9y9tts:torch0.3
 ```

# traing LJSpeech
    
## prepare dataset
```
data structure of /home/ailab/Documents/alvin_doc/VoiceConv/LJSpeech-1.1
./LJSpeech-1.1
├── metadata.csv
├── README
└── wavs
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    ├── LJ001-0003.wav
        ...
        ...
    ├── LJ050-0276.wav
    ├── LJ050-0277.wav
    └── LJ050-0278.wav

 ```

 ## training step 1, wav2npy

 ```
    python preprocess.py ljspeech \
                         /home/docker/LJSpeech/ \
                         ./data/ljspeech \
 ```

 ## training step 2, traing

 ```
    python train.py --data-root=./data/ljspeech/ \
                    --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech"\
 ```

# inference/synthesis 

```
   python synthesis.py ./checkpoints/20210412_cei_ljspeech_checkpoint_step001630000.pth \
                       ./demostring.txt \
                       ./checkpoints \
                       --hparams "builder=deepvoice3,preset=deepvoice3_ljspeech \
```

# adaptation
> cei_util_wav.py: generate sub-dataset accoording to total duration limiataion such that each speaker can only have totally 10min data\
> preprocess.py: generate npy by wav, generate train.txt
```
python preprocess.py vctk ./VCTK_r9y9 ./VCTK_r9y9/data
                     [name] [input dir] [output dir]

./VCTK_r9y9_10min
├── speaker-info.txt         <--- [1] copy from VCTK 0.92
├── data                     <--- [4] output of preprocess.py, training data by this folder  
│   ├── vctk-mel-00001.npy
│   ├── vctk-mel-00002.npy
│   ├── vctk-spec-00001.npy
│   ├── vctk-spec-00002.npy
│   ├── ....
│   └── train.txt
├── txt                      <--- [3] cei_util_wav.py copy txt files according to [2] result
│   ├── p225
│   │   ├── p225_001.txt
│   │   ├── p225_002.txt
│   │   ├── p225_003.txt
│   │   ├── ....
│   │   └── p225_273.txt
│   ├── p226
│   ├── p227
│   ├── ...
│   ├── p376
│   └── s5
└── wav48                    <--- [2] cei_util_wav.py get wav files according to total durationn limitation
    ├── p225
    │   ├── p225_001.wav
    │   ├── p225_002.wav
    │   ├── p225_003.wav
    │   ├── ....
    │   └── p225_273.wav
    ├── p226
    ├── p227
    ├── ...
    ├── p376
    └── s5
```

 > symbolic link ./VCTK_r9y9/data to train.py path as ./vctk\
 > ln -s /VCTK_r9y9 ./vctk

```
python train.py --data-root=./vctk\
   --checkpoint-dir=checkpoints_vctk_adaptation\
   --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech"\
   --log-event-path=log/deepvoice3_vctk_adaptation\
   --restore-parts=./checkpoints/20210412_cei_ljspeech_checkpoint_step001630000.pth\
   --speaker-id=2
```

