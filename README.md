# cei_pytorch_vc


# docker build/run
```
    docker build -t deepvoice3:latest .

     docker run -it --rm --gpus=1 --shm-size=256m -v /home/ailab/Documents/alvin_doc/VoiceConv/LJSpeech-1.1:/home/docker/LJSpeech --name deepvoice --net=host deepvoice3:latest
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
```
python preprocess.py vctk ./VCTK_r9y9 ./VCTK_r9y9/data
                     [name] [input dir] [output dir]

./VCTK_r9y9                     <-- input dir
├── data                        <-- output dir
|   | [npy]
|   └── train.txt
├── speaker-info.txt            <--- [1] copy from VCTK 0.92
├── txt                         <--- [2] copy from VCTK 0.92
|   ├── p225 
|   |    ├── p225_001.txt
|   |    ├── p225_002.txt
|   |    └── p225_003.txt
|   ├── p226
    .....
|   ├── p376
|   └──s5
└── wav48                       <--- [3] copy from VCTK 0.92
    ├── p225
    |    ├── p225_001.wav
    |    ├── p225_002.wav
    |    └── p225_003.wav
    ├── p226
    └── p227
```

 > symbolic link ./VCTK_r9y9/data to train.py path as ./vctk
 > ln -s /VCTK_r9y9 ./vctk

```
python train.py --data-root=./vctk\
   --checkpoint-dir=checkpoints_vctk_adaptation\
   --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech"\
   --log-event-path=log/deepvoice3_vctk_adaptation \
   --restore-parts=./checkpoints/20210412_cei_ljspeech_checkpoint_step001630000.pth\ 
   --speaker-id=2
```

