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
    python preprocess.py ljspeech /home/docker/LJSpeech/ ./data/ljspeech
 ```

 ## training step 2, traing

 ```
    python train.py --data-root=./data/ljspeech/ --hparams="builder=deepvoice3,preset=deepvoice3_ljspeech"
 ```


 
