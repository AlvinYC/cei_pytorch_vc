from pathlib import Path
from scipy.io import wavfile
from datetime import timedelta
from glob import glob
import numpy as np
import shutil
import re

def get_duration_info(target_dir):
    p = Path(target_dir).glob('./*')
    files = [x for x in p if x.is_file()]
    sample_info = list(map(lambda x: wavfile.read(x), files)) #  sr, # of sample
    sec_info = list(map(lambda x: len(x[1])/x[0], sample_info))

    print(target_dir + ' have totally seconds: ' + str(timedelta(seconds=sum(sec_info))))
    return sample_info

def select_wav(target_dir,max_sec):
    p = Path(target_dir).glob('./*')
    files = [x for x in p if x.is_file()]
    sample_acc = 0
    select_files = []
    for wav_file in sorted(files):
        sr, sample = wavfile.read(wav_file)
        sample_acc += len(sample)
        if sample_acc > (max_sec*sr):
            return select_files
        else:
            select_files.append(wav_file)
    return select_files

def copy_sel(vctkdirs, cp_path, max_sec):
    sel_files = []
    for vctk in vctkdirs:
        sel_wavs = select_wav(vctk,max_sec)   # '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48/p225/'
        vctk_name = Path(vctk).name         # 'p225'
        Path.mkdir(Path(cp_path,vctk_name), parents=True, exist_ok=True) # '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_05min/wav48/p225'
        list(map(lambda x: shutil.copy(x, Path(cp_path,vctk_name)),sel_wavs))
        sel_files.extend(sel_wavs)
    return sel_files

def copy_sel_txt(vctkdirs, cp_path, sel_files):
    for vctk in vctkdirs:
        
        vctk_name = Path(vctk).name
        if vctk_name == 'p315': continue
        vctk_filter = list(filter(lambda x: re.search(vctk_name, x.name), sel_files))   # [PosixPath('/media/alvin/HD/dataset_audio/raw/VCTK_0.92_wav/wav48/p225/p225_001.wav'), ....]
        vctk_filter = list(map(lambda x: re.sub('wav$','txt',x.name), vctk_filter))         # [p225_001.txt, p225_002.txt, ...]

        vctk_txt_path = Path(Path(vctk).parent.parent, 'txt')                                               # PosixPath('/media/alvin/HD/dataset_audio/raw/VCTK_0.92_wav/txt')
        cp_txt_path = Path(Path(cp_path).parent,'txt')                                                           # PosixPath('/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_05min/txt')
        Path.mkdir(Path(cp_txt_path,vctk_name), parents=True, exist_ok=True)
        list(map(lambda x:shutil.copy(Path(vctk_txt_path,vctk_name,x), Path(cp_txt_path,vctk_name)),vctk_filter))
        #sel_txt = list(map(lambda x: Path(vctk.parent,'txt'
        #list(map


if __name__== "__main__":
    # query duration for each vctk folder
    #vctk_path = '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48'
    vctk_path = '/media/alvin/HD/dataset_audio/raw/VCTK_0.92_wav/wav48'
    vctkdirs = sorted(glob(vctk_path+"/*/"))    # ['./p225/', './p226/', './p227/', './p376/']
    print(str(vctkdirs)+'\n')
    list(map(lambda x:get_duration_info(x), vctkdirs))

    # copy with limitation 
    cp_path = '/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48'
    max_sec = 600
    sel_files = copy_sel(vctkdirs,cp_path,max_sec)                      # ['/media/alvin/HD/dataset_audio/raw/VCTK_r9y9_10min/wav48/p225/',...]

    # copy txt according to filtered wav
    copy_sel_txt(vctkdirs ,cp_path,sel_files)

    # query duration for each new vctk folder    
    vctkdirs = sorted(glob(cp_path+'/*/')) 
    print(str(vctkdirs)+'\n')
    list(map(lambda x:get_duration_info(x), vctkdirs))
    


