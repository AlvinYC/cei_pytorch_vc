from pathlib import Path
from scipy.io import wavfile
from datetime import timedelta
from glob import glob
import numpy as np


def get_duration_info(target):
    p = Path(r'./').glob(target + '/*')
    files = [x for x in p if x.is_file()]
    sample_info = list(map(lambda x: wavfile.read(x), files)) #  sr, # of sample
    sec_info = list(map(lambda x: len(x[1])/x[0], sample_info))

    print(target + ' have totally seconds: ' + str(timedelta(seconds=sum(sec_info))))


if __name__== "__main__":
    dirlist = sorted(glob("./*/"))
    print(str(dirlist)+'\n')
    list(map(lambda x:get_duration_info(x), dirlist))

