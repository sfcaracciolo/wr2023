import numpy as np 
import zarr 
from utils import InfoReader
import ecg_tools 
import sys 

ROOT_PATH = sys.argv[1]
fs = float(sys.argv[2])
INFO_PATH = sys.argv[3]

root = zarr.open(ROOT_PATH, mode='r+')

max_size = 0 
ecgs = []

ir = InfoReader(INFO_PATH)
ic = ir.inclusion_criteria()
for i, n in enumerate(ic):

    group = root[f'{n}']

    # filtering
    ecg = group['raw'][:,1] # pick d2
    ecg[:] = ecg_tools.highpass_filter(ecg, 4, 2, fs)
    ecg[:] = ecg_tools.lowpass_filter(ecg, 4, 300, fs)
    ecg[:] = ecg_tools.adapt_notch_filter(ecg, fs, 50, 1e-3)
    ecg[:] = ecg_tools.isoline_correction(ecg, 'numpy', bins=1000)

    # save size of max ecg and ecg
    max_size = max(max_size, ecg.size)
    ecgs.append(ecg)

    print(f'register completed: {n}({i:03})')

# save all ecgs
all_ecgs = np.zeros((len(ic), max_size), dtype=np.float32)
for i, ecg in enumerate(ecgs):
    all_ecgs[i, :ecg.size] = ecg

root.array(
    'ic_filtered_ecgs',
    data=all_ecgs,
    overwrite=True
)

