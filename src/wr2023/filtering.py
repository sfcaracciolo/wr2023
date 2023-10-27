import scipy as sp 
import numpy as np 
import zarr 
import utils
import ecg_tools 
import sys 

DST_PATH = sys.argv[1]
SRC_PATH = sys.argv[2]
fs = float(sys.argv[3])

root = zarr.open(DST_PATH, mode='a')

max_size = 0 
ecgs = []

SELECTION = utils.selection_criteria()
for n in SELECTION:

    group = root.create_group(
        f'{n}/ground_truth/',
        overwrite=True
        )

    # filtering
    fmat = sp.io.loadmat(SRC_PATH + str(n) + '.mat', simplify_cells=True)
    ecg = fmat['DII']
    ecg[:] = ecg_tools.highpass_filter(ecg, 4, 2, fs)
    ecg[:] = ecg_tools.lowpass_filter(ecg, 4, 300, fs)
    ecg[:] = ecg_tools.adapt_notch_filter(ecg, fs, 50, 1e-3)
    ecg[:] = ecg - ecg_tools.isoline_correction(ecg[int(ecg.size*.1):int(ecg.size*.9)], 'numpy', bins=100, return_isoline=True)
    group.array(
        'ecg',
        data=ecg,
        overwrite=True)

    # save size of max ecg and ecg
    max_size = max(max_size, ecg.size)
    ecgs.append(ecg)

    print('register completed: ', n)

# save all ecgs
all_ecgs = np.zeros((len(SELECTION), max_size), dtype=np.float32)
for i, ecg in enumerate(ecgs):
    all_ecgs[i, :ecg.size] = ecg

root.array(
    'all',
    data=all_ecgs,
    overwrite=True
)

utils.export_matfiles(DST_PATH, 'ground_truth')
