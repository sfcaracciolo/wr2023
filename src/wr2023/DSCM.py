import scipy as sp 
import numpy as np 
import ecg_tools 
from fpt_tools import FPTMasked
import zarr 
import utils
import sys

# DCSM: Delination / Segmentation / Criteria / Measurements

DST_PATH = sys.argv[1]
DELI_PATH = sys.argv[2]
GROUP_NAME = sys.argv[3]
fs = float(sys.argv[4])

root = zarr.open(DST_PATH, mode='a')

for n in utils.selection_criteria():

    group = root[f'{n}/{GROUP_NAME}/']

    # delineation 
    fpt = sp.io.loadmat(DELI_PATH + f'Delineada_Reg_{n}.mat', simplify_cells=True)['Resultados']
    fpt -= 1 # numpy zero based indexing
    fpt[:, 9] = fpt[:, 7] # QRSend = Ton in WR
    fpt = np.nan_to_num(fpt, nan=2147483647, copy=True).astype(np.int32)
    group.array(
        'fpt',
        data=fpt,
        overwrite=True
        )
    
    # window size
    r_pos = fpt[:, 5]
    RR = np.diff(r_pos)
    window_size = np.array([sp.stats.mode(RR, nan_policy='raise').mode], dtype=np.int32) 

    # beat matrix
    ecg = group['ecg'][:]
    beat_matrix = ecg_tools.beat_matrix(ecg, r_pos, window_size)
    group.array(
        'beat_matrix',
        data=beat_matrix,
        overwrite=True
        )

    # beat selection criteria
    model = FPTMasked(fpt, [4], beat_matrix) # remove Q for WR
    ofw_amount = model.set_out_of_window_mask() # excluding beats with any fiducial out of window
    ifm_amount = model.set_invalid_fiducial_mask() # excluding beats with any nan in valid cols
    cc_amount = model.set_lesser_than_cc_mask(.90)
    bad_beats = model.beat_matrix.get_masked_indices() # take indexes of excluded beats
    ix = group.array(
        'bad_beats',
        data=bad_beats,
        overwrite=True
        )
    ix.attrs['out of window mask'] = ofw_amount 
    ix.attrs['invalid fiducial mask'] = ifm_amount
    ix.attrs['lesser cc mask'] = cc_amount

    # measurements
    model.set_measurements(fs)
    group.array(
        'measurements', 
        data=model.measurements.data,
        overwrite=True
        )
    
    print('register completed: ', n)

utils.report_beat_criteria(DST_PATH, GROUP_NAME)

if GROUP_NAME == 'transform_validation':
    utils.report_transform_metrics(DST_PATH)