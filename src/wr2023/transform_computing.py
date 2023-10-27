import numpy as np 
from ecg_models.Rat import Waves, f
from ecg_models.utils import modelize
from fpt_tools import FPTMasked
import utils
import zarr 
from wr_transform import TransformModel, TransformParameters
import sys 

DST_PATH = sys.argv[1]
fs = float(sys.argv[2])

root = zarr.open(DST_PATH, mode='r+')

K = TransformParameters(
    P=1.6,
    R=3.0,
    S=2.5,
    T=2.6891,
    W=1.0,
    D=2.0
)

model = lambda x, fea: f(x, modelize([0]+fea.tolist(), Waves))
tr_model = TransformModel(K, model)

for n in utils.selection_criteria():

    tr_group = root.create_group(
        f'{n}/transform_validation/',
        overwrite=True
        )
    
    gt_group = root[f'{n}/ground_truth/']
    n_beats, window = gt_group['beat_matrix'].shape
    bad_beats = gt_group['bad_beats'][:].tolist()
    measurements = gt_group['measurements'][:]

    # beat_matrix only for dummy beat
    beat_matrix = gt_group['beat_matrix'][:]
    fpt = gt_group['fpt'][:]
    fpt_model = FPTMasked(fpt, [4], beat_matrix)
    dummy_beat = fpt_model.template_beat()

    synthetic_beat_matrix = np.empty_like(beat_matrix)
    θ = np.linspace(0, 2*np.pi, num=window)

    for i in range(n_beats):

        if i in bad_beats:
            synthetic_beat_matrix[i] = dummy_beat
            continue
        
        mea = measurements[i]
        mea[:4] *= fs * 2*np.pi / window # time 2 rad conversion
        features = tr_model.inverse(mea)
        synthetic_beat_matrix[i] = model(θ, features)

    # save model features
    tr_group.array(
        'beat_matrix',
        data=synthetic_beat_matrix,
        overwrite=True)

    # save ravel ecg
    tr_group.array(
        'ecg',
        data=np.ravel(synthetic_beat_matrix, order='c'),
        overwrite=True)
    
    print('register completed: ', n)

utils.export_matfiles(DST_PATH, 'transform_validation')