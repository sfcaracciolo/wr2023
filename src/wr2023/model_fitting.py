import scipy as sp 
import numpy as np 
from ecg_models.Rat import Waves, f
from ecg_models.waves import BeatFeatures, FunFeatures
from ecg_models.utils import FeatureEditor, vectorize, fvec, modelize
from fpt_tools import FPTMasked
import zarr 
from utils import InfoReader 
import sys 

DST_PATH = sys.argv[1]
INFO_PATH = sys.argv[2]

root = zarr.open(DST_PATH, mode='r+')

def init_estimation(fid, beat):

    # features defined in samples
    base_fe = BeatFeatures(
        RR = 0, # irrelevant feature for fitting
        Waves= Waves(
            P = FunFeatures(
                a=beat[fid[1]],
                μ=fid[1],
                σ=fid[2]-fid[0]
            ),
            R = FunFeatures(
                a=beat[fid[5]],
                μ=fid[5],
                σ=fid[7]-fid[3]
            ),
            S = FunFeatures(
                a=beat[fid[6]],
                μ=fid[6],
                σ=fid[7]-fid[3]
            ),
            T = FunFeatures(
                a=beat[fid[10]],
                μ=fid[10],
                σ=fid[11]-fid[9]
            ),
        )
    )

    # temporal features scaled to rad for fitting
    editor = FeatureEditor(base_fe)
    editor.scale(2*np.pi/beat.size, 'μ' )
    editor.scale(2*np.pi/beat.size, 'σ' )
    editor.scale(.5, 'σ' )
    init_fe = editor.get_feature()

    # factors
    af, μf, σf = .5, .15, .8

    # inf bounds
    editor = FeatureEditor(init_fe)
    editor.collapse(af, 'a')
    editor.scale(1-μf, 'μ')
    editor.scale(1-σf, 'σ')
    inf_fe = editor.get_feature()

    # sup bounds
    editor = FeatureEditor(init_fe)
    editor.expand(af, 'a')
    editor.scale(1+μf, 'μ')
    editor.scale(1+σf, 'σ')
    sup_fe = editor.get_feature()

    inf_vfe = vectorize(inf_fe)
    init_vfe = vectorize(init_fe)
    sup_vfe = vectorize(sup_fe)
    return init_vfe[1:], (inf_vfe[1:], sup_vfe[1:])

def model(θ, *args):
    return fvec(f, Waves, θ, 0, *args)

ir = InfoReader(INFO_PATH)
sc = ir.selection_criteria()
for n in sc:
    
    mv_group = root.create_group(
        f'{n}/model_validation/',
        overwrite=True
        )
    
    gt_group = root[f'{n}/ground_truth/']

    bad_beats = gt_group['bad_beats'][:].tolist()
    beat_matrix = gt_group['beat_matrix'][:]
    fpt = gt_group['fpt'][:]
    fpt_model = FPTMasked(fpt, [4], beat_matrix)
    n_beats, window = beat_matrix.shape
    
    θ = np.linspace(0, 2*np.pi, num=window)
    dummy_beat = fpt_model.template_beat()

    features = np.empty((n_beats, 3*4))
    fitted_beat_matrix = np.empty_like(beat_matrix)

    for i in range(n_beats):

        if i in bad_beats:
            features[i] = np.nan
            fitted_beat_matrix[i] = dummy_beat
            continue
        
        # init features estimation from measurements
        init_features, bounds = init_estimation(fpt_model.cfpt[i], beat_matrix[i])

        # make fitting model
        features[i], _ = sp.optimize.curve_fit(
            model,
            θ,
            beat_matrix[i],
            p0 = init_features,
            bounds = bounds,
            max_nfev = 200 * window
        )

        fitted_beat_matrix[i] = f(θ, modelize([0]+features[i].tolist(), Waves))

    # save beat matrix
    mv_group.array(
        'beat_matrix',
        data=fitted_beat_matrix,
        overwrite=True)
    
    print('register completed: ', n)