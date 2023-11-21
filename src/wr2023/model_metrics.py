import numpy as np 
import zarr 
from utils import InfoReader, report_model_metrics
import sys 

DST_PATH = sys.argv[1]
INFO_PATH = sys.argv[2]

root = zarr.open(DST_PATH, mode='r+')

ir = InfoReader(INFO_PATH)
sc = ir.selection_criteria()
for n in sc:
    
    gt_group = root[f'{n}/ground_truth/']
    mv_group = root[f'{n}/model_validation/']

    gt_beat_matrix = gt_group['beat_matrix'][:]
    mv_beat_matrix = mv_group['beat_matrix'][:]
    n_beats, window = gt_beat_matrix.shape
    bad_beats = gt_group['bad_beats'][:]

    metrics = np.empty((n_beats, 5))

    # compute cc metric
    metrics[:,0] = np.corrcoef(
        gt_beat_matrix,
        mv_beat_matrix
    ).diagonal(n_beats)

    # compute rdms metric
    metrics[:,1] = np.sqrt(
        np.square(
            gt_beat_matrix/np.linalg.norm(gt_beat_matrix, axis=1)[:, np.newaxis] -
            mv_beat_matrix/np.linalg.norm(mv_beat_matrix, axis=1)[:, np.newaxis]
        ).sum(axis=1)
    )

    # compute rmse metric
    metrics[:,2] = np.sqrt(
        np.square(
            gt_beat_matrix -
            mv_beat_matrix
        ).mean(axis=1)
    )

    # compute mae metric
    metrics[:,3] = np.abs(
            gt_beat_matrix -
            mv_beat_matrix
    ).mean(axis=1)

    # compute re metric
    metrics[:,4] = np.linalg.norm(gt_beat_matrix - mv_beat_matrix, axis=1) / np.linalg.norm(gt_beat_matrix, axis=1)
    # metrics[:,4] = np.sqrt(
    #     np.square(
    #         gt_beat_matrix -
    #         mv_beat_matrix
    #     ).sum(axis=1) / np.square(gt_beat_matrix).sum(axis=1)
    # )
    
    # force nan in bad beats
    metrics[bad_beats] = np.nan
    # save metrics
    mv_group.array('metrics', data=metrics, overwrite=True)

    print(f'register completed: {n}')

report_model_metrics(DST_PATH, INFO_PATH)