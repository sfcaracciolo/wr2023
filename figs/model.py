from geometric_plotter import Plotter 
import zarr 
from src.wr2023 import utils
import numpy as np 

def set_violin_style(d):
    for k, v in d.items():
        if not isinstance(v, list):
            v.set_facecolors('gray')
            v.set_edgecolors('black')
        else:
            for p in v:
                p.set_facecolor('gray')
                p.set_edgecolor('black')
        
root = zarr.open('E:\wrdb\data.zarr', mode='r')
ir = utils.InfoReader('E:\wrdb\info.xlsx')
males = [v for v in ir.get_males() if v in utils.selection_criteria()]
females = [v for v in ir.get_females() if v in utils.selection_criteria()]

for group in [males, females]:
    ccs, res = [], []

    for n in group:
        gt_group = root[f'{n}/ground_truth']
        mv_group = root[f'{n}/model_validation']
        bad_beats = gt_group['bad_beats'][:]

        cc = np.delete(mv_group['metrics'][:, 0], bad_beats, axis=0)
        ccs.append(cc)

        re = np.delete(mv_group['metrics'][:, 4], bad_beats, axis=0)
        res.append(re)

    pos_ticks = [i + 1 for i in range(len(group))]

    for ms, ylims in zip([ccs, res],[(None, 1), (0, None)]):

        p = Plotter(_2d=True, figsize = (12, 5))
        v = p.axs.violinplot(ms, vert=True)
        p.axs.set_ylim(ylims)
        set_violin_style(v)

        # quantiles
        full = np.concatenate(ms)
        q1 = np.quantile(full, .25)
        q2 = np.quantile(full, .5)
        q3 = np.quantile(full, .75)
        p.axs.set_xticks(pos_ticks, labels=[i for i in group])
        p.axs.axhline(q1, linewidth=.4, color='gray', linestyle='dashed')
        p.axs.axhline(q2, linewidth=.4, color='gray', linestyle='dashed')
        p.axs.axhline(q3, linewidth=.4, color='gray', linestyle='dashed')

Plotter.show()