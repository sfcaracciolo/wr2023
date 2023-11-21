
from geometric_plotter import Plotter
import numpy as np
from src.wr2023 import utils
import zarr 

def pearson_plot(data1, data2):
    p = Plotter(_2d=True, figsize=(5,5))
    p.axs.scatter(data1, data2, s=3, color='k')

    # Add correlation line
    m, b = np.polyfit(data1, data2, 1)
    rho = np.corrcoef(data1, data2)[0,1]
    X_plot = np.linspace(*p.axs.get_xlim(), 100)
    p.axs.plot(X_plot, m*X_plot + b, '--', color='gray', linewidth=.5)
    p.axs.plot(X_plot, X_plot, '--', color='gray', linewidth=.5)
    p.axs.text(.1,.9,f'$\\rho = {rho:.3f}$', transform=p.axs.transAxes)
    p.axs.text(.1,.8,f'$m = {m:.3f}$', transform=p.axs.transAxes)

def bland_altman_plot(data1, data2):
    p = Plotter(_2d=True, figsize=(5,5))

    _diff = data1 - data2 # Difference between data1 and data2
    _sum = (data1 + data2)/2.
    md = np.mean(_diff) # Mean of the difference
    sd = np.std(_diff, axis=0) # Standard deviation of the difference

    p.axs.scatter(_sum, _diff, s=3, color='k')
    p.axs.axhline(md, color='gray', linestyle='--', linewidth=.5)
    p.axs.axhline(md + 1.96*sd, color='gray', linestyle='--', linewidth=.5)
    p.axs.axhline(md - 1.96*sd, color='gray', linestyle='--', linewidth=.5)
    p.axs.text(.1,.9,f'$\\mu = {md:.3f}$', transform=p.axs.transAxes)
    p.axs.text(.1,.8,f'$\\sigma = {sd:.3f}$', transform=p.axs.transAxes)

root = zarr.open('E:\wrdb\wr2023.zarr', mode='r')
ir = utils.InfoReader('E:\wrdb\\raw\info.csv')
sc = ir.selection_criteria()
gt_meas, tr_meas = [], []
for n in sc:
    gt_group = root[f'{n}/ground_truth']
    tr_group = root[f'{n}/transform_validation']
    bad_beats = np.union1d(gt_group['bad_beats'][:], tr_group['bad_beats'][:])

    gt_mea = np.delete(gt_group['measurements'][:], bad_beats, axis=0)
    gt_meas.append(1e3*gt_mea) # to mv and uV

    tr_mea = np.delete(tr_group['measurements'][:], bad_beats, axis=0)
    tr_meas.append(1e3*tr_mea) # to mv and uV

full_gt = np.vstack(gt_meas)
full_tr = np.vstack(tr_meas)

for col in [0, 1, 2, 3, 4, 6, 7, 8]: # P dur, PR, RS, QT, P, R, S, T
    
    data1 = full_gt[:, col]
    data2 = full_tr[:, col]

    p = pearson_plot(data1, data2)
    p = bland_altman_plot(data1, data2)

Plotter.show()
