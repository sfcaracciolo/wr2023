import numpy as np 
import zarr 
from utils import InfoReader, export_matfiles
import ecg_tools 
import sys 

ROOT_PATH = sys.argv[1]
INFO_PATH = sys.argv[2]

root = zarr.open(ROOT_PATH, mode='r+')

ir = InfoReader(INFO_PATH)
ic = ir.inclusion_criteria()
ec = ir.exclusion_criteria_after_filtering()
for i, n in enumerate(ic):

    if n in ec:
        print(n)
        continue

    group = root[f'{n}']
    gt_group = group.create_group(
        'ground_truth',
        overwrite=True
        )

    # trimming
    start, end = ir.trim_value(n)
    if end == -1: end = group['raw'].shape[0]
    ecg = root['ic_filtered_ecgs'][i,start:end]

    gt_group.array(
        'ecg',
        data=ecg,
        overwrite=True)

    print(f'register completed: {n}({i:03})')

export_matfiles(ROOT_PATH, INFO_PATH, 'ground_truth')
