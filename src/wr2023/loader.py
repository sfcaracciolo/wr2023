"""
Load raw data from *.mat files and create a zarr struct.
"""
import scipy as sp
import pathlib
from glob import glob
import zarr 
import numpy as np 
import sys 

ROOT_PATH = sys.argv[1]
SRC_PATH = sys.argv[2]

root = zarr.open(ROOT_PATH, mode='a')
paths = glob(SRC_PATH + '/*.mat')
for f in sorted(paths):
    _, n, _ = pathlib.Path(f).stem.split('_')
    group = root.create_group(
        f'{n}',
        overwrite=True
        )
    fmat = sp.io.loadmat(f, simplify_cells=True)['d']
    ecg = fmat['RowDATA']*fmat['Pendiente'] + fmat['OrdenadaAlOrigen']
    group.array('raw', data=ecg, dtype=np.float32, overwrite=True)
    print('register completed: ', n)
