
import subprocess

DST_PATH = 'E:/wrdb/data.zarr'
SRC_PATH = 'E:/wrdb/raw/'
GT_DELI_PATH = 'E:/Delineadas_Reales_18_07_22/'
TR_DELI_PATH = 'E:/Transform_Validation_28_07_22/'
fs = '1024'

print('filtering ... (save .mat)')
subprocess.call(['python', 'src/wr2023/filtering.py', DST_PATH, SRC_PATH, fs], shell=True)

print('\ndscm gt ... (require .mat)"')
subprocess.call(['python', 'src/wr2023/DSCM.py', DST_PATH, GT_DELI_PATH, 'ground_truth', fs], shell=True)

print('\nfitting ...')
subprocess.call(['python', 'src/wr2023/model_fitting.py', DST_PATH], shell=True)

print('\nmodel metrics ...')
subprocess.call(['python', 'src/wr2023/model_metrics.py', DST_PATH], shell=True)

print('\ninverse transform ... (save .mat)')
subprocess.call(['python', 'src/wr2023/transform_computing.py', DST_PATH, fs], shell=True)

print('\ndscm tr ... (require .mat)"')
subprocess.call(['python', 'src/wr2023/DSCM.py', DST_PATH, TR_DELI_PATH, 'transform_validation', fs], shell=True)