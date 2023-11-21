
import subprocess

SRC_PATH = 'E:/wrdb/raw'
DST_PATH = 'E:/wrdb/wr2023.zarr'
INFO_PATH = 'E:/wrdb/raw/info.csv'
GT_DELI_PATH = 'E:/wrdb/231102_Delineadas_Reales/'
TR_DELI_PATH = 'E:/wrdb/231108_Delineadas_Sinteticas/'
fs = '1024'

# print('loader ... (save .zarr)')
# subprocess.call(['python', 'src/wr2023/loader.py', DST_PATH, SRC_PATH], shell=True)

# print('\nfiltering ...')
# subprocess.call(['python', 'src/wr2023/filtering.py', DST_PATH, fs, INFO_PATH], shell=True)

# print('\ntrimming ... (save .mat)')
# subprocess.call(['python', 'src/wr2023/trimming.py', DST_PATH, INFO_PATH], shell=True)

# print('\ndscm gt ... (require .mat)"')
# subprocess.call(['python', 'src/wr2023/DSCM.py', DST_PATH, GT_DELI_PATH, 'ground_truth', fs, INFO_PATH], shell=True)

# print('\nfitting ...')
# subprocess.call(['python', 'src/wr2023/model_fitting.py', DST_PATH, INFO_PATH], shell=True)

# print('\nmodel metrics ...')
# subprocess.call(['python', 'src/wr2023/model_metrics.py', DST_PATH, INFO_PATH], shell=True)

# print('\ninverse transform ... (save .mat)')
# subprocess.call(['python', 'src/wr2023/transform_computing.py', DST_PATH, fs, INFO_PATH], shell=True)

# print('\ndscm tr ... (require .mat)"')
# subprocess.call(['python', 'src/wr2023/DSCM.py', DST_PATH, TR_DELI_PATH, 'transform_validation', fs, INFO_PATH], shell=True)