import utils 
import zarr 

INFO_PATH = 'E:/wrdb/raw/info.csv'
ROOT_PATH = 'E:/wrdb/wr2023.zarr'
# utils.report_transform_metrics(ROOT_PATH, INFO_PATH)
# utils.validation_beats(ROOT_PATH, INFO_PATH, group_name='ground_truth')
# ir = InfoReader.stats('E:/wrdb/raw/info.csv', MAT_PATH=None)
utils.total_stats(ROOT_PATH, INFO_PATH)
