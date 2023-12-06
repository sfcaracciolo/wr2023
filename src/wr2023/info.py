import utils 
import zarr 
from wr_transform import TransformModel, TransformParameters

INFO_PATH = 'E:/wrdb/raw/info.csv'
ROOT_PATH = 'E:/wrdb/wr2023.zarr'
# utils.report_transform_metrics(ROOT_PATH, INFO_PATH)
utils.validation_beats(ROOT_PATH, INFO_PATH)
ir = utils.InfoReader.stats('E:/wrdb/raw/info.csv', MAT_PATH=None)
# utils.total_stats(ROOT_PATH, INFO_PATH)
K = TransformParameters(
    P=TransformParameters.kP(.95, .05),
    R=3.0,
    S=2.5,
    T=TransformParameters.kT(.8, .4),
    W=1.0,
    D=2.0,
    J=TransformParameters.kJ()
)
print(K)