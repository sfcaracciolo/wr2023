from datetime import date
import statistics
from typing import Literal
import zarr 
import pathlib
import csv
import scipy as sp 
import numpy as np 

def validation_beats(ROOT_PATH, INFO_PATH):
    root = zarr.open(ROOT_PATH, mode='r')
    ir = InfoReader(INFO_PATH)
    sc = ir.selection_criteria()
    mv_nbeats, tr_nbeats = 0, 0
    for n in sc:
        tr_group = root[f'{n}/transform_validation']
        gt_group = root[f'{n}/ground_truth']
        mv_bad_beats = gt_group['bad_beats'].size
        tr_bad_beats = np.union1d(gt_group['bad_beats'][:], tr_group['bad_beats'][:]).size
        mv_nbeats += gt_group['fpt'].shape[0] - mv_bad_beats
        tr_nbeats += gt_group['fpt'].shape[0] - tr_bad_beats
    print(f'Total beats validation for model = {mv_nbeats}')
    print(f'Total beats validation for transform = {tr_nbeats}')

def total_stats(ROOT_PATH, INFO_PATH):
    root = zarr.open(ROOT_PATH, mode='r')
    ir = InfoReader(INFO_PATH)
    sc = ir.selection_criteria()
    metrics = []
    for n in sc:
        group = root[f'{n}/model_validation/']
        metrics.append(group['metrics'][:,[0,4]])
    
    data = np.vstack(metrics)
    Q1 = np.nanquantile(data, .25, axis=0)
    Q2 = np.nanquantile(data, .50, axis=0)
    Q3 = np.nanquantile(data, .75, axis=0)

    for q1, q2, q3 in zip(Q1,Q2,Q3):
        print(f'Q1 = {q1:.3f}, Q2 = {q2:.3f}, Q3 = {q3:.3f}')

def report_model_metrics(ROOT_PATH, INFO_PATH):
    root = zarr.open(ROOT_PATH, mode='r')

    HEADER = ['id', '# nbeats', '# nbad', 'cc', 're'] # 'rdms', 'rmse', 'mae']
    file = pathlib.Path(__file__).parent / 'reports' / 'model_metrics.csv'

    with open(file, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t')
        filewriter.writerow(HEADER)

        ir = InfoReader(INFO_PATH)
        sc = ir.selection_criteria()
        for n in sc:
            gt_group = root[f'{n}/ground_truth/']
            mv_group = root[f'{n}/model_validation/']

            metrics = mv_group['metrics'][:]
            gt_bad_beats = gt_group['bad_beats'][:]

            nbeats = gt_group['beat_matrix'].shape[0] # amount of beats
            nbad = gt_bad_beats.size # amount of bad beats

            cc = f'{np.nanmean(metrics[:,0]):.4f}({np.nanstd(metrics[:,0]):.4f})'
            re = f'{np.nanmean(metrics[:,4]):.4f}({np.nanstd(metrics[:,4]):.4f})'
            # rdms = f'{np.nanmean(metrics[:,1]):.4f}({np.nanstd(metrics[:,1]):.4f})'
            # rmse = f'{np.nanmean(metrics[:,2]):.4f}({np.nanstd(metrics[:,2]):.4f})'
            # mae = f'{np.nanmean(metrics[:,3]):.4f}({np.nanstd(metrics[:,3]):.4f})'

            filewriter.writerow([f'{n:03}', f'{nbeats:04}', f'{nbad:04}', cc, re ]) # rdms, rmse, mae, re])

def report_transform_metrics(ROOT_PATH, INFO_PATH):
    root = zarr.open(ROOT_PATH, mode='r')

    HEADER = ['id', '# nbeats', '# nbad', 'P', 'PR', 'RS', 'QT', 'P', 'R', 'S', 'T']
    file = pathlib.Path(__file__).parent / 'reports' / 'transform_metrics.csv'
    with open(file, 'w', newline='') as csvfile:

        filewriter = csv.writer(csvfile, delimiter='\t')
        filewriter.writerow(HEADER)

        ir = InfoReader(INFO_PATH)
        sc = ir.selection_criteria()
        for n in sc:
            gt_group = root[f'{n}/ground_truth']
            tr_group = root[f'{n}/transform_validation']
            bad_beats = np.union1d(gt_group['bad_beats'][:], tr_group['bad_beats'][:])

            nbeats = gt_group['beat_matrix'].shape[0] # amount of beats
            nbad = bad_beats.size # amount of bad beats

            gt_mea = np.delete(np.delete(gt_group['measurements'][:], bad_beats, axis=0), 5, axis=1)
            tr_mea = np.delete(np.delete(tr_group['measurements'][:], bad_beats, axis=0), 5, axis=1)

            # if runtime warning raised implies null std (measurements are equal in all beats)
            rho = np.corrcoef(gt_mea, y=tr_mea, rowvar=False).diagonal(8) 
            rho = np.round(rho, decimals=4)

            filewriter.writerow([f'{n:03}', f'{nbeats:04}', f'{nbad:04}', *rho.tolist() ]) # rdms, rmse, mae, re])

def report_beat_criteria(ROOT_PATH, INFO_PATH, group_name='ground_truth'):
    root = zarr.open(ROOT_PATH, mode='r')
    
    HEADER = ['id', 'nbeats', 'invalid fiducial mask', 'lesser p rate mask', 'lesser cc mask', 'out of window mask', 'lost [%]']
    file = pathlib.Path(__file__).parent / 'reports' / f'beat_criteria_of_{group_name}.csv'

    with open(file, 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter='\t')
        filewriter.writerow(HEADER)

        ir = InfoReader(INFO_PATH)
        sc = ir.selection_criteria()
        for n in sc:
            group = root[f'{n}/{group_name}/']
            bad_beats = group['bad_beats']
            n_beats = group['beat_matrix'].shape[0]
            att = bad_beats.attrs.items()
            row = [None]*len(HEADER)
            row[0] = f'{n:03}'
            row[1] = f'{n_beats:06}'
            acc = 0
            for k, v in att:
                ix = HEADER.index(k)
                row[ix] = f'{v:04}'
                acc += v
            row[-1] = f'{100 * acc / n_beats :.1f}'
            filewriter.writerow(row)

def export_matfiles(ROOT_PATH, INFO_PATH, group_name='ground_truth'):
    root = zarr.open(ROOT_PATH, mode='r')
    db = {}
    ir = InfoReader(INFO_PATH)
    if group_name == 'ground_truth':
        sc = ir.inclusion_criteria()
        for i in ir.exclusion_criteria_after_filtering():
            sc.remove(i)
    elif group_name == 'transform_validation':
        sc = ir.selection_criteria()
    else:
        raise ValueError('ERROR group name')

    for n in sc:
        group = root[f'{n}/{group_name}']
        ecg = group[f'ecg'][:]
        try:
            n_beats = group[f'{group_name}/beat_matrix'].shape[0]
        except:
            n_beats = -1
        db[f'Reg{n}'] = {'ecg': ecg, 'beats':n_beats}
    sp.io.savemat(f'{group_name}.mat', db)
class InfoReader:
    def __init__(self, CSV_PATH, MAT_PATH='E:/wrdb/raw/') -> None:
        # InfoReader.stats('E:/wrdb/raw/info.csv')
        self.fs = 1024
        self._data = {}
        with open(CSV_PATH, 'r', newline='') as f:
            table = csv.reader(f)
            next(table) # skip header
            for row in table:
                n = int(row[0])
                self._data[n] = {
                    'nrat': int(row[1]),
                    'birth date': date.fromisoformat(row[2]),
                    'recording date': date.fromisoformat(row[3]),
                    'castration date': date.fromisoformat(row[4]) if row[4] != '' else None,
                    'weight': int(row[5]),
                    'sex': row[6],
                    'protocol': row[7][0],
                    'comment': row[8],
                    'cable': int(row[7][1]),
                    'age': (date.fromisoformat(row[3]) - date.fromisoformat(row[2])).total_seconds() / 60 / 60 / 24 / 7,
                    'samples': sp.io.loadmat(MAT_PATH + f'RegECG_{n}_1.mat', simplify_cells=True)['d']['CantidadDeDatos'] if MAT_PATH is not None else None
                }

    def inclusion_criteria(self):
        control = self.get_protocol('C')
        cable = self.get_cable(2)
        noncas = self.get_nocastradas()
        return sorted(set(control).intersection(cable).intersection(noncas))

    def exclusion_criteria_after_filtering(self):
        # for i in [ 89, 92, 95, 107, 152, 155, 168, 173, 179, 208, 210, 237, 257, 265, 267, 273, 316, 325]: # see morphology
        #     sc.remove(i)
        return [ 
            257, # noisy
            264, # pathology
            325, # noisy 
        ]
    
    def exclusion_criteria_after_delineation(self):
        return [ 
            173, # low T wave
            66, # bad delineation
            107, # bad delineation
            179 # bad delineation
        ]
    
    def exclusion_criteria(self):
        return self.exclusion_criteria_after_filtering() + self.exclusion_criteria_after_delineation()
     
    def selection_criteria(self):
        sc = self.inclusion_criteria()
        for i in self.exclusion_criteria():
            sc.remove(i)
        return sc
    
    def filter_by(self, k, v, data=None, condition= lambda a, b: a == b):
        _d = self._data if data is None else data
        return [ nreg for nreg, item in _d.items() if condition(item[k], v)] 

    def get_castradas(self, data=None):
        return self.filter_by('castration date', None, data=data, condition=lambda a, b: a != b)

    def get_nocastradas(self, data=None):
        return self.filter_by('castration date', None, data=data)

    def get_cable(self, c: Literal[1, 2], data=None):
        return self.filter_by('cable', c, data=data)
    
    def get_protocol(self, p: Literal['C', 'A', 'P'], data=None):
        return self.filter_by('protocol', p, data=data)
    
    def get_males(self, data=None):
        return self.filter_by('sex', 'M', data=data)

    def get_females(self, data=None):
        return self.filter_by('sex', 'F', data=data)

    def get(self, lst):
        return { k:self._data[k] for k in lst} 
    
    def mean(self, data, key):
        return statistics.mean(map(lambda r: r[key], data.values()))
    
    def std(self, data, key):
        return statistics.stdev(map(lambda r: r[key], data.values()))
    
    def min(self, data, key):
        return min(map(lambda r: r[key], data.values()))
    
    def max(self, data, key):
        return max(map(lambda r: r[key], data.values()))
    
    def get_rats(self, data=None):
        _d = self._data if data is None else data
        rats = {}
        for k0, v0 in _d.items():
            for k1, v1 in rats.items():
                if (v0['nrat'] == v1['nrat']) and (v0['recording date'] == v1['recording date']) and (v0['sex'] == v1['sex']):
                    break
            else:
                rats[k0] = v0
        return rats
    
    @classmethod
    def stats(cls, CSV_PATH, MAT_PATH):
        ir = cls(CSV_PATH, MAT_PATH)

        rats = ir.get_rats()
        print(f"# ratas = {len(rats)}")
        print(f"\t# hembras = {len(ir.get_females(data=rats))}")
        print(f"\t# machos = {len(ir.get_males(data=rats))}")
        print(f"\t# castradas = {len(ir.get_castradas(data=rats))}")
        print(f"\t# no castradas = {len(ir.get_nocastradas(data=rats))}")


        print(f'# registros = {len(ir._data)}')
        print(f"\t# castradas = {len(ir.get_castradas())}")
        print(f"\t# no castradas = {len(ir.get_nocastradas())}")
        print(f"\t# control = {len(ir.get_protocol('C'))}")
        print(f"\t# atropina = {len(ir.get_protocol('A'))}")
        print(f"\t# propranolol = {len(ir.get_protocol('P'))}")
        print(f"\t# cable 1 = {len(ir.get_cable(1))}")
        print(f"\t# cable 2 = {len(ir.get_cable(2))}")
        print(f"\t# cable 3 = {len(ir.get_cable(3))}")

        # print(f"tiempo = {ir.min(ir._data, 'samples')/fs/60:.1f} a {ir.max(ir._data, 'samples')/fs/60:.1f} minutos\n")
        # print(f"tiempo = {ir.mean(ir._data, 'samples')/fs/60:.1f}({ir.std(ir._data, 'samples')/fs/60:.1f}) minutos\n")

        ic = ir.inclusion_criteria()
        irats = ir.get_rats(ir.get(ic))
        males = ir.get_males(data=irats)
        females = ir.get_females(data=irats)
        print(f'\n# registros incluídos = {len(ic)}')
        print(f'# ratas incluídas = {len(irats)}')
        m = ir.get(males)
        print(f'\t# machos = {len(m)}\t')
        print(f"\tpeso = {ir.mean(m, 'weight'):.1f}({ir.std(m, 'weight'):.1f}) gramos")
        print(f"\tedad = {ir.mean(m, 'age'):.1f}({ir.std(m, 'age'):.1f}) semanas\n")
        m = ir.get(females)
        print(f'\t# hembras = {len(m)}\t')
        print(f"\tpeso = {ir.mean(m, 'weight'):.1f}({ir.std(m, 'weight'):.1f}) gramos")
        print(f"\tedad = {ir.mean(m, 'age'):.1f}({ir.std(m, 'age'):.1f}) semanas\n")

        print(f'# registros excluídos por filtrado  = {len(ir.exclusion_criteria_after_filtering())}')
        print(f'# registros excluídos por delineado  = {len(ir.exclusion_criteria_after_delineation())}')
        sc = ir.selection_criteria()
        print(f'# registros útiles = {len(sc)}')

        return ir
    
    def trim_value(self, nreg): # trim for avoid spikes
        if nreg == 56: return 10580, -1 
        if nreg == 104: return 0, 124150 
        if nreg == 155: return 2380, -1 
        if nreg == 164: return 3780, -1 
        if nreg == 320: return 124750, -1
        return 0, -1