
from sklearn.model_selection import train_test_split
import new_functions as nf
import pickle
import numpy as np
import pandas as pd

# seeds
FOREST_SEED = 5343
RANDOM_SEED = 354213

# test points
SMEARS = [0.5, 1, 1.5, 2]
MTRYS = [0.05, 0.15, 0.3]
POWERS = [0,0.25,0.5,0.75,1,1.5,2]
POWERS_OS = [-i for i in POWERS]
BOOT_PERC = [0.15, 0.3,0.45, 0.6]

TREES = 250

def test_dataset(loc, n_good):
    data_good = [pd.read_csv(f'{loc}good_part{i}.csv') for i in range(n_good)]
    data_reg = [pd.read_csv(f'{loc}regular_part{i}.csv') for i in range(50)]

    data = data_good + data_reg

    training_X = []
    training_y = []
    training_y_cert = []

    for i in data:
        training_X += list(i.iloc[:, :1024].values)
        training_y += list(i.iloc[:, 1024:-1].values.T[0])
        training_y_cert += list(i.iloc[:, -1].values.T)

    training_y_cert = 1/np.array(training_y_cert)

    name = loc[-5:-1]+f'_{n_good}'
    res_dict = nf.test_points(training_X, training_y, training_y_cert, mtrys = MTRYS, powers_para =POWERS, powers_weight=POWERS, boot_perc = BOOT_PERC, trees = TREES, n_jobs=-1)
    save_loc = open(f'new_data/results/{name}.pkl', 'wb')
    pickle.dump(res_dict, save_loc)
    save_loc.close()
    res_dict_os = nf.test_points_os(training_X, training_y, training_y_cert, smears =SMEARS, powers=POWERS_OS, boot_perc =BOOT_PERC, trees = TREES, n_jobs=-1)
    save_loc2 = open(f'new_data/results/{name}os.pkl', 'wb')
    pickle.dump(res_dict_os, save_loc2)
    save_loc2.close()


sources = ['652039/652039_and_686949_','720582/720582_and_743254_','720704/720704_and_743261_','scp1/scp1_','gli_sufu/gli_sufu_','TARR1/TARR1_','CRF-R2/CRF-R2_', 'PAFAH1B2/PAFAH1B2_','DAX1/DAX1','EBI2/EBI2_', 'PFG/PFG_']

n_goods = [1]
if __name__ == '__main__':
    for loc in sources:
        for n_good in n_goods:
            test_dataset('new_data/'+loc, n_good)
