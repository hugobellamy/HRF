from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3, make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import new_functions as nf
import pickle
from joblib import Parallel, delayed
import multiprocessing


def noise_at_point(x, lims):
    lim1, lim2 = lims
    if lim1[0]>x[0]>lim1[1] and lim2[0]>x[1]>lim2[1]:
        return 10
    else:
        return 1


N_cores = multiprocessing.cpu_count()
print(N_cores)
NJB = 4 # larger is slower but less ram demand

# seeds
DATA_SEED = 36383
FOREST_SEED = 5343
RANDOM_SEED = 354213

# test points
SMEARS = [0, 0.5, 1, 1.5, 3]
MTRYS = [0.1, 0.3,0.5, 0.7, 1]
POWERS = [0,0.25,0.5,0.75,1,1.5,2]
POWERS_OS = [-i for i in POWERS]
BOOT_PERC = [0.1, 0.3, 0.5, 0.7, 1]
ALPHAS = [0,0.5,1,2,3,5]

datasets = [make_friedman1, make_friedman2, make_friedman3, make_regression]

lims = [[[1,0.6],[0.6,0.2]],[[100,60],[408*np.pi,200*np.pi]],[[100,60],[408*np.pi,200*np.pi]],[[1000,0.6],[1,0]]]
noise_types = [0,1]
dataset_n_vals = [1,4]

noise_vals = [1,3]

TREES = 1000

def test_dataset(dataset_n, noise, noise_type):
    print(f"Starting dataset {dataset_n}, noise {noise}, noise type {noise_type}")
    X,y = datasets[dataset_n-1](2200, noise=0, random_state=DATA_SEED)
    if dataset_n==2:
        y = y/10000 # linear transfromation stops errors from massive values we otherwise get
    if dataset_n==4:
        y = y/np.std(y)
    base_noise = noise*np.std(y) # mean of added noise
    X_train, _, y_train_clean, _ = train_test_split(X,y, test_size=10/11, random_state=DATA_SEED)
    random_generator = np.random.default_rng(RANDOM_SEED)
    if noise_type == 0:
        name = f'sim_var_friedman{dataset_n}_uni_noise-{noise}'
        indi_noises = random_generator.uniform(0, 2*base_noise, 200) # only adding noise to training set
    else:
        indi_noises = [noise_at_point(i, lims[dataset_n-1]) for i in X_train]
        indi_noises = np.array(indi_noises)*base_noise/np.mean(indi_noises)
        name = f'sim_var_friedman{dataset_n}_output_dep_noise-{noise}'

    y_train = y_train_clean + [random_generator.normal(0, i) for i in indi_noises]

    res_dict = nf.test_points(X_train, y_train, indi_noises, mtrys = MTRYS, powers_para =POWERS, powers_weight=POWERS, boot_perc = BOOT_PERC, trees = TREES, n_jobs=NJB)
    save_loc = open('results/'+name+'.pkl', 'wb')
    pickle.dump(res_dict, save_loc)
    res_dict_os = nf.test_points_os(X_train, y_train, indi_noises, smears = SMEARS, powers=POWERS_OS, boot_perc = BOOT_PERC, trees = TREES, n_jobs=NJB)
    save_loc2 = open('results/'+name+'_os.pkl', 'wb')
    pickle.dump(res_dict_os, save_loc2)

    if dataset_n==4:
        indi_noises_flipped = [1/i for i in indi_noises]
        linear_res = nf.test_points_linear(X_train, y_train, indi_noises_flipped, alphas=ALPHAS, powers=POWERS,  n_jobs=1, test_weights=False)
        save_loc3 = open('results/'+name+'_linear.pkl', 'wb')
        pickle.dump(linear_res, save_loc3)


            
if __name__ == '__main__':
    test_points = []
    for noise in noise_vals:
        for noise_type in noise_types:
            for dataset_n in dataset_n_vals:
                test_points.append((dataset_n, noise, noise_type))

    Parallel(n_jobs=round(N_cores/NJB), verbose=2)(delayed(test_dataset)(*i) for i in test_points)

  
