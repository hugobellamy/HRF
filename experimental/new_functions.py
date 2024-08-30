import numpy as np
import variable_bootstrap_forest as vbf
from outputsmearing_forest import osRandomForest
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import itertools


class SmearOptimizer:
    def __init__(self, X_train, y_train, y_train_uncerts):
        self.X_train = X_train
        self.y_train = y_train
        self.y_train_uncerts = y_train_uncerts

    def make_cross_val_sets(self, n_splits=5):
        self.n_splits = n_splits
        self.X_train_sets = []
        self.y_train_sets = []
        self.X_test_sets = []
        self.y_test_sets = []
        self.y_train_uncert = []
        self.y_test_uncert = []
        for j in range(n_splits):
            self.X_train_sets.append([self.X_train[i] for i in range(len(self.X_train)) if i%n_splits != j])
            self.y_train_sets.append([self.y_train[i] for i in range(len(self.y_train)) if i%n_splits != j])
            self.X_test_sets.append([self.X_train[i] for i in range(len(self.X_train)) if i%n_splits == j])
            self.y_test_sets.append([self.y_train[i] for i in range(len(self.y_train)) if i%n_splits == j])
            self.y_train_uncert.append([self.y_train_uncerts[i] for i in range(len(self.y_train_uncerts)) if i%n_splits != j])
            self.y_test_uncert.append([self.y_train_uncerts[i] for i in range(len(self.y_train_uncerts)) if i%n_splits == j])
    
    def get_cross_val_test(self):
        sets = []
        for i in range(self.n_splits):
            sets.append([self.X_train_sets[i], self.y_train_sets[i], self.y_train_uncert[i], self.X_test_sets[i], self.y_test_sets[i], self.y_test_uncert[i]])
        return sets


def get_best_dict_vals(indexes, dict):
    combos = [i for i in itertools.product(*indexes)]
    best = 1000000000
    for i in combos:
        if dict[i] < best:
            best = dict[i]
            best_combo = i
    return best_combo


def single_test(vals, trees, data, sets,n_jobs=-1, test_weights=False):
    mtry, power, boot_per, power_weight = vals
    model = vbf.vbRandomForest(n_estimators=trees, max_features=mtry, probs_power=power, weight_power=power_weight, max_samples=boot_per, n_jobs=n_jobs)
    scores=[]
    for i in range(data.n_splits):
        base_conv_probs = [np.exp(-i) for i in sets[i][2]]
        model.fit(sets[i][0], sets[i][1], probs=base_conv_probs, sample_weight=base_conv_probs)
        preds = model.predict(sets[i][3])
        try:
            if test_weights:
                score = mean_squared_error(sets[i][4], preds, sample_weight=sets[i][5])
            else:
                score = mean_squared_error(sets[i][4], preds)
            scores.append(score)
        except:
            print(f"Error at mtry{mtry}, power{power}, power_weight{power_weight}, boot_per{boot_per}")
    if len(scores)>0:
        return np.mean(scores)
    else:
        return np.nan


def test_points(X_train, y_train, indi_noises, mtrys = [1], powers_para = [0], powers_weight=[0],boot_perc = [1], trees = 100, n_jobs=-1, test_weights=False):
    data = SmearOptimizer(X_train, y_train, indi_noises)
    data.make_cross_val_sets()
    sets = data.get_cross_val_test()
    res = {}
    # going to first check only individual points ons poweres and powers_weight
    if len(powers_para)>1 and len(powers_weight)>1:
        r_new = test_points(X_train, y_train, indi_noises, mtrys, [0], powers_weight, boot_perc, trees, n_jobs, test_weights=test_weights)
        r_new_2 = test_points(X_train, y_train, indi_noises, mtrys, powers_para, [0], boot_perc, trees, n_jobs, test_weights=test_weights)

        res = {**r_new, **r_new_2}
    else:
        for mtry in mtrys:
            for power in powers_para:
                    for power_weight in powers_weight:
                        for boot_per in boot_perc:
                            res[(mtry, power, boot_per, power_weight)] = single_test((mtry, power, boot_per, power_weight), trees, data, sets, test_weights=test_weights, n_jobs=n_jobs)
        
    return res



def test_points_os(X_train, y_train, indi_noises, smears = [0], powers=[1], boot_perc = [1], trees = 100, n_jobs=1, test_weights=False):
    data = SmearOptimizer(X_train, y_train, indi_noises)
    data.make_cross_val_sets()
    sets = data.get_cross_val_test()
    res = {}
    for smear in smears:
        for power in powers:
            for boot_per in boot_perc:
                model = osRandomForest(n_estimators=trees, output_smear=smear, power_smear=power, bootstrap=True, max_samples=boot_per, max_features=1.0, n_jobs=n_jobs)
                scores=[]
                for i in range(data.n_splits):
                    conv_probs = [np.exp(-i) for i in sets[i][2]]
                    model.fit(sets[i][0], sets[i][1], probs=conv_probs)
                    preds = model.predict(sets[i][3])
                    if test_weights:
                        score = mean_squared_error(sets[i][4], preds, sample_weight=sets[i][5])
                    else:
                        score = mean_squared_error(sets[i][4], preds)
                    scores.append(score)
                    res[(smear, power, boot_per)] = np.mean(scores)
    return res


def test_points_linear(X_train, y_train, noise_weights, alphas= [1], powers=[0],  n_jobs=1, test_weights=False):
    data = SmearOptimizer(X_train, y_train, noise_weights)
    data.make_cross_val_sets()
    sets = data.get_cross_val_test()
    res = {}
    for alpha in alphas:
        for power in powers:
            model = Ridge(alpha=alpha)
            scores=[]
            for i in range(data.n_splits):
                model.fit(sets[i][0], sets[i][1], sample_weight=np.array(sets[i][2])**power)
                preds = model.predict(sets[i][3])
                if test_weights:
                    score = mean_squared_error(sets[i][4], preds, sample_weight=sets[i][5])
                else:
                    score = mean_squared_error(sets[i][4], preds)
                scores.append(score)
                res[(alpha, power)] = np.mean(scores)
    return res
                
                
