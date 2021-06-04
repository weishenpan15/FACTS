import networkx as nx
from copy import deepcopy
# from dataset import load_nhanes
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.stats import pearsonr
import xgboost
import random
import json

from sklearn.neural_network import MLPClassifier


class customMLPClassifer(MLPClassifier):
    def resample_with_replacement(self, X_train, y_train, sample_weight):
        # normalize sample_weights if not already
        sample_weight = sample_weight.values
        sample_weight = sample_weight / sample_weight.sum(dtype=np.float64)

        X_train_resampled = np.zeros((len(X_train), len(X_train[0])), dtype=np.float32)
        y_train_resampled = np.zeros((len(y_train)), dtype=np.int)
        for i in range(len(X_train)):
            # draw a number from 0 to len(X_train)-1
            draw = np.random.choice(np.arange(len(X_train)), p=sample_weight)

            # place the X and y at the drawn number into the resampled X and y
            X_train_resampled[i] = X_train[draw]
            y_train_resampled[i] = y_train[draw]

        return X_train_resampled, y_train_resampled

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            X, y = self.resample_with_replacement(X, y, sample_weight)

        return self._fit(X, y, incremental=(self.warm_start and
                                            hasattr(self, "classes_")))

def path_transfer(path_dict):
    new_path_dict = {}
    for path in path_dict:
        new_path = "-".join(list(path))
        new_path_dict[new_path] = path_dict[path]

    return new_path_dict

# def predict_func(model, data):
#     if isinstance(model, xgboost.Booster):
#         return model.predict(xgboost.DMatrix(data)) > 0.5
#     else:
#         return model.predict(data)
#
#
# def predict_proba_func(model, data):
#     if isinstance(model, xgboost.Booster):
#         return model.predict(xgboost.DMatrix(data))
#     else:
#         return model.predict_proba(data)[:,1]


def DividePathByPre(pre_paths):
    pre_path_dict = {}
    for pre_path in pre_paths:
        if pre_path[-2] not in pre_path_dict:
            pre_path_dict[pre_path[-2]] = []
        pre_path_dict[pre_path[-2]].append(pre_path[:-1])
    return pre_path_dict


def SampleWithPro(prob_array):
    n_list = [n for n in range(prob_array.shape[1])]
    new_array = np.zeros_like(prob_array)
    for idx in range(new_array.shape[0]):
        s_idx = np.random.choice(n_list,p = prob_array[idx, :].ravel())
        new_array[idx, s_idx] = 1.0
    return new_array


def LookForDpIdx(node, f_map, dir_pre_dict):
    in_f = dir_pre_dict[node]
    in_idx = []
    for tmp_f in in_f:
        if "R#" not in tmp_f:
            for f_idx in tmp_f.split(":"):
                in_idx += list(f_map[f_idx])
    return in_idx


def ObtainResidual(pred, target):
    residual = np.zeros(pred.shape[0])
    for idx in range(target.shape[0]):
        if target[idx, :].max() < 0.99999:
            residual[idx] = np.random.random()
        else:
            while True:
                rnd_value = np.random.random()
                if target[idx, :].argmax() == 0:
                    if rnd_value < pred[idx, 0]:
                        residual[idx] = rnd_value
                        break
                else:
                    if rnd_value > pred[idx, 0]:
                        residual[idx] = rnd_value
                        break
    return residual


def RecoverSample(pred, residual):
    output = np.zeros_like(pred)
    for idx in range(pred.shape[0]):
        if residual[idx] < pred[idx, 0]:
            output[idx, 0] = 1.0
        else:
            output[idx, 1] = 1.0
    return output