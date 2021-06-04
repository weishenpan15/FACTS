import networkx as nx
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from scipy.stats import pearsonr
import xgboost
import random
import json
from utils import *
import matplotlib.pyplot as plt


def predict_func(model, data, y, a, A_name):
    if isinstance(model, xgboost.Booster):
        return model.predict(xgboost.DMatrix(data)) > 0.5
    else:
        return model.predict(data)


def predict_proba_func(model, data, y, a, A_name):
    if isinstance(model, xgboost.Booster):
        return model.predict(xgboost.DMatrix(data))
    else:
        return model.predict_proba(data)[:,1]

def TrainCausalPredictorsResidual(args, graph_info, X_train, X_test, f_types, model_class="lr"):
    f_map = graph_info["f_map"]
    active_nodes = graph_info["active_nodes"]
    parent_dict = graph_info["parent_dict"]
    A_name = graph_info["A_name"]

    active_idx = []
    for node in active_nodes:
        for item in node.split(":"):
            active_idx += list(f_map[item])
    inactive_idx = [idx for idx in range(X_train.shape[1]) if idx not in active_idx]
    if A_name in active_nodes:
        active_idx.remove(0)

    predictors = {}
    residual_X_test = {}
    residual_X_train = {}
    for f in active_nodes:
        if f != A_name:
            in_idx = LookForDpIdx(f, f_map, parent_dict)
            X_train_in = X_train[:, np.array(in_idx)]
            X_test_in = X_test[:, np.array(in_idx)]
            for out_f in f.split(":"):
                if len(f_map[out_f]) == 1:
                    X_train_out = X_train[:, f_map[out_f]]
                    if model_class == "lr":
                        predictor = LinearRegression()
                    else:
                        predictor = MLPRegressor(hidden_layer_sizes=(8,),random_state=0)
                    predictor.fit(X_train_in, X_train_out)
                    predictors[out_f] = predictor
                    pred = predictor.predict(X_train_in)
                    r = X_train_out - pred
                    print(out_f, r.var(), X_train_out.var())
                    residual_X_train[out_f] = r
                    residual_X_test[out_f] = X_test[:, f_map[out_f]] - predictor.predict(X_test_in)
                else:
                    assert len(f_map[out_f]) <= 3
                    X_train_out = X_train[:, f_map[out_f]]
                    X_train_max = X_train_out.max(axis = 1)
                    X_train_in_select = X_train_in[X_train_max > 0.9999999,:]
                    X_train_out_select = X_train_out[X_train_max > 0.999999,:]
                    X_train_out_1dim_select = X_train_out_select.argmax(axis=1)
                    if model_class == "lr":
                        predictor = LogisticRegression()
                    else:
                        predictor = MLPClassifier(hidden_layer_sizes=(8,),random_state=0)
                    predictor.fit(X_train_in_select, X_train_out_1dim_select)
                    print(out_f, (predictor.predict(X_train_in_select) == X_train_out_1dim_select).mean())
                    predictors[out_f] = predictor

                    X_test_out = X_test[:, f_map[out_f]]
                    pred_proba = predictor.predict_proba(X_train_in)
                    residual_X_train[out_f] = ObtainResidual(pred_proba, X_train_out)
                    pred_proba = predictor.predict_proba(X_test_in)
                    residual_X_test[out_f] = ObtainResidual(pred_proba, X_test_out)

    return predictors, residual_X_train, residual_X_test, active_idx, inactive_idx


def TrainFeaturePredictorsResidual(args, graph_info, X_train, X_test, f_types, model_class="lr"):
    f_map = graph_info["f_map"]
    active_nodes = graph_info["active_nodes"]
    dir_pre_dict = graph_info["dir_pre_dict"]
    A_name = graph_info["A_name"]

    active_idx = []
    for node in active_nodes:
        for item in node.split(":"):
            active_idx += list(f_map[item])
    inactive_idx = [idx for idx in range(X_train.shape[1]) if idx not in active_idx]
    if A_name in active_nodes:
        active_idx.remove(0)

    predictors = {}
    residual_X_test = {}
    residual_X_train = {}
    for f in active_nodes:
        if f != A_name:
            if args.use_inactive:
                in_idx = LookForDpIdx(f, f_map, dir_pre_dict) + inactive_idx
            else:
                in_idx = LookForDpIdx(f, f_map, dir_pre_dict)
            X_train_in = X_train[:, np.array(in_idx)]
            X_test_in = X_test[:, np.array(in_idx)]
            for out_f in f.split(":"):
                if len(f_map[out_f]) == 1:
                    X_train_out = X_train[:, f_map[out_f]]
                    if model_class == "lr":
                        predictor = LinearRegression()
                    else:
                        predictor = MLPRegressor(hidden_layer_sizes=(8,),random_state=0)
                    predictor.fit(X_train_in, X_train_out.ravel())
                    predictors[out_f] = predictor
                    pred = predictor.predict(X_train_in)[:, None]
                    r = X_train_out - pred

                    # print(out_f, r.var(), X_train_out.var())
                    residual_X_train[out_f] = r
                    residual_X_test[out_f] = X_test[:, f_map[out_f]] - predictor.predict(X_test_in)[:, None]
                else:
                    assert len(f_map[out_f]) <= 3
                    X_train_out = X_train[:, f_map[out_f]]
                    X_train_max = X_train_out.max(axis = 1)
                    X_train_in_select = X_train_in[X_train_max > 0.9999999,:]
                    X_train_out_select = X_train_out[X_train_max > 0.999999,:]
                    X_train_out_1dim_select = X_train_out_select.argmax(axis=1)
                    if model_class == "lr":
                        predictor = LogisticRegression()
                    else:
                        predictor = MLPClassifier(hidden_layer_sizes=(8,),random_state=0)
                    predictor.fit(X_train_in_select, X_train_out_1dim_select)
                    # print(out_f, (predictor.predict(X_train_in_select) == X_train_out_1dim_select).mean())
                    predictors[out_f] = predictor

                    X_test_out = X_test[:, f_map[out_f]]
                    pred_proba = predictor.predict_proba(X_train_in)
                    residual_X_train[out_f] = ObtainResidual(pred_proba, X_train_out)
                    pred_proba = predictor.predict_proba(X_test_in)
                    residual_X_test[out_f] = ObtainResidual(pred_proba, X_test_out)

    return predictors, residual_X_train, residual_X_test, active_idx, inactive_idx


def CalPathContribution(args, model, predictors, graph_info, X_test, a_test, y_test, r_test, path_permutation, node_permutation, path_ctb_DP, path_ctb_Acc):
    foreground_X, background_X, sample_a, sample_y = [], [], [], []
    f_map = graph_info["f_map"]
    active_nodes = graph_info["active_nodes"]
    A_name = graph_info["A_name"]
    inactive_idx = graph_info["inactive_idx"]
    dir_pre_dict = graph_info["dir_pre_dict"]
    a_aware = args.a_aware
    if a_aware:
        begin_idx = 0
    else:
        begin_idx = 1
    w_a0, w_a1 = (1 - a_test).mean(), a_test.mean()
    sample_r = {out_f:[] for a_node in active_nodes for out_f in a_node.split(":") if out_f != A_name}
    for r_idx in range(2):
        foreground_X_sample = deepcopy(X_test)
        background_X_sample = deepcopy(X_test)
        rnd_idx = np.array([i for i in range(X_test.shape[0])])
        np.random.seed(r_idx)
        np.random.shuffle(rnd_idx)
        background_X_sample[:, 0] = 0

        for a_node in node_permutation:
            if a_node != A_name:
                if args.use_inactive:
                    in_idx = LookForDpIdx(a_node, f_map, dir_pre_dict) + inactive_idx
                else:
                    in_idx = LookForDpIdx(a_node, f_map, dir_pre_dict)
                X_in = background_X_sample[:, np.array(in_idx)]
                for out_f in a_node.split(":"):
                    out_idx = f_map[out_f]
                    if len(out_idx) == 1:
                        background_X_sample[:, out_idx] = r_test[out_f] + predictors[out_f].predict(X_in)[:, None]
                    else:
                        tmp_pred = predictors[out_f].predict_proba(X_in)
                        tmp_sample = RecoverSample(tmp_pred, r_test[out_f])
                        background_X_sample[:, out_idx] = tmp_sample
                    sample_r[out_f].append(r_test[out_f])
        if a_aware == 0:
            background_X_sample[:, 0] = X_test[:, 0]

        foreground_X.append(foreground_X_sample)
        background_X.append(background_X_sample)
        sample_a.append(a_test)
        sample_y.append(y_test)
    foreground_X = np.concatenate(foreground_X)
    background_X = np.concatenate(background_X)
    sample_a = np.concatenate(sample_a)
    sample_y = np.concatenate(sample_y)
    for node_key in sample_r:
        sample_r[node_key] = np.concatenate(sample_r[node_key])

    search_path, search_values, search_dict = [], [], {}

    if args.prob_output == 1:
        last_pred = predict_proba_func(model, background_X[:, begin_idx:], sample_y, sample_a, A_name)
        foreground_pred = predict_proba_func(model, foreground_X[:, begin_idx:], sample_y, sample_a, A_name)
    else:
        last_pred = predict_func(model, background_X[:, begin_idx:], sample_y, sample_a, A_name)
        foreground_pred = predict_func(model, foreground_X[:, begin_idx:], sample_y, sample_a, A_name)

    last_pred_a0, last_pred_a1 = last_pred[:len(y_test)], last_pred[len(y_test):]
    last_Acc = w_a0 * (last_pred_a0 == y_test).mean() + w_a1 * (last_pred_a1 == y_test).mean()
    last_DP = w_a0 * (last_pred_a0[a_test == 1].mean() - last_pred_a0[a_test == 0].mean()) + w_a1 * (
                last_pred_a1[a_test == 1].mean() - last_pred_a1[a_test == 0].mean())
    foreground_DP = foreground_pred[sample_a == 1].mean() - foreground_pred[sample_a == 0].mean()
    foreground_Acc = (foreground_pred == sample_y).mean()

    current_X_test = deepcopy(background_X)
    last_Acc = w_a0 * (last_pred_a0 == y_test).mean() + w_a1 * (last_pred_a1 == y_test).mean()
    last_DP = w_a0 * (last_pred_a0[a_test == 1].mean() - last_pred_a0[a_test == 0].mean()) + w_a1 * (
                last_pred_a1[a_test == 1].mean() - last_pred_a1[a_test == 0].mean())
    current_pred_list = []
    current_X_list = []
    background_X_list = []
    for idx in range(2):
        current_pred_list.append(last_pred[idx * len(y_test): (idx + 1) * len(y_test)][:,None])
        current_X_list.append(current_X_test[idx * len(y_test): (idx + 1) * len(y_test), :][:, :, None])
        background_X_list.append(background_X[idx * len(y_test): (idx + 1) * len(y_test), :][:, :, None])
    last_pred_list = np.concatenate(current_pred_list, axis=1)
    last_pred_aver = last_pred_list.mean(axis=1)
    last_X_list = np.concatenate(current_X_list, axis=2)
    background_X_list = np.concatenate(background_X_list, axis=2)

    node_paths = []
    for (cur_idx, cur_path) in enumerate(path_permutation):
        if cur_path[-1] == 'Relationship':
            a = 1
        if len(cur_path) == 1:
            assert a_aware == 1
            current_X_test[:, 0] = foreground_X[:, 0]
        else:
            out_result = []
            if args.use_inactive:
                in_idx = LookForDpIdx(cur_path[-1], f_map, dir_pre_dict) + inactive_idx
            else:
                in_idx = LookForDpIdx(cur_path[-1], f_map, dir_pre_dict)
            X_in = deepcopy(background_X)
            if len(cur_path) == 2:
                assert cur_path[0] == A_name
                X_in[:, 0] = foreground_X[:, 0]
            else:
                all_pre_paths = deepcopy(node_paths)
                all_pre_paths.append(cur_path)
                pre_path_dict = DividePathByPre(all_pre_paths)
                for pre_key in pre_path_dict:
                    if pre_key == A_name:
                        X_in[:, 0] = foreground_X[:, 0]
                    else:
                        pre_path = pre_path_dict[pre_key]
                        node_tuples = []
                        for node_path in pre_path:
                            node_tuples.append(tuple(node_path))
                        node_tuples = tuple(node_tuples)
                        pre_idx = []
                        for item in pre_key.split(":"):
                            pre_idx += list(f_map[item])
                        X_in[:, np.array(pre_idx)] = search_values[search_dict[node_tuples]]
            X_in = X_in[:, np.array(in_idx)]
            for out_f in cur_path[-1].split(":"):
                if len(f_map[out_f]) == 1:
                    out_result.append(predictors[out_f].predict(X_in)[:, None] + sample_r[out_f])
                else:
                    tmp_pred = predictors[out_f].predict_proba(X_in)
                    if cur_idx == len(path_permutation) - 1 or path_permutation[cur_idx + 1][-1] != cur_path[-1]:
                        tmp_sample = foreground_X[:, np.array(f_map[out_f])]
                    else:
                        tmp_sample = RecoverSample(tmp_pred, sample_r[out_f])
                    out_result.append(tmp_sample)
            out_result = np.concatenate(out_result, axis=1)
            node_paths.append(cur_path)
            node_tuples = []
            for node_path in node_paths:
                node_tuples.append(tuple(node_path))
            node_tuples = tuple(node_tuples)
            search_dict[node_tuples] = len(search_path)
            search_path.append(deepcopy(node_paths))
            search_values.append(out_result)

            update_idx = []
            for out_f in cur_path[-1].split(":"):
                update_idx += list(f_map[out_f])
            current_X_test[:, np.array(update_idx)] = out_result

            if cur_idx == len(path_permutation) - 1 or path_permutation[cur_idx + 1][-1] != cur_path[-1]:
                node_paths = []

        if args.prob_output == 1:
            current_pred = predict_proba_func(model, current_X_test[:, begin_idx:], sample_y, sample_a, A_name)
        else:
            current_pred = predict_func(model, current_X_test[:, begin_idx:], sample_y, sample_a, A_name)

        current_pred_a0, current_pred_a1 = current_pred[:len(y_test)], current_pred[len(y_test):]
        current_Acc = w_a0 * (current_pred_a0 == y_test).mean() + w_a1 * (current_pred_a1 == y_test).mean()
        current_DP = w_a0 * (current_pred_a0[a_test == 1].mean() - current_pred_a0[a_test == 0].mean()) + w_a1 * (
                    current_pred_a1[a_test == 1].mean() - current_pred_a1[a_test == 0].mean())
        path_ctb_DP[tuple(cur_path)].append(current_DP - last_DP)
        path_ctb_Acc[tuple(cur_path)].append(current_Acc - last_Acc)

        current_pred_list = []
        current_X_list = []
        for idx in range(2):
            current_pred_list.append(current_pred[idx * len(y_test): (idx + 1) * len(y_test)][:, None])
            current_X_list.append(current_X_test[idx * len(y_test): (idx + 1) * len(y_test), :][:, :, None])
        current_pred_list = np.concatenate(current_pred_list, axis=1)
        current_pred_aver = current_pred_list.mean(axis=1)
        current_X_list = np.concatenate(current_X_list, axis=2)

        last_DP, last_Acc = current_DP, current_Acc
        last_X_list = deepcopy(current_X_list)
        last_X, last_pred, last_pred_list, last_pred_aver = deepcopy(current_X_test), deepcopy(current_pred), deepcopy(current_pred_list), deepcopy(current_pred_aver)

    return path_ctb_DP, path_ctb_Acc


def SearchByEdgeSet(args, model, predictors, graph_info, train_dict, test_dict, node_permutation, path_permutation, edge_sets):
    search_acc_train, search_dp_train = [], []
    search_acc_test, search_dp_test = [], []
    for edge_set in edge_sets:
        current_acc_train, current_dp_train, current_acc_test, current_dp_test = PredictWithEdgeSet2(args, model, predictors, graph_info, train_dict, test_dict, node_permutation, path_permutation, edge_set)
        search_acc_train.append(current_acc_train)
        search_dp_train.append(current_dp_train)
        search_acc_test.append(current_acc_test)
        search_dp_test.append(current_dp_test)
    return search_acc_train, search_dp_train, search_acc_test, search_dp_test


def trans_json_dict(json_dict):
    new_dict = {}
    for key in json_dict:
        key_info = key.split("-")
        key_tuple = tuple(key_info)
        new_dict[key_tuple] = json_dict[key]

    return new_dict





