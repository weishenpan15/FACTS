from dataset import *
import xgboost

import numpy as np
import random
from path_search import SearchPathPermutation
from path_shap import CalPathContribution, TrainFeaturePredictorsResidual
import networkx as nx
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import argparse
import os.path as osp
import os
import json


def path_transfer(path_dict):
    new_path_dict = {}
    for path in path_dict:
        new_path = "-".join(list(path))
        new_path_dict[new_path] = path_dict[path]

    return new_path_dict


def run_experiment(args):
    data_rnd_seed = args.data_rnd_seed
    model_rnd_seed = args.model_rnd_seed
    dataset = args.dataset
    model_class = args.model_class
    if args.a_aware:
        begin_idx = 0
    else:
        begin_idx = 1
    np.random.seed(model_rnd_seed)
    random.seed(model_rnd_seed)

    data_dict, f_names, f_types, A_name, edge_list, f_map = load_data(dataset, data_rnd_seed)

    X_train, X_train_ori, y_train, a_train = data_dict['X_train'], data_dict['X_train_ori'], data_dict['y_train'], data_dict['a_train']
    X_test, X_test_ori, y_test, a_test = data_dict['X_test'], data_dict['X_test_ori'], data_dict['y_test'], data_dict['a_test']

    G = nx.Graph()
    G.add_nodes_from(f_names)
    G.add_edges_from(edge_list)
    active_nodes, dir_pre_dict, node_permutations, edge_permutations, path_permutations = SearchPathPermutation(G,A_name)
    graph_info = {"f_map": f_map, "active_nodes": active_nodes, "dir_pre_dict": dir_pre_dict, "A_name": A_name}
    predictors, X_train_res, X_test_res, active_idx, inactive_idx = TrainFeaturePredictorsResidual(args, graph_info, X_train, X_test, f_types)
    graph_info["active_idx"],graph_info["inactive_idx"] = active_idx, inactive_idx

    if model_class == "mlp":
        if dataset == "adult":
            model = MLPClassifier(hidden_layer_sizes=(16,), alpha=0.1, random_state=model_rnd_seed, warm_start=True)
        elif dataset == "compas":
            model = MLPClassifier(hidden_layer_sizes=(8,), random_state=model_rnd_seed, warm_start=True)
        else:
            model = MLPClassifier(hidden_layer_sizes=(16,), random_state=model_rnd_seed, warm_start=True)
        model.fit(X_train[:, begin_idx:], y_train)
        pred_train = model.predict(X_train[:,begin_idx:])
        pred_test = model.predict(X_test[:,begin_idx:])
    elif model_class == "xgboost":
        model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train[:, begin_idx:], label=y_train), 1000)
        pred_train = (model.predict(xgboost.DMatrix(X_train[:, begin_idx:])) > 0.5)
        pred_test = (model.predict(xgboost.DMatrix(X_test[:, begin_idx:])) > 0.5)
    else:
        model = LogisticRegression(random_state=model_rnd_seed)
        model.fit(X_train[:, begin_idx:], y_train)
        pred_train = model.predict(X_train[:,begin_idx:])
        pred_test = model.predict(X_test[:,begin_idx:])

    DP_train = pred_train[a_train == 1].mean() - pred_train[a_train == 0].mean()
    DP_test = pred_test[a_test == 1].mean() - pred_test[a_test == 0].mean()
    print("Total GT DP", y_test[a_test == 1].mean() - y_test[a_test == 0].mean())
    print("Total DP", DP_train, DP_test)

    print((pred_test[y_test == 0])[a_test[y_test == 0] == 1].mean(), (pred_test[y_test == 0])[a_test[y_test == 0] == 0].mean())
    print((pred_test[y_test == 1])[a_test[y_test == 1] == 1].mean(), (pred_test[y_test == 1])[a_test[y_test == 1] == 0].mean())
    print("Total EO", DP_train, DP_test)
    Acc_train = (pred_train == y_train).mean()
    Acc_test = (pred_test == y_test).mean()
    print("Train Accuracy", Acc_train)
    print("Test Accuracy", Acc_test)
    print("Number of permutation", len(path_permutations))
    path_ctb_dp_train = {tuple(node):[] for node in path_permutations[0]}
    path_ctb_acc_train = {tuple(node):[] for node in path_permutations[0]}
    path_ctb_dp_test = {tuple(node):[] for node in path_permutations[0]}
    path_ctb_acc_test = {tuple(node):[] for node in path_permutations[0]}

    rnd_idxes = [idx for idx in range(len(path_permutations))]
    random.shuffle(rnd_idxes)
    s_size = min(5, len(path_permutations))
    for rnd_idx in rnd_idxes[:s_size]:
        path_ctb_dp_train, path_ctb_acc_train = CalPathContribution(args, model, predictors, graph_info, X_train, a_train, y_train, X_train_res, path_permutations[rnd_idx], node_permutations[rnd_idx], path_ctb_dp_train, path_ctb_acc_train)
        path_ctb_dp_test, path_ctb_acc_test = CalPathContribution(args, model, predictors, graph_info, X_test, a_test, y_test, X_test_res, path_permutations[rnd_idx], node_permutations[rnd_idx], path_ctb_dp_test, path_ctb_acc_test)

    print("Path#Contribution to Disparity#Contribution to Accuracy")
    for path_key in path_ctb_acc_test:
        if len(path_ctb_acc_test[path_key]) > 0 and "R#" not in path_key[0]:
            path_ctb_dp_train[path_key] = np.array(path_ctb_dp_train[path_key]).mean()
            path_ctb_acc_train[path_key] = np.array(path_ctb_acc_train[path_key]).mean()
            path_ctb_dp_test[path_key] = np.array(path_ctb_dp_test[path_key]).mean()
            path_ctb_acc_test[path_key] = np.array(path_ctb_acc_test[path_key]).mean()
            print(path_key, "#", path_ctb_dp_test[path_key], "#", path_ctb_acc_test[path_key])

    out_dir = osp.join(args.result_dir, "facts/{}_{}_{}_{}_{}".format(dataset, model_class, args.use_inactive, args.data_rnd_seed, args.model_rnd_seed))
    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    out_fname = osp.join(out_dir, "result.json")

    if not osp.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_fname, "w") as out_f:
        out_str = json.dumps({'path_ctb_dp_train': path_transfer(path_ctb_dp_train), 'path_ctb_acc_train': path_transfer(path_ctb_acc_train),
                              'path_ctb_dp_test': path_transfer(path_ctb_dp_test), 'path_ctb_acc_test': path_transfer(path_ctb_acc_test),
                              'DP_train': DP_train, 'DP_test': DP_test, 'Acc_train':Acc_train, 'Acc_test':Acc_test})
        out_f.write(out_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_rnd_seed', default=0, type=int, help='data random seed')
    parser.add_argument('--model_rnd_seed', default=0, type=int, help='data random seed')
    parser.add_argument('--use_inactive', default=1, type=int, help='data random seed')
    parser.add_argument('--prob_output', default=0, type=int, help='data random seed')
    parser.add_argument('--a_aware', default=0, type=int, help='data random seed')
    parser.add_argument('--dataset', default="compas", type=str, help='name of dataset')
    parser.add_argument('--model_class', default="mlp", type=str, help='class of prediction model')
    parser.add_argument('--result_dir', default="./results", type=str, help='class of prediction model')

    args = parser.parse_args()
    run_experiment(args)