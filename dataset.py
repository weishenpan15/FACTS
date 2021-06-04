import sklearn
import shap
from sklearn.neural_network import MLPClassifier
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import copy


def load_data(dataset ,data_rnd_seed = 0, conditional = 0):
    if dataset == "adult":
        X_train, a_train, y_train, f_names, f_types = load_adult("adult.data")
        X_test, a_test, y_test, _, _ = load_adult("adult.test")
        A_name = 'Sex'
    elif dataset == "compas":
        X, a, y, f_names, f_types = load_compas(rnd_seed=data_rnd_seed)
        A_name = 'race'
    elif dataset == "nutrition":
        X, a, y, f_names, f_types = load_nhanes(rnd_seed=data_rnd_seed)
        A_name = 'race'
    else:
        raise NotImplementedError

    if dataset == "adult":
        X_train, f_map = get_dummy(X_train, f_names, f_types)
        X_test, _ = get_dummy(X_test, f_names, f_types)
    else:
        new_X, f_map = get_dummy(X, f_names, f_types)
        X_train, X_test, y_train, y_test, a_train, a_test = sklearn.model_selection.train_test_split(new_X, y, a,
                                                                                                     test_size=0.3,
                                                                                                     random_state=data_rnd_seed)

    X_train, X_test, X_train_ori, X_test_ori = data_preprocess(X_train, X_test, f_names, f_map, f_types)
    data_dict = {"X_train": X_train, "X_test":X_test, "y_train":y_train, "y_test":y_test, "a_train":a_train, "a_test":a_test,
                 "X_train_ori":X_train_ori, "X_test_ori":X_test_ori}

    edge_list = []
    if conditional == 1:
        edge_postfix = "edge_eo"
    else:
        edge_postfix = "edge"

    with open("data/{}_{}".format(dataset, edge_postfix)) as fout:
        lines = fout.readlines()
        for line in lines:
            line_info = line.strip().split(",")
            if line_info[2] == "none":
                edge = (line_info[0], line_info[1], {'From': None})
            else:
                edge = (line_info[0], line_info[1], {'From': line_info[2]})
            edge_list.append(edge)

    return data_dict, f_names, f_types, A_name, edge_list, f_map


def get_dummy(X, f_names, f_types):
    f_map = {}
    new_X = []
    dim_cnt = 0
    for (idx, f_name) in enumerate(f_names):
        if f_types[idx] == 0:
            num_dim = 1
            f_map[f_name] = np.array([dim_cnt])
            new_X.append(X[:, idx][:, None])
        else:
            X_idx = X[:, idx].astype(np.float)
            X_idx = X_idx[True ^ np.isnan(X_idx)].astype(np.int)
            unique_X_idx = np.sort(np.unique(X_idx))
            unique_X_dict = {value: value_idx for (value_idx, value) in enumerate(unique_X_idx)}
            num_dim = len(unique_X_idx)
            new_X_idx = np.zeros((X.shape[0], num_dim)).astype(np.float)
            for i in range(X.shape[0]):
                if np.isnan(X[i, idx]):
                    new_X_idx[i, :] = np.nan
                else:
                    new_X_idx[i, unique_X_dict[int(X[i, idx])]] = 1
            new_X.append(new_X_idx)
            f_map[f_name] = np.array([i for i in range(dim_cnt, dim_cnt + num_dim)])
        dim_cnt += num_dim
    new_X = np.concatenate(new_X, 1)

    return new_X, f_map


def data_preprocess(X_train, X_test, f_names, f_map, f_types):
    imp = SimpleImputer()
    imp.fit(X_train[:, 1:])
    X_train[:, 1:] = imp.fit_transform(X_train[:, 1:])
    X_test[:, 1:] = imp.fit_transform(X_test[:, 1:])

    X_train_ori = copy.deepcopy(X_train)
    X_test_ori = copy.deepcopy(X_test)
    scaler = StandardScaler()
    for (idx, name) in enumerate(f_names):
        if idx > 0 and f_types[idx] == 0:
            f_idx = f_map[name]
            scaler.fit(X_train[:, f_idx])
            X_train[:, f_idx] = scaler.fit_transform(X_train[:, f_idx])
            X_test[:, f_idx] = scaler.fit_transform(X_test[:, f_idx])
    return X_train, X_test, X_train_ori, X_test_ori


def adult(data_file, display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        "data/" + data_file,
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    data["Target"] = data["Target"] == " >50K"
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    if display:
        return raw_data.drop(["Education", "Target", "fnlwgt"], axis=1), data["Target"].values
    else:
        return data.drop(["Target", "fnlwgt"], axis=1), data["Target"].values


def load_adult(data_file):
    X, y = adult(data_file)
    X['Marital Status'] = X['Marital Status'].astype(str)
    X['Race'] = X['Race'].astype(str)
    X['Occupation'] = X['Occupation'].astype(str)

    X['Workclass'] = (X['Workclass'] == 4).astype(int)
    X['Country'] = (X['Country'] == 39).astype(int)
    X['Race'] = (X['Race'] == '4').astype(np.int)
    X['Capital Gain'] = np.log(X['Capital Gain'] + 1)
    X['Capital Gain'] = (X['Capital Gain'] > 0).astype(np.int)
    X['Marital Status'] = (X['Marital Status'] == '2').astype(np.int)
    X['Relationship'] = (X['Relationship'] == 4).astype(np.int)
    # X['Education-Num'] = np.log(X['Education-Num'] + 1)
    # X['Hours per week'] = (X['Hours per week'] < 40).astype(np.int)

    a = X['Sex']
    X = X.drop(['Occupation', 'Capital Loss', 'Capital Gain', 'Sex'], axis=1)
    X_names = ['Sex'] + [item for item in X.columns]
    X_types = np.zeros(len(X_names))
    # X_types[2] = 1
    # X_types[3] = 1
    # X_types[5] = 1
    # X_types[4] = 1
    X_types[2] = 1
    X_types[4] = 1
    X_types[5] = 1
    X_types[6] = 1
    # X_types[7] = 1
    X_types[8] = 1
    X_values = X.values
    a_values = a.values
    y_values = y

    return np.concatenate((a_values[:, None], X_values), 1), a_values, y_values, X_names, X_types


def load_compas(rnd_seed=0):
    df_compas = pd.read_csv('data/propublica-recidivism.csv', sep=',', index_col=0)
    df_names = list(df_compas.columns)
    df_names.remove('sex')
    df_names.remove('age')
    df_names.remove('race')
    df_names.remove('two_year_recid')
    df_names.remove('priors_count')
    df_names.remove('juv_fel_count')
    df_names.remove('juv_misd_count')
    df_names.remove('juv_other_count')
    df_names.remove('c_charge_degree')
    df_compas = df_compas.drop(df_names, axis=1)

    df_compas['priors_count'] = np.log(df_compas['priors_count'] + 1)
    df_compas['juv_fel_count'] = (df_compas['juv_fel_count'] > df_compas['juv_fel_count'].values.mean()).astype(np.int)
    df_compas['juv_misd_count'] = (df_compas['juv_misd_count'] > df_compas['juv_misd_count'].values.mean()).astype(np.int)
    df_compas['juv_other_count'] = (df_compas['juv_other_count'] > df_compas['juv_other_count'].values.mean()).astype(np.int)

    df_compas['sex'] = (df_compas['sex'] == 'Male').astype(int)
    df_compas['race'] = (df_compas['race'] == 'Caucasian').astype(int)
    df_compas['c_charge_degree'] = (df_compas['c_charge_degree'] == 'F').astype(int)

    a_values = df_compas['race'].values.astype(np.int)
    y_values = 1 - df_compas['two_year_recid'].values
    df_compas = df_compas.drop(['race','two_year_recid'], axis=1)
    X_values = df_compas.values
    f_names = ['race'] + list(df_compas.columns)
    f_types = np.zeros(len(f_names))
    f_types[1] = 1
    f_types[3] = 1
    f_types[4] = 1
    f_types[5] = 1
    f_types[7] = 1
    return np.concatenate((a_values[:, None], X_values), 1), a_values, y_values, f_names, f_types


if __name__ == "__main__":
    load_data('adult')



