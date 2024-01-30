import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan, Birch, SpectralClustering
from sklearn.metrics import r2_score, mean_squared_error
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from model_state import get_model_params, get_model_params_gradientNorm
from model_state import features_concat, learning_state_features_concat
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.func import vmap 
from torch.func import grad
from functorch import make_functional_with_buffers, make_functional
from sklearn import preprocessing
import toolz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_variance(array):

    means = [row.mean() for row in array]
    squared_errors = [(row-mean)**2 for row, mean in zip(array, means)]
    variances = [row.mean() for row in squared_errors]
    return np.array(variances)


def qbc(committee, unlabeled_data):

    # prediction for every single data with regressors
    vote_prediction = committee.vote(unlabeled_data)
    res = np.var(vote_prediction, axis=1)

    return res


def normalization(data):
    min_value = min(data)
    max_value = max(data)
    new_list = []

    if min_value < 0:
        data = data + np.abs(min_value)
    for i in data:
        new_list.append((i-min_value) / (max_value-min_value))
    return new_list


def total_disagrement(model, data):
    return np.sum(np.var(model.vote(data), axis=1))


# Total gradient norm
def grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


# Averaged gradient norm of the model and current training set
def averaged_grad_norm(committee):
    avg_norm = 0
    for i in committee:
        avg_norm = avg_norm + grad_norm(i.module_)
    avg_norm = avg_norm/len(committee)
    return avg_norm


def normalize_the_gradient(an_array):
    norm = np.linalg.norm(an_array)
    return an_array/norm


def normalize_the_LALfeatures(X_train):
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train


def error_reduct_fuc(X_train, Pool_X, committee, learning_state_features, loss_reduction_target, remaining_samples, predictor):
    
    # Fit the Predictor of MAL:
    predictor.fit(learning_state_features, loss_reduction_target.ravel().reshape(-1,))
    y_train_pred = predictor.predict(learning_state_features)
    training_score = r2_score(loss_reduction_target, y_train_pred)
    print("Embeddings training score:", training_score)
    
    Error_reduction_list = np.empty(shape=(0, 0))
    # The model states in this iteration:
    model_params = get_model_params_gradientNorm(committee)
    unlabeled_data = X_train[Pool_X]
    LALfeatures = unlabeled_data 
    expected_error_reduction = predictor.predict(LALfeatures)

    return normalization(expected_error_reduction.tolist()), training_score


def loss_fn(params, data, targets, func_model):
    preds = func_model(params, data)
    return F.mse_loss(preds, targets)


# Averaged gradient norm of the candidate sample
def get_avg_grad_norm(committee, sample, target):
    sample = torch.from_numpy(sample).to(device)
    target = torch.from_numpy(target).to(device)
    avg_norm = 0
    for i in committee:
        total_norm = 0
        model = i.module_
        func_model, params = make_functional(model)
        ft_per_sample_grads = vmap(grad(loss_fn), (None, 0, 0, None))(params, sample, target, func_model)
        for p in ft_per_sample_grads:
            param_norm = p.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        avg_norm += total_norm

    return avg_norm/len(committee)


def training_loss(model, X_train, Y_train):
    return mean_squared_error(Y_train, model.predict(X_train).reshape(-1))