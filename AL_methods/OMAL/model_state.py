import numpy as np
from torch.nn import Sigmoid, ReLU, BatchNorm1d, ELU
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torch.func import vmap
from torch.func import grad
from functorch import make_functional_with_buffers, make_functional 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_input_for_hidden_layers(weights, bias, data):

    number_of_neurons = weights.shape[0]
    matrix = np.dot(weights, data.T)
    res = matrix + bias.reshape(-1, 1)

    return res


def features_concat(model_params, data_params):
    return np.concatenate([model_params, data_params])


def learning_state_features_concat(d1, d2):

    return np.vstack((d1, d2))


def get_model_params(committee, remaining_data, data_idx):

    total_output_hidden1 = []
    total_output_hidden2 = []
    total_output_hidden3 = []
    total_output_hidden4 = []

    total_weight1 = []
    total_bias1 = []
    total_weight2 = []
    total_bias2 = []
    total_weight3 = []
    total_bias3 = []

    return_nodes = {"hidden1": "layer1", "hidden2": "layer2", "hidden3": "layer3"}
    for i in committee:

        model2 = create_feature_extractor(i.module_, return_nodes=return_nodes)
        intermediate_outputs = model2(torch.from_numpy(remaining_data[data_idx]))

        output_1 = intermediate_outputs['layer1'].detach().cpu().numpy().astype(np.float32)

        total_output_hidden1.append(output_1)

        output_2 = intermediate_outputs['layer2'].detach().cpu().numpy().astype(np.float32)
        total_output_hidden2.append(output_2)


        output_3 = intermediate_outputs['layer3'].detach().cpu().numpy().astype(np.float32)
        total_output_hidden3.append(output_3)


    averaged_output_1 = np.array([sum(x) for x in zip(* total_output_hidden1)])/len(committee)
    averaged_output_2 = np.array([sum(x) for x in zip(* total_output_hidden2)])/len(committee)
    averaged_output_3 = np.array([sum(x) for x in zip(* total_output_hidden3)])/len(committee)

    model_state = {'o1': averaged_output_1.reshape(1, -1).astype(np.float32), 'o2':averaged_output_2.reshape(1, -1).astype(np.float32), 'o3':averaged_output_3.reshape(1, -1).astype(np.float32)}


    return model_state


def get_model_params_gradientNorm(committee):
    norm_of_committee = []
    for i in committee:
        norm_of_model = []
        parameters = [p for p in i.module_.parameters() if p.grad is not None and p.requires_grad]
        for p in parameters:
            param_norm = p.grad.detach().data.norm(2)
            norm_of_model.append(param_norm.data.cpu().detach().numpy())
        norm_of_committee.append(norm_of_model)
    return np.array([sum(x) for x in zip(* norm_of_committee)])/len(committee)







