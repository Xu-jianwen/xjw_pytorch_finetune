from __future__ import print_function
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt



def calculate_overall_acc(y_true, y_pred, val_num_idex):
    j = 0
    overall_correct = 0
    error_data = np.zeros(len(val_num_idex))
    for i in range(1, len(val_num_idex)):
        i = i - 1
        correct = np.sum(y_true[j:j+val_num_idex[i]] == y_pred[j:j+val_num_idex[i]])
        if correct == val_num_idex[i]:
            overall_correct += 1
        else:
            error_data[i] = i + 1
        j += val_num_idex[i]
    y = np.nonzero(error_data)
    error_data = error_data[y]
    overall_acc = overall_correct / len(val_num_idex)
    return overall_acc, error_data


def calculate_error_single_data(y_true, y_pred):
    error_single_data = np.zeros(len(y_true))
    for i in range(1, len(y_true)):
        i = i - 1
        correct = np.sum(y_true[i]==y_pred[i])
        if correct != 1:
            error_single_data[i] = i + 1
    y = np.nonzero(error_single_data)
    error_single_data = error_single_data[y]
    return error_single_data


def tsne_feature_visualization(name, features, n_components):
    features_tsne = TSNE(n_components=n_components).fit_transform(features)
    return features_tsne


def D2_images_sar_plot(name, features, labels):
    ship_labels = labels
    label_com = ['cargo', 'container', 'tanker']
    colors =    ['r', 'g', 'b']
    marker =    ['o', 'v', 's']
    figsize = 10,8

    figure, ax = plt.subplots(figsize=figsize)
    for index in range(len(label_com)):
        data_ship = features[ship_labels==index]
        data_ship_x = data_ship[:,0]
        data_ship_y = data_ship[:,1]
        plt.scatter(data_ship_x,data_ship_y,c=colors[index],marker=marker[index])   
    handles,labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels = label_com, loc='best')
    plt.savefig(name+'_tsne.png', dpi=300)
    plt.show()