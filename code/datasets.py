import torch
import numpy as np

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

def load_data(path):
    path_data = path
    # which features are binary
    contfeats = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    # which features are continuous
    binfeats = [i for i in range(72) if i not in contfeats]

    data = np.loadtxt(path_data + '/RHC.csv', delimiter=',', skiprows=1)

    t, y = data[:, 0], data[:, 1][:, np.newaxis]
    x = data[:, 2:74]
    s = data[:, 0]
    w = data[:, 2:74]

    x = torch.from_numpy(x)
    y = torch.from_numpy(y).squeeze()
    t = torch.from_numpy(t).squeeze()
    s = torch.from_numpy(s).squeeze()
    w = torch.from_numpy(w).squeeze()

    data = (x, t, y, s, w)

    return data, contfeats, binfeats