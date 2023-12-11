# dummy dataloader for debugging unet

import numpy as np
import torch

data = [
    [0, 0, 2.1, 0, 0],
    [0, 1, 1.4, 3, 0],
    [0, 0, 4.0, 0, 0]
]


dataArray = np.asarray(data)
print("this is dataArray.shape: ", dataArray.shape)

def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)

def load_data():
    coordTensor = [3,2,1,5]
    featTensor = [4,3,5,3]
    truth = [4,9,7,4]

    coordTensor = np.asarray(coordTensor)
    coordTensor = coordTensor.reshape((2,2))
    featTensor = np.asarray(featTensor)
    featTensor = featTensor.reshape((2,2))
    truth = np.asarray(truth)

    coord_t = torch.from_numpy(coordTensor).int()
    feat_t = torch.from_numpy(featTensor).int()
    truth_t = torch.from_numpy(truth).int()

    N = 5

    #X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], 
    #                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    
    X = (torch.rand(N,3)*128).int()
    F = torch.rand(N,1)
    
    #X = torch.rand(1,32,32)
    #F = torch.rand(1,32,32)

    print("This is X: ", X)
    print("This is the size of X: ", X.shape)

    print("This is F: ", F)
    print("This is the size of F: ", F.shape)

    #X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], 
    #                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    #K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

    #dataset = torch.utils.data.TensorDataset(X, F)

    coords, feats = to_sparse_coo(data)

    print("This is coords: ", coords )
    print("This is coords.shape: ", coords.shape )
    print("This is feats: ", feats )
    print("This is feats.shape: ", feats.shape )


    return X, F, truth_t
