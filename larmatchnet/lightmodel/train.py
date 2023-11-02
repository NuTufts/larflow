import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

import wandb

from lm_dataloader import load_lm_data
from lightmodelnet import LightModelNet

input_file = "100events_062323_FMDATA_coords_withErrorFlags_100Events.root"
entry = 0

wandb.login()

net = LightModelNet(3, 32, D=3)

num_iterations = 1000

error = nn.MSELoss()

learning_rate = 0.0001
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

run = wandb.init(
    # Set the project where this run will be logged
    project="lightmodel-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": 1000,
    })

# put net in training mode (vs. validation)
net.train()

for iteration in range(num_iterations): 

    #coords, feat, label, SA = load_lm_data(input_file, entry)
    coord, feat, label= load_lm_data(input_file, entry)

    print("coords.shape: ", coord.shape )
    print("feat.shape: ", feat.shape )

    SA = np.genfromtxt ('SA_10102023.csv', delimiter=",")
    SA_t = torch.from_numpy( SA )
    # print("SA.shape: ", SA_t.shape)
    # print("type(SA): ", type(SA_t) )

    # print("This is the truth: ", label)
    # print("Shape of the truth: ", label.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    #input = ME.SparseTensor(feat, coords, device=device)
    coords, feats = ME.utils.sparse_collate( [coord], [feat] )
    input = ME.SparseTensor(features=feats, coordinates=coords)

    print("This is iteration: ", iteration, " out of 1000")

    print("coords.shape: ", coords.shape )
    print("feats.shape: ", feats.shape )

    print("input.shape: ", input.shape)

    print("input.D: ", input.D)

    # Forward
    output = net(input)

    print("output.shape: ", output.shape)
    #print("type(output): ", type(output) )

    eltmult = output*SA_t
    #print("eltmult.shape: ", eltmult.shape)
    C = eltmult.C
    F = eltmult.F
    #print("eltmult.C: ", C)
    #print("C.shape: ", C.shape)
    #print("type(C): ", type(C) )
    #print("eltmult.F: ", F)
    #print("F.shape: ", F.shape)
    #print("type(F): ", type(F) )

    '''
    eltmult_t1, eltmult_t2, eltmult_t3 = eltmult.sparse()
    print("This is eltmult_t1: ", eltmult_t1 )
    print("This is eltmult_t1 type: ", type(eltmult_t1) )
    print("This is eltmult_t4: ", eltmult_t3 )
    #print("This is eltmult_t[0]: ", eltmult_t[0] )
    #print("This is eltmult_t shape: ", eltmult_t.shape )
    #print("This is eltmult_t type: ", type(eltmult_t) )
    sum_t = torch.sparse.sum(eltmult_t1, 0)

    #print("eltmult.shape: ", eltmult.shape)
    #print("type(eltmult): ", type(eltmult) )

    #eltmult2 = np.array(eltmult)
    #sum_np = eltmult2.sum(0)
    #print("This is the sum_np: ", sum_np)
    #print("This is the shape of the sum_np: ", sum_np.shape)

    #sum_t = torch.from_numpy( sum_np )
    #sum_t = sum_np.sparse()

    '''

    sum_t = torch.sum(F, 0)
    #print("This is the sum_t: ", sum_t)
    #print("This is the shape of the sum_t: ", sum_t.shape)


    loss = error(sum_t, label)

    print("This is the loss: ", loss)

    wandb.log({"loss": loss.detach().item()})

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

