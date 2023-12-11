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

num_iterations = 100

error = nn.MSELoss()

learning_rate = 0.0001
optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)

#EPOCH = 5
#PATH = "model.pt"
#LOSS = 0.4

run = wandb.init(
    # Set the project where this run will be logged
    project="lightmodel-project",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": learning_rate,
        "epochs": 1,
    })

wandb.watch(net, log="all", log_freq=1)

# put net in training mode (vs. validation)
net.train()

for iteration in range(num_iterations): 

    #coords, feat, label, SA = load_lm_data(input_file, entry)
    coord0, feat0, label0= load_lm_data(input_file, entry)
    coord1, feat1, label1= load_lm_data(input_file, entry)

    print("coord0.shape: ", coord0.shape )
    print("feat0.shape: ", feat0.shape )
    print("coord0 (before sparse_collate): ", coord0)
    print("feat0 (before sparse_collate): ", feat0)

    print("coord1.shape: ", coord1.shape )
    print("feat1.shape: ", feat1.shape )
    print("coord1 (before sparse_collate): ", coord1)
    print("feat1 (before sparse_collate): ", feat1)

    SA = np.genfromtxt ('SA_10102023.csv', delimiter=",")
    SA_t = torch.from_numpy( SA )
    print("SA.shape: ", SA_t.shape)
    # print("type(SA): ", type(SA_t) )

    '''
    meanSA = torch.mean(SA_t)
    stdSA = torch.std(SA_t)
    maxSA = torch.max(SA_t)
    minSA = torch.min(SA_t)

    print("Mean SA: ", meanSA)
    print("Std SA: ", stdSA)
    print("Max SA: ", maxSA)
    print("Min SA: ", minSA)
    '''

    # print("This is the truth: ", label)
    # print("Shape of the truth: ", label.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    #input = ME.SparseTensor(feat, coords, device=device)
    coords, feats = ME.utils.sparse_collate( [coord0], [feat0] )
    input = ME.SparseTensor(features=feats, coordinates=coords)

    print("This is iteration: ", iteration, " out of 1000")

    print("coords.shape: ", coords.shape )
    print("feats.shape: ", feats.shape )
    print("coords (after sparse_collate): ", coords)
    print("feats (after sparse_collate): ", feats)

    uni_coords = np.unique(coords, axis=0)
    print("uni_coords", uni_coords)
    print("uni_coords.shape", uni_coords.shape)

    print("input.shape: ", input.shape)

    print("input.D: ", input.D)

    # Forward
    output = net(input)

    print("Printing output. Is it all between 0 and 1?")
    print("Output: ", output)
    outputMax = torch.max(output.F)

    print("output.shape: ", output.shape)
    print("output max: ", outputMax)
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

    print("This is the eltmult.F. Is it all between 0 and 1?")
    print("eltmult.F: ", F)
    eltMax = torch.max(eltmult.F)
    print("This is eltmax: ", eltMax)

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
    print("This is the sum_t. Is it all between 0 and 1? ", sum_t)
    #print("This is the shape of the sum_t: ", sum_t.shape)
    sumMax = torch.max(sum_t)
    print("This is sumMax: ", sumMax)


    print("label0: ", label0)
    maxPE = torch.max(label0, 0)
    print("MaxPE is: ", maxPE)
    print("The max value is: ", maxPE[0])
    print("The index is: ", maxPE[1])
    indexPE = maxPE[1].item()
    print("indexPE is: ", indexPE )

    loss = error(sum_t, label0)

    print("This is the loss: ", loss)

    wandb.log({"loss": loss.detach().item()})

    lossMaxPE = error(sum_t[indexPE], maxPE[0])

    print("This is the lossMaxPE: ", lossMaxPE)

    wandb.log({"lossMaxPE": lossMaxPE.detach().item()})

    '''
    if (iteration == 5): 
        torch.save({'epoch': iteration,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss}, 
	            '/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/larmatchnet/lightmodel/checkpoint.pth')
    '''

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

#net.save(os.path.join(wandb.run.dir, "model.h5"))
#wandb.save('model.h5')
#wandb.save('../logs/*ckpt*')
#wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
