import numpy as np
import torch
import torch.nn as nn
import MinkowskiEngine as ME

import wandb

from lm_dataloader import load_lm_data
from lightmodelnet import LightModelNet

input_file = "testfm_010724_FMDATA_coords_withErrorFlags_100Events_voxelsize5cm_test.root"
opfile = "100events_062323_FMDATA_filtered_MCTracks_opflash.root"
entry = 0
batchnum = 2

wandb.login()

net = LightModelNet(3, 32, D=3)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

####### 
# load SA tensor in here
# put on device too
###############

# put net in training mode (vs. validation)
net.train()

for iteration in range(num_iterations): 

    #coords, feat, label, SA = load_lm_data(input_file, entry)
    ##coord0, feat0, label0= load_lm_data(input_file, opfile, entry[0])
    ##coord1, feat1, label1= load_lm_data(input_file, opfile, entry[1])
    ##coord2, feat2, label2= load_lm_data(input_file, opfile, entry[2])

    # load in batch of 2: 
    coordList, featList, labelList = load_lm_data(input_file, opfile, entry, batchnum)

### CANT ADD LOST TO DEVICE ######
#    coordList = coordList.to(device)
#    featList = featList.to(device)
#    labelList = labelList.to(device)
###

    label0 = labelList[0]

    ##print("coord0.shape: ", coord0.shape )
    ##print("feat0.shape: ", feat0.shape )
    ##print("coord0 (before sparse_collate): ", coord0)
    ##print("feat0 (before sparse_collate): ", feat0)

    ##print("coord1.shape: ", coord1.shape )
    ##print("feat1.shape: ", feat1.shape )
    ##print("coord1 (before sparse_collate): ", coord1)
    ##print("feat1 (before sparse_collate): ", feat1)

    #SA = np.genfromtxt ('SA_10102023.csv', delimiter=",")
    SA = np.genfromtxt ('SA_020624_voxelsize5_allPMTS_entry0.csv', delimiter=",")
    SA1 = np.genfromtxt ('SA_020624_voxelsize5_allPMTS_entry1.csv', delimiter=",")
    ##SA2 = np.genfromtxt ('SA_020624_voxelsize5_allPMTS_entry2.csv', delimiter=",")
    SA_t = torch.from_numpy( SA )
    SA_t1 = torch.from_numpy( SA1 )
    ##SA_t2 = torch.from_numpy( SA2 )
    print("SA.shape: ", SA_t.shape)
    ##print("SA1.shape: ", SA_t1.shape)
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

    

    net = net.to(device)
    #input = ME.SparseTensor(feat, coords, device=device)
    #coords, feats = ME.utils.sparse_collate( [coord0], [feat0] )

    print("coordList is (should be batch of 2)", coordList)

    coords, feats = ME.utils.sparse_collate( coords=coordList, feats=featList )

    print("Now collated coords here are: ", coords)

    ##coords, feats = ME.utils.sparse_collate( coords=[coord0], feats=[feat0] )
    input = ME.SparseTensor(features=feats, coordinates=coords)

    print("This is iteration: ", iteration, " out of 1000")

    print("coords.shape: ", coords.shape )
    print("feats.shape: ", feats.shape )
    print("coords (after sparse_collate): ", coords)
    print("feats (after sparse_collate): ", feats)

    uni_coords = np.unique(coords, axis=0)
    print("uni_coords", uni_coords)
    print("uni_coords.shape", uni_coords.shape)

    print("input: ", input)
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

    SA_batch2 = torch.cat((SA_t, SA_t1), 0)
    print("SA_batch2.shape: ", SA_batch2.shape)

    ##eltmult = output*SA_t
    eltmult = output*SA_batch2
    #print("eltmult.shape: ", eltmult.shape)

    print("output.C: ", output.C)
    print("output.F: ", output.F)

    C = eltmult.C
    F = eltmult.F
    print("eltmult.C: ", eltmult.C)
    print("C.shape: ", C.shape)
    #print("type(C): ", type(C) )
    #print("eltmult.F: ", F)
    #print("F.shape: ", F.shape)
    #print("type(F): ", type(F) )

    print("This is the eltmult.F. Is it all between 0 and 1?")
    print("eltmult.F: ", F)
    print("F.shape: ", F.shape)
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
    

    #t_list = torch.stack(F)
    #print("This is t_list: ", t_list)
    #print("This is t_list.shape: ", t_list.shape)

    sum_t = torch.sum(F, 0)
    print("This is the sum_t. Is it all between 0 and 1? ", sum_t)
    #print("This is the shape of the sum_t: ", sum_t.shape)
    sumMax = torch.max(sum_t)
    print("This is sumMax: ", sumMax)


    print("label0: ", label0)
    maxPE = torch.max(label0, 0)
    maxThreePE = torch.topk(label0, 3, dim=0)
    print("torch.topk is :", maxThreePE)
    print("MaxPE is: ", maxPE)
    print("The max value is: ", maxPE[0])
    print("The index is: ", maxPE[1])
    indexPE = maxPE[1].item()
    indexSecondPE = maxThreePE[1][1].item()
    print("2nd largest index is: ", indexSecondPE )
    print("2nd largest value is: ", sum_t[indexSecondPE] )
    indexThirdPE = maxThreePE[1][2].item()
    print("3rd largest index is: ", indexThirdPE )
    print("3rd largest value is: ", sum_t[indexThirdPE] )

    loss = error(sum_t, label0)

    print("This is the loss: ", loss)

    wandb.log({"loss": loss.detach().item()})

    lossMaxPE = error(sum_t[indexPE], maxPE[0])
    print("This is the lossMaxPE: ", lossMaxPE)
    wandb.log({"lossMaxPE": lossMaxPE.detach().item()})

    lossSecondMaxPE = error(sum_t[indexSecondPE], maxThreePE[0][1])
    print("This is the lossSecondMaxPE: ", lossSecondMaxPE)
    wandb.log({"lossSecondMaxPE": lossSecondMaxPE.detach().item()})

    lossThirdMaxPE = error(sum_t[indexThirdPE], maxThreePE[0][2])
    print("This is the lossThirdMaxPE: ", lossThirdMaxPE)
    wandb.log({"lossThirdMaxPE": lossThirdMaxPE.detach().item()})

    #lossThirdMaxPE = error(sum_t[indexPE], maxPE[0])
    #print("This is the lossMaxPE: ", lossMaxPE)
    #wandb.log({"lossMaxPE": lossMaxPE.detach().item()})

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
