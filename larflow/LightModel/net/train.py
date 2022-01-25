# LM training script

import os, sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('GTK')
import matplotlib.pyplot as plt

from lightmodelnet import LightModelNet 
from lm_dataloader import load_lm_data

from tensorboardX import SummaryWriter
from datetime import datetime
import socket

# tensorboardX: creating directories for runs
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
top_log_dir = os.path.join(
    'runs', current_time + '_' + socket.gethostname())
sub_log_dir = socket.gethostname()
print("DIRNAME:", top_log_dir+"/"+sub_log_dir)
if not os.path.exists(top_log_dir):
    os.mkdir(top_log_dir)
writer_train = SummaryWriter(top_log_dir+"/"+sub_log_dir+"_train")
writer_val   = SummaryWriter(top_log_dir+"/"+sub_log_dir+"_val")

    
model = LightModelNet(3, True)

num_iterations = 2700
#num_iterations = 200

error = nn.MSELoss()

learning_rate = 0.0001
#SGD optimizer:
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

#input_file = "../../Ana/CRTPreppedTree_crttrack_b40ad76a-1eb4-4ab0-8bf5-afbf194f216f-jobid0035.root"
input_file = "../../Ana/CRTPreppedTree_crttrack_f73c7b55-9e81-4ae7-8b87-129b3efa3248-jobid1065.root"
#input_file = "../../Ana/CRTPreppedTree_crt_all_temp.root"
#entry = 2


#train = torch.utils.data.TensorDataset(coord_train, feat_train, flash_train)

for iteration in range(num_iterations):

    entry = iteration
    
    # loads one entry
    trainset = load_lm_data(input_file, entry)
    testset = load_lm_data(input_file, entry)
    
    coord_train = trainset["coord_t"]
    feat_train = trainset["feat_t"].reshape(trainset["feat_t"].shape[0],1)
    
    flash_train = trainset["flash_t"]
    flash_train = trainset["flash_t"].reshape(8,32) # hack, change later (changed from 32 -> 256 for batch size of 8)
    # but basically want this truth flash_t to be same size as network output!
    
    coord_test = testset["coord_t"]
    feat_test = testset["feat_t"]
    flash_test = testset["flash_t"]

    '''
    print("coord_t: ",coord_train)
    print("coord_t size: ",coord_train.size())
    print("feat_t: ",feat_train)
    print("feat_t size: ",feat_train.size())
    print("flash_t: ",flash_train)
    print("flash_t size: ",flash_train.size())
    '''

    optimizer.zero_grad() # clear gradients
    
    outputs = model(coord_train,feat_train) # forward prop
    #print("output size: ",outputs.size())

    loss = error(outputs, flash_train)

    loss.backward()
    optimizer.step()

    count += 1

    loss_list.append(float(loss.data))
    iteration_list.append(count)

    print('Iteration: {} Loss: {} %'.format(count, loss.data))
    writer_train.add_scalar("loss", loss.data,count)
    print(count)


#loss_list.tolist()
print(loss_list)
print(iteration_list)
    
'''
# visualization of loss

# visualization of loss
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.title("CNN: Loss vs. Number of Iterations")
plt.show()
'''
