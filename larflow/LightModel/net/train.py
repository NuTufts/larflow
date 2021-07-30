# LM training script

import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
#matplotlib.use('GTK')
import matplotlib.pyplot as plt


from lightmodelnet import LightModelNet 
from lm_dataloader import load_lm_data

model = LightModelNet(3, True)

num_iterations = 200

error = nn.MSELoss()

learning_rate = 0.000000000001
#SGD optimizer:
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

#input_file = "../../Ana/CRTPreppedTree_crttrack_b40ad76a-1eb4-4ab0-8bf5-afbf194f216f-jobid0035.root"
input_file = "../../Ana/CRTPreppedTree_crt_all_temp.root"
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
    flash_train = trainset["flash_t"].reshape(1,32) # hack, change later
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