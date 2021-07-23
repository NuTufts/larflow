# LM training script

import torch
import torch.nn as nn
import numpy as np

from lightmodelnet import LightModelNet 
from lm_dataloader import load_lm_data

model = LightModelNet(3, True)

num_epochs = 1

error = nn.CrossEntropyLoss()

learning_rate = 0.001
#SGD optimizer:
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training
count = 0
loss_list = []
iteration_list = []
accuracy_list = []

input_file = "../../Ana/CRTPreppedTree_crttrack_b40ad76a-1eb4-4ab0-8bf5-afbf194f216f-jobid0035.root"
entry = 2

trainset = load_lm_data(input_file, entry)
testset = load_lm_data(input_file, entry)

coord_train = trainset["coord_t"]
feat_train = trainset["feat_t"]
flash_train = trainset["flash_t"]
coord_test = testset["coord_t"]
feat_test = testset["feat_t"]
flash_test = testset["flash_t"]

print("coord_t: ",coord_train)
print("coord_t size: ",coord_train.size())
print("feat_t: ",feat_train)
print("feat_t size: ",feat_train.size())
print("flash_t: ",flash_train)
print("flash_t size: ",flash_train.size())

#train = torch.utils.data.TensorDataset(coord_train, feat_train, flash_train)

for epoch in range(num_epochs):
    optimizer.zero_grad() # clear gradients
    
    outputs = model(coord_train,feat_train,flash_train) # forward prop
