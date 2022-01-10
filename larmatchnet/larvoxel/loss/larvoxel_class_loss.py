import torch
import torch.nn as nn

class LArVoxelClassLoss(nn.Module):

    def __init__(self):
        super( LArVoxelClassLoss,self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self,pred,target):
        return self.ce(pred,target)
