import os,sys
import torch
import torch.nn as nn

class SparseLArMatchLoss(nn.Module):
    def __init__(self):
        super(SparseLArMatchLoss,self).__init__()

    def forward(self,pred1,pred2,truth1,truth2):
        return forward_2plane(pred1,pred2,truth1,truth2)
        
    def forward_2plane(self,pred1,pred2,truth1,truth2):

        # FLOW1
        #print "pred1, truth1.sum: ",float(pred1.shape[2])," ",float(truth1.sum())
        weight1    = torch.ones( (1,), requires_grad=False, dtype=torch.float ).to(pred1.device)
        weight1[0] = float(pred1.shape[2])/float(truth1.sum())
        bce1 = torch.nn.BCEWithLogitsLoss( pos_weight=weight1, reduction='mean' )
        loss1 = bce1( pred1, truth1.type(torch.float) )

        # FLOW2
        weight2    = torch.ones( (1,), requires_grad=False, dtype=torch.float ).to(pred1.device)
        weight2[0] = float(pred2.shape[2])/float(truth2.sum())
        bce2 = torch.nn.BCEWithLogitsLoss( pos_weight=weight2, reduction='mean' )
        loss2 = bce2( pred2, truth2.type(torch.float) )

        #print "larmatchloss pos-example weights: flow1={} flow2={}".format( weight1[0], weight2[0] )

        return 0.5*(loss1+loss2)
        
    def forward_triplet(self,pred,truth):
        weight    = torch.ones( (1,), requires_grad=False, dtype=torch.float ).to(pred.device)
        weight[0] = float(pred.shape[2])/float(truth.sum())
        bce       = torch.nn.BCEWithLogitsLoss( pos_weight=weight, reduction='mean' )
        loss      = bce( pred, truth.type(torch.float) )
        return loss
