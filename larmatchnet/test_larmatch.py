import os,sys,time
from array import array
import ROOT as rt
from ROOT import std
import numpy as np
import torch

from larmatch import LArMatch

from larcv import larcv
larcv.load_pyutil()
from larflow import larflow

from load_larmatch_data import LArMatchDataset
from loss_larmatch import SparseLArMatchLoss

if __name__ == "__main__":

    DEVICE = torch.device("cpu")
    
    model = LArMatch(neval=50000).to(DEVICE)
    if False:
        print model

    criterion = SparseLArMatchLoss()

    input_larcv_files = ["test_larcv.root"]
    input_ana_files   = ["ana_flowmatch_data.root"]
    device = torch.device("cpu")
    io = LArMatchDataset( input_larcv_files, input_ana_files )

    data = io.gettensorbatch(1,DEVICE)

    start = time.time()
    with torch.set_grad_enabled(True):
        match1,match2,truth1,truth2 = model( data["coord_source"],  data["feat_source"],
                                             data["coord_target1"], data["feat_target1"],
                                             data["coord_target2"], data["feat_target2"],                                             
                                             data["pairs_flow1"],   data["pairs_flow2"],
                                             1, DEVICE, return_truth=True )
    
    print "output: ",match1.shape,match2.shape,truth1.shape,truth2.shape,truth1.sum(),truth2.sum()
    print "forward time: ",time.time()-start," secs"

    start = time.time()
    loss = criterion( match1, match2, truth1, truth2 )
    print "loss calc time: ",time.time()-start," secs; loss=",loss

    start = time.time()
    loss.backward()
    print "back-prop time: ",time.time()-start," secs"

    # DUMP GRADIENTS TO CHECK
    if False:
        for name, param in model.named_parameters():
            print name, "requires_grad=",param.requires_grad,type(param),param.shape
            if param.requires_grad:
                print "  grad: ",param.grad
    print "DONE"
    
    
