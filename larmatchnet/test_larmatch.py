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

    #DEVICE = torch.device("cpu")
    DEVICE = torch.device("cuda")

    MAX_PAIRS_TESTED=20000
    
    model = LArMatch(neval=MAX_PAIRS_TESTED).to(DEVICE)
    if False:
        print model

    criterion = SparseLArMatchLoss()

    input_larcv_files = ["/home/twongj01/data/larmatch_training_data/loose_positive_examples/larmatch_train_p00.root"]
    input_ana_files   = ["/home/twongj01/data/larmatch_training_data/loose_positive_examples/larmatch_train_p00.root"]
    device = torch.device("cpu")
    io = LArMatchDataset( input_larcv_files, input_ana_files, npairs=MAX_PAIRS_TESTED )

    data = io.gettensorbatch(1,DEVICE)

    start = time.time()
    with torch.set_grad_enabled(True):
        match1,match2,truth1,truth2 = model( data["coord_source"],  data["feat_source"],
                                             data["coord_target1"], data["feat_target1"],
                                             data["coord_target2"], data["feat_target2"],                                             
                                             data["pairs_flow1"],   data["pairs_flow2"],
                                             1, DEVICE, return_truth=True,
                                             npts1=data["npairs1"][0],
                                             npts2=data["npairs2"][0])
    
    print "output: ",match1.shape,match2.shape,truth1.shape,truth2.shape,truth1.sum(),truth2.sum()
    print "forward time: ",time.time()-start," secs"

    start = time.time()
    loss = criterion( match1, match2, truth1, truth2 )
    print "loss calc time: ",time.time()-start," secs; loss=",loss.item()

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
    
    
