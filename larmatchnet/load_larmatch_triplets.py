import os,sys,time
import ROOT as rt
import torch
import numpy as np
from larcv import larcv
from larflow import larflow
from ROOT import std
from larcvdataset.larcvserver import LArCVServer
from ctypes import c_int

"""
Load LArMatch Triplet data files for training and deploy
"""


def load_larmatch_triplets(tchain, current_entry,
                           npairs=50000,verbose=False):
    
    data    = {"entry":current_entry}

    t_start = time.time()
    
    # get data from match tree
    tio     = time.time()
    nbytes  = tchain.GetEntry(current_entry)
    nfilled = c_int()
    nfilled.value = 0
    spdata_v    = [ tchain.triplet_v[0].make_sparse_image( p ) for p in xrange(3) ]
    index_array = tchain.triplet_v[0].sample_triplet_matches( npairs, nfilled, True )    
    dtio        = time.time()-tio

    # prepare data dictionary
    for p in xrange(3):
        data["coord_%d"%(p)] = spdata_v[p][:,0:2].astype( dtype=np.int32 )
        data["feat_%d"%(p)]  = spdata_v[p][:,2].reshape( (spdata_v[p].shape[0], 1) )
    data["matchpairs"] = index_array[:,0:3].astype( dtype=np.long )
    data["labels"]     = index_array[:,3].astype( dtype=np.long )
    data["npairs"]     = nfilled.value

    tottime = time.time()-t_start

    if verbose:
        print "[load cropped sparse dual flow]"        
        print "  io time: %.3f secs"%(dtio)
        print "  tot time: %.3f secs"%(tottime)
        
    return data

    
if __name__ == "__main__":


    input_files = ["example_triplet_data.root"]
    device      = torch.device("cpu")

    tchain = rt.TChain("larmatchtriplet")
    for f in input_files:
        tchain.Add(f)
    print "num entries: ",tchain.GetEntries()

    for ientry in xrange(tchain.GetEntries()):
        data = load_larmatch_triplets( tchain, ientry, verbose=True )
        for x,arr in data.items():
            if isinstance(arr,np.ndarray):
                print x,": ",arr.shape
            else:
                print x,": ",arr
