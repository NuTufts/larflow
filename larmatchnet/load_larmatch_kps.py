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


def load_larmatch_kps(loader, current_entry,
                      npairs=50000,verbose=False):
    
    data    = {"entry":current_entry,
               "tree_entry":int(current_entry)%int(loader.GetEntries())}

    t_start = time.time()
    
    # get data from match tree
    tio     = time.time()
    nbytes  = loader.load_entry(data["tree_entry"])
    dtio        = time.time()-tio

    
    if verbose:
        print "nbytes: ",nbytes," for tree entry=",data['tree_entry']
    nfilled = c_int()
    nfilled.value = 0
    spdata_v    = [ loader.triplet_v[0].make_sparse_image( p ) for p in xrange(3) ]
    matchdata   = loader.sample_data( npairs, nfilled, True )
    data.update(matchdata)


    # prepare data dictionary
    for p in xrange(3):
        data["coord_%d"%(p)] = spdata_v[p][:,0:2].astype( dtype=np.int32 )
        data["feat_%d"%(p)]  = spdata_v[p][:,2].reshape( (spdata_v[p].shape[0], 1) )
    data["matchpairs"]     = matchdata["matchtriplet"][:,0:3].astype( dtype=np.long )
    data["larmatchlabels"] = matchdata["matchtriplet"][:,3].astype( dtype=np.long )
    data["npairs"]     = nfilled.value

    tottime = time.time()-t_start

    if verbose:
        print "[load larmatch kps sample]"        
        print "  io time: %.3f secs"%(dtio)
        print "  tot time: %.3f secs"%(tottime)
        
    return data

    
if __name__ == "__main__":


    from ROOT import std
    input_files = ["larmatch_keypointssnet_small_sample_test.root"]
    device      = torch.device("cpu")

    input_v = std.vector("string")()
    for i in input_files:
        input_v.push_back(i)

    loader = larflow.keypoints.LoaderKeypointData( input_v )
    nentries = loader.GetEntries()
    print "num entries: ",nentries

    nmax    = c_int()
    nfilled = c_int()
    nmax.value = 50000

    nentries = 1000000000
    
    for ientry in xrange(nentries):
        print "[LOAD ENTRY ",ientry,"]"
        data = load_larmatch_kps( loader, ientry, verbose=True )
        for x,arr in data.items():
            if isinstance(arr,np.ndarray):
                print x,": ",arr.shape
                #print "   ",arr
            else:
                print x,": ",arr
        del data

    print "[fin]"
    raw_input()
