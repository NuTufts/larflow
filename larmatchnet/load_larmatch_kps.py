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


def load_larmatch_kps(tchain, current_entry,
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
    print "index_array: ",index_array.shape

    ssnet_label  = np.zeros((index_array.shape[0]),  dtype=np.long)
    ssnet_weight = np.zeros((index_array.shape[0]),  dtype=np.float)
    kpd_label    = np.zeros((index_array.shape[0]),  dtype=np.long)
    kpd_shift    = np.zeros((index_array.shape[0],3),dtype=np.float)
    for i in xrange(index_array.shape[0]):
        ssnet_label[i]  = tchain.trackshower_label_v[index_array[i,3]]
        ssnet_weight[i] = tchain.trackshower_weight_v[index_array[i,3]]
        kpd_label[i]    = tchain.kplabel[index_array[i,3]][0]
        for j in xrange(3):
            kpd_shift[i,j] = tchain.kplabel[index_array[i,3]][1+j]

    # prepare data dictionary
    for p in xrange(3):
        data["coord_%d"%(p)] = spdata_v[p][:,0:2].astype( dtype=np.int32 )
        data["feat_%d"%(p)]  = spdata_v[p][:,2].reshape( (spdata_v[p].shape[0], 1) )
    data["matchpairs"] = index_array[:,0:3].astype( dtype=np.long )
    data["larmatchlabels"]  = index_array[:,3].astype( dtype=np.long )
    data["ssnetlabels"]     = ssnet_label
    data["ssnetweights"]    = ssnet_weight
    data["kplabels"]        = kpd_label
    data["kpshifts"]        = kpd_shift
    data["npairs"]     = nfilled.value

    tottime = time.time()-t_start

    if verbose:
        print "[load cropped sparse dual flow]"        
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
    
    for ientry in xrange(nentries):
        print "[LOAD ENTRY ",ientry,"]"
        loader.load_entry( ientry )
        data = loader.sample_data( 50000, nfilled, True )
        print type(data)
        for x,arr in data.items():
            if isinstance(arr,np.ndarray):
                print x,": ",arr.shape
                print "   ",arr
            else:
                print x,": ",arr

    print "[fin]"
    raw_input()
