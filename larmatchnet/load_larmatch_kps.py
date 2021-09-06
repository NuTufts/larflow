from __future__ import print_function
import os,sys,time
import ROOT as rt
import torch
import numpy as np
from larcv import larcv
from larflow import larflow
from ROOT import std
from ctypes import c_int

"""
Load LArMatch Triplet data files for training and deploy.
This also loads Key-Point-SSNet (KPS) data.
"""


def load_larmatch_kps(loaders, current_entry, batchsize,
                      npairs=5000,
                      verbose=False,
                      exclude_neg_examples=False,
                      single_batch_mode=False):
    """
    loaders dict holding different larmatch data loaders. Should have following keys:
             "kps":instance of larflow.keypoints.LoaderKeypointData
             "affinity": affinity vector field (points along track direction)
    current_entry int the current entry we are on (if reading sequentially)
    batchsize int number of events to be in batch
    npairs int [default:5000] Number of larmatch triplets to return among all the triplets
               in the event.
    verbose [default: false} 
    exclude_neg_examples: sampled points only return true matches (in other words no ghost points)
    single_batch_mode: 
    """
    batch = []
    batch_npts_per_plane = []
    batch_tot_per_plane = [0,0,0]

    t_start = time.time()

    dtio = 0.0

    if single_batch_mode:
        batchsize = 1
    loaders["kps"].exclude_false_triplets( exclude_neg_examples )
    
    for ibatch in range(batchsize):
        ientry = current_entry + ibatch    
        data    = {"entry":ientry,
                   "tree_entry":int(ientry)%int(loaders["kps"].GetEntries())}


    
        # get data from match trees
        tio     = time.time()
        for name,loader in loaders.items():
            nbytes = loader.load_entry(data["tree_entry"])
            if verbose:
                print("[load_larmatch_kps] nbytes: ",nbytes," for tree[",name,"] entry=",data['tree_entry'])
        dtio    += time.time()-tio

        nfilled = c_int()
        nfilled.value = 0
        # the sparse image comes from the KPS loader
        spdata_v    = [ loaders["kps"].triplet_v[0].make_sparse_image( p ) for p in range(3) ]
        # sample the possible spacepoint matches
        matchdata   = loaders["kps"].sample_data( npairs, nfilled, True )
        # get the particle affinity field data
        pafdata = loaders["affinity"].get_match_data( matchdata["matchtriplet"], exclude_neg_examples )
        
        # add the contents to the data dictionary
        data.update(matchdata)
        data.update(pafdata)

        # prepare data dictionary
        npts_per_plane = [0,0,0]
        # separate the sparse charge image matrix into coordinates and features (the charge)
        for p in range(3):
            data["coord_%d"%(p)] = spdata_v[p][:,0:2].astype( dtype=np.int32 )
            # reshape and scale feature (i.e. pixel intensities)            
            data["feat_%d"%(p)]  = np.clip( spdata_v[p][:,2].reshape( (spdata_v[p].shape[0], 1) )/40.0, 0, 10.0 )
            npts_per_plane[p] = spdata_v[p].shape[0]
            batch_tot_per_plane[p] += npts_per_plane[p]
        batch_npts_per_plane.append(npts_per_plane)
        # split the spacepoint match information into the 3-plane sparse matrxi indices 
        data["matchpairs"]     = matchdata["matchtriplet"][:,0:3].astype( dtype=np.long )
        data["larmatchlabels"] = matchdata["matchtriplet"][:,3].astype( dtype=np.long )
        data["npairs"]         = nfilled.value
        # resetting the topological weights for ssnet
        nboundary = np.sum(data["ssnet_top_weight"][data["ssnet_top_weight"]==10.0])
        nvertex   = np.sum(data["ssnet_top_weight"][data["ssnet_top_weight"]==100.0])        
        data["ssnet_top_weight"][ data["ssnet_top_weight"]==10.0 ]  = 2.0
        data["ssnet_top_weight"][ data["ssnet_top_weight"]==100.0 ] = 5.0

        batch.append(data)

    if single_batch_mode:
        tottime = time.time()-t_start
        if verbose:
            print("[load larmatch kps single-batch sample]")
            print("  io time: %.3f secs, %.3f secs/batch"%(dtio, dtio/float(batchsize)))
            print("  tot time: %.3f secs. %.3f secs/batch"%(tottime, tottime/float(batchsize)))
        return batch[0]

        
    # build the combined tensor
    if batchsize>0:
        batch_coord = [ np.zeros( ( batch_tot_per_plane[p], 3 ), dtype=np.int32 ) for p in range(3) ]
        batch_feat  = [ np.zeros( ( batch_tot_per_plane[p], 1 ), dtype=np.int32 ) for p in range(3) ]    
        npts = [0,0,0]
        for ibatch in range(batchsize):
            for p in range(3):
                batch_coord[p][npts[p]:npts[p]+batch_npts_per_plane[ibatch][p],0:2] = batch[ibatch]["coord_%d"%(p)]
                batch_coord[p][npts[p]:npts[p]+batch_npts_per_plane[ibatch][p],2]   = ibatch            
                batch_feat[p][npts[p]:npts[p]+batch_npts_per_plane[ibatch][p],0]    = batch[ibatch]["feat_%d"%(p)][:,0]
                npts[p] += batch_npts_per_plane[ibatch][p]
    else:
        batch_coord = [ batch[0]["coord_%d"%(p)] for p in range(3) ]
        batch_feat  = [ batch[0]["feat_%d"%(p)]  for p in range(3) ]        


    batch_data = {"batchsize":batchsize,
                  "batch_npts_per_plane":batch_npts_per_plane,
                  "batch_tot_per_plane":batch_tot_per_plane }
    for p in range(3):
        batch_data["batch_coord_%d"%(p)] = batch_coord[p]
        batch_data["batch_feat_%d"%(p)]  = batch_feat[p]

    tottime = time.time()-t_start

    if verbose:
        print("[load larmatch kps sample]")
        print("  io time: %.3f secs, %.3f secs/batch"%(dtio, dtio/float(batchsize)))
        print("  tot time: %.3f secs. %.3f secs/batch"%(tottime, tottime/float(batchsize)))
        
    return batch_data

    
if __name__ == "__main__":


    from ROOT import std
    input_files = ["output_alldata.root"]
    device      = torch.device("cpu")

    input_v = std.vector("string")()
    for i in input_files:
        input_v.push_back(i)

    loaders = {"kps":larflow.keypoints.LoaderKeypointData( input_v ),
               "affinity":larflow.keypoints.LoaderAffinityField( input_v )}
    for name,loader in loaders.items():
        loader.exclude_false_triplets( False )
    nentries = loaders["kps"].GetEntries()
    print("num entries: ",nentries)

    nmax    = c_int()
    nfilled = c_int()
    nmax.value = 50000

    nentries = 10
    batchsize = 1
    
    for ientry in range(0,nentries,batchsize):
        print("[LOAD ENTRY ",ientry,"]")
        data = load_larmatch_kps( loaders, ientry, batchsize, exclude_neg_examples=False, verbose=True, single_batch_mode=True )
        for x,arr in data.items():
            if isinstance(arr,np.ndarray):
                print(x,": ",arr.shape,arr[:10])
                #print "   ",arr
            else:
                print(x,": ",arr)
        del data

    print("[fin]")
    raw_input()
