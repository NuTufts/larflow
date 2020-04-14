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


def load_larmatch_kps(loader, current_entry, batchsize,
                      npairs=5000,verbose=False,single_batch_mode=False):

    batch = []
    batch_npts_per_plane = []
    batch_tot_per_plane = [0,0,0]

    t_start = time.time()

    dtio = 0.0

    if single_batch_mode:
        batchsize = 1
    
    for ibatch in xrange(batchsize):
        ientry = current_entry + ibatch    
        data    = {"entry":ientry,
                   "tree_entry":int(ientry)%int(loader.GetEntries())}


    
        # get data from match tree
        tio     = time.time()
        nbytes  = loader.load_entry(data["tree_entry"])
        dtio    += time.time()-tio

        if verbose:
            print "nbytes: ",nbytes," for tree entry=",data['tree_entry']
        nfilled = c_int()
        nfilled.value = 0
        spdata_v    = [ loader.triplet_v[0].make_sparse_image( p ) for p in xrange(3) ]
        matchdata   = loader.sample_data( npairs, nfilled, True )
        data.update(matchdata)

        # prepare data dictionary
        npts_per_plane = [0,0,0]
        for p in xrange(3):
            data["coord_%d"%(p)] = spdata_v[p][:,0:2].astype( dtype=np.int32 )
            data["feat_%d"%(p)]  = spdata_v[p][:,2].reshape( (spdata_v[p].shape[0], 1) )
            npts_per_plane[p] = spdata_v[p].shape[0]
            batch_tot_per_plane[p] += npts_per_plane[p]
        batch_npts_per_plane.append(npts_per_plane)
        data["matchpairs"]     = matchdata["matchtriplet"][:,0:3].astype( dtype=np.long )
        data["larmatchlabels"] = matchdata["matchtriplet"][:,3].astype( dtype=np.long )
        data["npairs"]     = nfilled.value
        nboundary = np.sum(data["ssnet_top_weight"][data["ssnet_top_weight"]==10.0])
        nvertex   = np.sum(data["ssnet_top_weight"][data["ssnet_top_weight"]==100.0])        
        data["ssnet_top_weight"][ data["ssnet_top_weight"]==10.0 ]  = 2.0
        data["ssnet_top_weight"][ data["ssnet_top_weight"]==100.0 ] = 5.0
        print "nboundary=",nboundary," nvertex=",nvertex

        batch.append(data)

    if single_batch_mode:
        tottime = time.time()-t_start
        if verbose:
            print "[load larmatch kps single-batch sample]"        
            print "  io time: %.3f secs, %.3f secs/batch"%(dtio, dtio/float(batchsize))
            print "  tot time: %.3f secs. %.3f secs/batch"%(tottime, tottime/float(batchsize))        
        return batch[0]

        
    # build the combined tensor
    if batchsize>0:
        batch_coord = [ np.zeros( ( batch_tot_per_plane[p], 3 ), dtype=np.int32 ) for p in xrange(3) ]
        batch_feat  = [ np.zeros( ( batch_tot_per_plane[p], 1 ), dtype=np.int32 ) for p in xrange(3) ]    
        npts = [0,0,0]
        for ibatch in xrange(batchsize):
            for p in xrange(3):
                batch_coord[p][npts[p]:npts[p]+batch_npts_per_plane[ibatch][p],0:2] = batch[ibatch]["coord_%d"%(p)]
                batch_coord[p][npts[p]:npts[p]+batch_npts_per_plane[ibatch][p],2]   = ibatch            
                batch_feat[p][npts[p]:npts[p]+batch_npts_per_plane[ibatch][p],0]    = batch[ibatch]["feat_%d"%(p)][:,0]
                npts[p] += batch_npts_per_plane[ibatch][p]
    else:
        batch_coord = [ batch[0]["coord_%d"%(p)] for p in xrange(3) ]
        batch_feat  = [ batch[0]["feat_%d"%(p)]  for p in xrange(3) ]        


    batch_data = {"batchsize":batchsize,
                  "batch_npts_per_plane":batch_npts_per_plane,
                  "batch_tot_per_plane":batch_tot_per_plane }
    for p in xrange(3):
        batch_data["batch_coord_%d"%(p)] = batch_coord[p]
        batch_data["batch_feat_%d"%(p)]  = batch_feat[p]

    tottime = time.time()-t_start

    if verbose:
        print "[load larmatch kps sample]"        
        print "  io time: %.3f secs, %.3f secs/batch"%(dtio, dtio/float(batchsize))
        print "  tot time: %.3f secs. %.3f secs/batch"%(tottime, tottime/float(batchsize))        
        
    return batch_data

    
if __name__ == "__main__":


    from ROOT import std
    input_files = ["larmatch_keypointssnet_small_sample_test.root"]
    device      = torch.device("cpu")

    input_v = std.vector("string")()
    for i in input_files:
        input_v.push_back(i)

    loader = larflow.keypoints.LoaderKeypointData( input_v )
    loader.exclude_false_triplets( False )
    nentries = loader.GetEntries()
    print "num entries: ",nentries

    nmax    = c_int()
    nfilled = c_int()
    nmax.value = 50000

    nentries = 1000000000
    batchsize = 1
    
    for ientry in xrange(0,nentries,batchsize):
        print "[LOAD ENTRY ",ientry,"]"
        data = load_larmatch_kps( loader, ientry, batchsize, verbose=True, single_batch_mode=True )
        for x,arr in data.items():
            if isinstance(arr,np.ndarray):
                print x,": ",arr.shape,arr[:10]
                #print "   ",arr
            else:
                print x,": ",arr
        del data

    print "[fin]"
    raw_input()
