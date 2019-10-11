from __future__ import print_function
import os,sys,argparse

from ctypes import c_int
import numpy as np

import ROOT as rt
from ROOT import std
from larcv import larcv
from larflow import larflow
import torch

# dataset interface
from larcvdataset.larcvserver import LArCVServer

from larmatch import LArMatch
from load_larmatch_data import LArMatchDataset

inputfile = "test_larcv.root"
anafile   = "ana_flowmatch_data.root"

NUM_PAIRS=20000
DEVICE=torch.device("cpu")

preplarmatch = larflow.PrepFlowMatchData("deploy")
preplarmatch.setADCproducer("wiremc");
preplarmatch.initialize()

model = LArMatch(neval=NUM_PAIRS).to(DEVICE)

io = larcv.IOManager( larcv.IOManager.kREAD )
io.add_in_file( inputfile )
io.initialize()

#tchain = rt.TChain("flowmatchdata")
#tchain.AddFile( anafile )
#flowdata_v = std.vector("larflow::FlowMatchMap")()
#tchain.SetBranchAddress("matchmap",rt.AddressOf(flowdata_v))

ientry=0

io.read_entry(ientry)
#tchain.GetEntry(ientry)

print("Entry {}".format(ientry))

preplarmatch.process( io )
flowdata_v =  preplarmatch.getMatchData()
sparseimg_v = io.get_data(larcv.kProductSparseImage,"larflow")

nsparsepts = sparseimg_v.at(0).len()
source_np  = larcv.as_sparseimg_ndarray( sparseimg_v.at(0) )
target1_np = larcv.as_sparseimg_ndarray( sparseimg_v.at(1) )
target2_np = larcv.as_sparseimg_ndarray( sparseimg_v.at(2) )

coord_src_t  = torch.from_numpy( source_np[:,0:2].astype(np.long) )
coord_tar1_t = torch.from_numpy( target1_np[:,0:2].astype(np.long) )
coord_tar2_t = torch.from_numpy( target2_np[:,0:2].astype(np.long) )
feat_src_t   = torch.from_numpy( source_np[:,2].reshape(  (coord_src_t.shape[0], 1) ) )
feat_tar1_t  = torch.from_numpy( target1_np[:,2].reshape( (coord_tar1_t.shape[0],1) ) )
feat_tar2_t  = torch.from_numpy( target2_np[:,2].reshape( (coord_tar2_t.shape[0],1) ) )

print("number of flowmaps: {}".format(flowdata_v.size()))
print("num sparse indices=",nsparsepts)

sparse_index1 = 0
sparse_index2 = 0

while sparse_index1<nsparsepts and sparse_index2<nsparsepts:

    npairs1       = c_int()
    npairs2       = c_int()
    npairs1.value = 0
    npairs2.value = 0

    num_sparse_index1 = c_int()
    num_sparse_index2 = c_int()
    num_sparse_index1.value = 0
    num_sparse_index2.value = 0
    
    matchpair1 = larflow.get_chunk_pair_array( sparse_index1, NUM_PAIRS, flowdata_v.at(0), num_sparse_index1, npairs1 )
    matchpair2 = larflow.get_chunk_pair_array( sparse_index2, NUM_PAIRS, flowdata_v.at(1), num_sparse_index2, npairs2 )
    
    print("max(matchpair1)=",np.max(matchpair1))
    print("sparse_index1=",sparse_index1," npairs1_filled=",npairs1.value)
    print("sparse_index2=",sparse_index2," npairs2_filled=",npairs2.value)
    matchpair1_t = torch.from_numpy( matchpair1 )
    matchpair2_t = torch.from_numpy( matchpair2 )

    pred1_t, pred2_t = model( coord_src_t,  feat_src_t,
                              coord_tar1_t, feat_tar1_t,
                              coord_tar2_t, feat_tar2_t,
                              [matchpair1_t], [matchpair2_t], 1,
                              torch.device("cpu"), npts1=npairs1.value, npts2=npairs2.value )
    print("pred1_t=",pred1_t.shape," pred2_t=",pred2_t.shape)
    sparse_index1 += num_sparse_index1.value+1
    sparse_index2 += num_sparse_index2.value+1


print("DONE")
