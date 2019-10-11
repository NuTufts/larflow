from __future__ import print_function
import os,sys,argparse

from ctypes import c_int,c_double
import numpy as np

import ROOT as rt
from ROOT import std
from larlite import larlite,larutil
from larcv import larcv
from larflow import larflow
import torch

# dataset interface
from larcvdataset.larcvserver import LArCVServer

from larmatch import LArMatch
from load_larmatch_data import LArMatchDataset

inputfile = "test_larcv.root"
anafile   = "ana_flowmatch_data.root"

print(larutil.Geometry.GetME())
driftv = larutil.LArProperties.GetME().DriftVelocity()

checkpointfile = "train1/checkpoint.33000th.tar"
checkpoint = torch.load( checkpointfile, map_location={"cuda:0":"cpu",
                                                       "cuda:1":"cpu"} )
NUM_PAIRS=20000
DEVICE=torch.device("cpu")

preplarmatch = larflow.PrepFlowMatchData("deploy")
preplarmatch.setADCproducer("wiremc");
preplarmatch.initialize()

model = LArMatch(neval=NUM_PAIRS).to(DEVICE)
model.load_state_dict(checkpoint["state_dict"])

print("loaded MODEL")

io = larcv.IOManager( larcv.IOManager.kREAD )
io.add_in_file( inputfile )
io.initialize()

out = larlite.storage_manager( larlite.storage_manager.kWRITE )
out.set_out_filename( "output_larmatch.root" )
out.open()

evout_lfhits = out.get_data(larlite.data.kLArFlow3DHit,"larmatch")

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
adc_v       = io.get_data(larcv.kProductImage2D,"wiremc").Image2DArray()
srcmeta = adc_v.at(2).meta()

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

sigmoid = torch.nn.Sigmoid()

y = c_double()
z = c_double()

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
    prob1 = sigmoid(pred1_t)
    prob2 = sigmoid(pred2_t)

    # make 3d points from flow
    for ipair in xrange(pred1_t.shape[2]):
        p = prob1[0,0,ipair].item()
        if p<0.5:
            continue
        
        srcindex = matchpair1_t[ipair,0]
        tarindex = matchpair1_t[ipair,1]
        src_col = source_np[srcindex,1]
        tar_col = target1_np[tarindex,1]
        src_row = source_np[srcindex,0]
        x = (srcmeta.pos_y( int(src_row) )-3200)*0.5*driftv
        #print( (src_col,tar_col),": prob=",p)
        larutil.Geometry.GetME().IntersectionPoint( int(src_col), int(tar_col), 2, 0, y, z )
        lfhit = larlite.larflow3dhit()
        lfhit.resize(3,0)
        lfhit[0] = x
        lfhit[1] = y.value
        lfhit[2] = z.value
        evout_lfhits.push_back( lfhit )
    
    sparse_index1 += num_sparse_index1.value+1
    sparse_index2 += num_sparse_index2.value+1

print("save hits: ",evout_lfhits.size())
out.set_id( io.event_id().run(), io.event_id().subrun(), io.event_id().event() )
out.next_event(True)
out.close()

print("DONE")
