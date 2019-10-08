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

if __name__ == "__main__":

    DEVICE = torch.device("cpu")
    
    model = LArMatch(neval=50000).to(DEVICE)
    print model

    match_v = std.vector("larflow::FlowMatchMap")()
    tchain = rt.TChain("flowmatchdata")
    tchain.Add( "ana_flowmatch_data.root" )
    tchain.SetBranchAddress( "matchmap", rt.AddressOf(match_v) )
    
    io = larcv.IOManager( larcv.IOManager.kREAD )
    io.add_in_file( "test_larcv.root" )
    io.initialize()

    io.read_entry(0)
    tchain.GetEntry(0)

    ev_larflow = io.get_data( larcv.kProductSparseImage, "larflow" )
    larflow_v = ev_larflow.SparseImageArray()
    nfalsepairs = np.reshape( tchain.nfalsepairs, (2) )
    ntruepairs  = np.reshape( tchain.ntruepairs, (2) )
    
    print "len(larflow_v)=",larflow_v.size()
    print "len(match_v)=",match_v.size()
    print "nfalsepairs=",nfalsepairs
    print "ntruepairs=",ntruepairs
    
    spsrc  = larcv.as_sparseimg_ndarray( larflow_v.at(0), larcv.msg.kNORMAL )
    sptar1 = larcv.as_sparseimg_ndarray( larflow_v.at(1), larcv.msg.kNORMAL )
    sptar2 = larcv.as_sparseimg_ndarray( larflow_v.at(2), larcv.msg.kNORMAL )

    print "source: ",spsrc.shape

    coord_src_t = torch.zeros( (spsrc.shape[0],3), dtype=torch.int32 ).to(DEVICE)
    coord_src_t[:,0:2] = torch.from_numpy(spsrc[:,0:2]).type(dtype=torch.int32).to(DEVICE)
    coord_src_t[:,2] = 0
    adc_src_t   = torch.from_numpy( spsrc[:,2].reshape( (spsrc.shape[0], 1) ) ).to(DEVICE)

    coord_tar1_t = torch.zeros( (sptar1.shape[0],3), dtype=torch.int32 ).to(DEVICE)
    coord_tar1_t[:,0:2] = torch.from_numpy( sptar1[:,0:2] ).type(dtype=torch.int32).to(DEVICE)
    coord_tar1_t[:,2]   = 0
    adc_tar1_t   = torch.from_numpy( sptar1[:,2].reshape( (sptar1.shape[0], 1) ) ).to(DEVICE)

    coord_tar2_t = torch.zeros( (sptar2.shape[0],3), dtype=torch.int32 ).to(DEVICE)
    coord_tar2_t[:,0:2] = torch.from_numpy( sptar2[:,0:2] ).type(dtype=torch.int32).to(DEVICE)
    coord_tar2_t[:,2]   = 0        
    adc_tar2_t   = torch.from_numpy( sptar2[:,2].reshape( (sptar2.shape[0], 1) ) ).to(DEVICE)

    start = time.time()
    with torch.set_grad_enabled(True):
        match1,match2 = model( coord_src_t, adc_src_t,
                               coord_tar1_t, adc_tar1_t,
                               coord_tar2_t, adc_tar2_t,
                               [match_v[0]], [match_v[1]], 1, DEVICE )
    
    print "output: ",match1.shape,match2.shape
    print "forward time: ",time.time()-start," secs"
    
    print "DONE"
    
    
