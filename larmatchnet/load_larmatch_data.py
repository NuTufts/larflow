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
Load LArMatch data for training and deploy
"""


def load_larmatch_data(io,has_truth=True,
                       npairs=50000,verbose=False,
                       matchtree=None,match_v=None,
                       producer="larflow"):
    
    data    = {}

    # profiling variables
    tottime   = time.time()
    dtio      = 0    
    dtflow    = 0
    dtconvert = 0
    dtnpmanip = 0

    # get data from iomanager
    tio        = time.time()
    ev_larflow = io.get_data( larcv.kProductSparseImage, producer )
    larflow_v  = ev_larflow.SparseImageArray()
    dtio       += time.time()-tio

    nbytes = matchtree.GetEntry( io.current_entry() )

    # convert sparse data into numpy arrays
    spdata  = [ larcv.as_sparseimg_ndarray( larflow_v.at(x), larcv.msg.kNORMAL ) for x in range(3) ]

    # make the different coordinate and feature tensors
    for ii,name in enumerate(["source","target1","target2"]):
        coord_t = spdata[ii][:,0:2].astype( dtype=np.int32 )
        feat_t  = spdata[ii][:,2].astype( dtype=np.float32 )
        data["coord_"+name] = coord_t
        data["feat_"+name]  = feat_t

    # sample pairs in the image
    nfilled1 = c_int()
    matchidx1 = larflow.sample_pair_array( npairs, match_v[0], nfilled1, has_truth ).astype(np.long)    
    nfilled2 = c_int()
    matchidx2 = larflow.sample_pair_array( npairs, match_v[1], nfilled2, has_truth ).astype(np.long)
    data["matchpairs_flow1"] = matchidx1
    data["npairs_flow1"]     = nfilled1.value
    data["matchpairs_flow2"] = matchidx2
    data["npairs_flow2"]     = nfilled2.value
    data["entry"]            = io.current_entry()
    if has_truth:
        data["labels_flow1"] = matchidx1[:,2]
        data["labels_flow2"] = matchidx2[:,2]
         

    if verbose:
        print "[load cropped sparse dual flow]"        
        print "  io time: %.3f secs"%(dtio)
        print "  tot time: %.3f secs"%(tottime)
        
    return data


class LArMatchDataset:

    def __init__( self, input_larcv_files, input_ana_files, npairs=20000, use_triplets=True ):

        self.input_larcv_files = input_larcv_files
        self.input_ana_files   = input_ana_files
        
        # load chain
        self.match_v = std.vector("larflow::FlowMatchMap")()
        self.tchain = rt.TChain("flowmatchdata")
        for fin in input_ana_files:
            print "adding ana file: ",fin
            self.tchain.Add( fin )
        self.tchain.SetBranchAddress( "matchmap", rt.AddressOf(self.match_v))
        print "chain has ",self.tchain.GetEntries()," entries"

        self.params = {"has_truth":True,
                       "verbose":False,
                       "npairs":npairs,
                       "matchtree":self.tchain,
                       "match_v":self.match_v}
        
        self.nworkers = 1
        self.feeder = LArCVServer(1,"larmatchfeed",
                                  load_larmatch_data,
                                  self.input_larcv_files,
                                  self.nworkers,
                                  io_tickbackward=False,
                                  func_params=self.params)

    def __len__(self):
        return int(self.params["matchtree"].GetEntries())

    def gettensorbatch(self,batchsize,device):

        batches = []
        source_npts  = []
        target1_npts = []
        target2_npts = []
        pair1_npts   = []
        pair2_npts   = []
        source_tot = 0
        target1_tot = 0
        target2_tot = 0
        pair1_tot   = 0
        pair2_tot   = 0
        for ibatch in range(batchsize):
            data = self.feeder.get_batch_dict()

            source_npts.append(  data["coord_source"][0].shape[0] )
            source_tot += source_npts[-1]
            
            target1_npts.append( data["coord_target1"][0].shape[0] )
            target1_tot += target1_npts[-1]

            target2_npts.append( data["coord_target2"][0].shape[0] )
            target2_tot += target2_npts[-1]

            pair1_npts.append(  int(data["npairs_flow1"][0]) )
            pair1_tot += pair1_npts[-1]

            pair2_npts.append(  int(data["npairs_flow2"][0]) )
            pair2_tot += pair2_npts[-1]            

            batches.append( data )

        tdata = {"coord_source": np.zeros( (source_tot,3),  dtype=np.int32 ),
                 "coord_target1":np.zeros( (target1_tot,3), dtype=np.int32 ),
                 "coord_target2":np.zeros( (target2_tot,3), dtype=np.int32 ),
                 "feat_source":  np.zeros( (source_tot,1),  dtype=np.float32 ),
                 "feat_target1": np.zeros( (target1_tot,1), dtype=np.float32 ),
                 "feat_target2": np.zeros( (target2_tot,1), dtype=np.float32 ),
                 "pairs_flow1":  [],  
                 "pairs_flow2":  [],  
                 "entries":[],
                 "npairs1":[],
                 "npairs2":[]
                 }

        source_start  = 0
        target1_start = 0
        target2_start = 0
        npair1_start = 0
        npair2_start = 0
        for ibatch,data in enumerate(batches):
            source_end  = source_start  + source_npts[ibatch]
            target1_end = target1_start + target1_npts[ibatch]
            target2_end = target2_start + target2_npts[ibatch]
            tdata["coord_source"][source_start:source_end,0:2]    = data["coord_source"][0][:,0:2]
            tdata["coord_target1"][target1_start:target1_end,0:2] = data["coord_target1"][0][:,0:2]
            tdata["coord_target2"][target2_start:target2_end,0:2] = data["coord_target2"][0][:,0:2]
            tdata["feat_source"][source_start:source_end,0]     = data["feat_source"][0][:]
            tdata["feat_target1"][target1_start:target1_end,0]  = data["feat_target1"][0][:]
            tdata["feat_target2"][target2_start:target2_end,0]  = data["feat_target2"][0][:]
            source_start  = source_end
            target1_start = target1_end
            target2_start = target2_end

            tdata["pairs_flow1"].append( torch.from_numpy(data["matchpairs_flow1"][0][0:pair1_npts[ibatch],:]).to(device) )
            tdata["pairs_flow2"].append( torch.from_numpy(data["matchpairs_flow2"][0][0:pair2_npts[ibatch],:]).to(device) )
            tdata["entries"].append( data["entry"][0] )
            tdata["npairs1"].append( data["npairs_flow1"][0] )
            tdata["npairs2"].append( data["npairs_flow2"][0] )
            
        for name,arr_np in tdata.items():
            if type(arr_np) is np.ndarray:
                tdata[name] = torch.from_numpy( arr_np ).to(device)
            
        return tdata
    
if __name__ == "__main__":


    input_larcv_files = ["/home/twongj01/data/larmatch_training_data/loose_positive_examples/larmatch_train_p00.root"]
    input_ana_files   = ["/home/twongj01/data/larmatch_training_data/loose_positive_examples/larmatch_train_p00.root"]
    #input_larcv_files = ["test_larcv.root"]
    #input_ana_files   = ["ana_flowmatch_data.root"]
    device = torch.device("cpu")

    io = LArMatchDataset( input_larcv_files, input_ana_files )
    
    data = io.gettensorbatch(1,device)
    for name,arr in data.items():
        if "pairs_flow" in name:
            print "len(%s):"%(name),len(arr)
        elif isinstance(arr,torch.Tensor):
            print name,arr.shape
        else:
            print name,arr
