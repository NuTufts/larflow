import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata

def load_sparse_larflowdata(io,remove_bg_labels=True):
    """
    we need the input data to be a pixel list
    however, the ground truth can be dense arrays

    products returned:
    pixplane[]
    flow[]2[]
    """

    threshold = 10.0
    data = {}

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    tio = time.time()
    ev_wire   = io.get_data(larcv.kProductImage2D,"wiremc")
    ev_flow   = io.get_data(larcv.kProductImage2D,"larflow")
    dtio += time.time()-tio

    nimgs = ev_wire.Image2DArray().size()
    meta  = ev_wire.Image2DArray().front().meta()

    if ev_flow.Image2DArray().size()==0:
        has_truth = False
    else:
        has_truth = True

    #flows = [("uflow",0,1,2),("vflow",0,2),("yflow",1,0)]
    flows = [("yflow",2,0,1,(4,5))]


    # cut on ADC values
    for (flowname,src_plane,tar1_plane,tar2_plane,idx) in flows:

        tflowstart = time.time()

        tconvert = time.time()
        srcandtarpix = larcv.as_union_pixelarray( ev_wire.Image2DArray().at(src_plane),
                                                  ev_wire.Image2DArray().at(tar1_plane),
                                                  ev_wire.Image2DArray().at(tar2_plane),
                                                  threshold, larcv.msg.kNORMAL )
        if has_truth:
            flow1 = np.transpose( larcv.as_ndarray( ev_flow.Image2DArray().at(idx[0]) ), (1,0) )
            flow2 = np.transpose( larcv.as_ndarray( ev_flow.Image2DArray().at(idx[1]) ), (1,0) )
        dtconvert += time.time()-tconvert

        #print "src+tar pix: shape=",srcandtarpix.shape
        #print "dense flowimg: shape=",flowimg.shape
        #print "pixellist: ",srcandtarpix[:,0:2].shape
        #print srcandtarpix[:20,0:2]

        tnpmanip  = time.time()
        data["pix"+flowname] = srcandtarpix
        if has_truth:
            data[flowname+"1"]   = flow1[ srcandtarpix[:,0].astype(np.int), srcandtarpix[:,1].astype(int) ]
            data[flowname+"2"]   = flow2[ srcandtarpix[:,0].astype(np.int), srcandtarpix[:,1].astype(int) ]
        else:
            data[flowname+"1"] = None
            data[flowname+"2"] = None
        dtnpmanip += time.time()-tnpmanip

        #data["flow"+flowname]   = larcv.as_pixelarray_with_selection( ev_flow.Image2DArray().at(idx),
        #                                                              ev_wire.Image2DArray().at(src_plane),
        #                                                              threshold, True,
        #                                                              larcv.msg.kNORMAL  )
        # remove no-flow pixels
        #print "flow"+flowname,"=",data["flow"+flowname].shape,
        #print ";srcpix"+flowname,"=",data["srcpix"+flowname].shape,
        #print ";tarpix"+flowname,"=",data["tarpix"+flowname].shape
        dtflow += time.time()-tflowstart

    tottime = time.time()-tottime
    print "io time: ",dtio
    print "tot array manip time: ",tottime
    print "  time for each flow: ",dtflow/len(flows)
    print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data


def load_larflow_larcvdata( name, inputfile, batchsize, nworkers,
                            tickbackward=False, readonly_products=None ):
    #feeder = LArCVServer(batchsize,name,load_sparse_larflowdata,inputfile,nworkers,
    #                     server_verbosity=0,worker_verbosity=2,
    #                     io_tickbackward=tickbackward,readonly_products=readonly_products)
    feeder = SparseLArFlowPyTorchDataset(inputfile, batchsize,
                    tickbackward=tickbackward, nworkers=nworkers,
                    readonly_products=readonly_products, feedername=name)
    return feeder

class SparseLArFlowPyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,batchsize,tickbackward=False,nworkers=4,
                    readonly_products=None, feedername=None):
        super(SparseLArFlowPyTorchDataset,self).__init__()

        if type(inputfile) is str:
            self.inputfiles = [inputfile]
        elif type(inputfile) is list:
            self.inputfiles = inputfile

        # get length by querying the tree
        self.nentries  = 0
        tchain = rt.TChain("image2d_wire_tree")
        for finput in self.inputfiles:
            tchain.Add(finput)
        self.nentries = tchain.GetEntries()
        del tchain

        if feedername is None:
            self.feedername = "SparseLArFlowImagePyTorchDataset_%d"%\
                                (SparseImagePyTorchDataset.idCounter)
        else:
            self.feedername = feedername
        self.batchsize = batchsize
        self.nworkers  = nworkers
        self.feeder = LArCVServer(self.batchsize,self.feedername,
                                  load_sparse_larflowdata,self.inputfiles,
                                  self.nworkers,
                                  server_verbosity=0,worker_verbosity=2,
                                  io_tickbackward=tickbackward,
                                  readonly_products=readonly_products)
        SparseLArFlowPyTorchDataset.idCounter += 1

    def __len__(self):
        #print "return length of sample:",self.nentries
        return self.nentries

    def __getitem__(self,index):
        """ we do not have a way to get the index (can change that)"""
        #print "called get item for index=",index," ",self.feeder.identity,"pid=",os.getpid()
        data = self.feeder.get_batch_dict()
        # remove the feeder variable
        del data["feeder"]
        #print "called get item: ",data.keys()
        return data


    def get_tensor_batch(self, device):
        """
        get batch, convert into torch tensors

        inputs
        -------
        device: torch.device specifies either gpu or cpu
        """

        batch = self.feeder.get_batch_dict()
        ncoords  = 0
        batchlen = []
        for ib,srcpix in enumerate(batch["pixyflow"]):
            batchlen.append( srcpix.shape[0] )
            ncoords += batchlen[-1]

        if batch["yflow1"][0] is not None:
            has_truth = True
        else:
            has_truth = False

        # make tensor for coords
        coord_t = torch.zeros( (ncoords,3), dtype=torch.int ).to(device)

        # tensor for src pixel adcs
        srcpix_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        # tensor for target pixel adcs
        tarpix_flow1_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        tarpix_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        # tensor for true flow
        if has_truth:
            truth_flow1_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
            truth_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        else:
            truth_flow1_t = None
            truth_flow2_t = None

        flowdata = zip(batch["pixyflow"],batch["yflow1"],batch["yflow2"])

        nfilled = 0
        for ib,(srcpix,trueflow1,trueflow2) in enumerate(flowdata):
            start = nfilled
            end   = nfilled+batchlen[ib]
            coord_t[start:end,0:2] \
                = torch.from_numpy( srcpix[:,0:2].astype(np.int) )
            coord_t[start:end,2] = ib
            srcpix_t[start:end,0]       = torch.from_numpy(srcpix[:,2])
            tarpix_flow1_t[start:end,0] = torch.from_numpy(srcpix[:,3])
            tarpix_flow2_t[start:end,0] = torch.from_numpy(srcpix[:,4])
            if has_truth:
                truth_flow1_t[start:end,0]  = torch.from_numpy(trueflow1)
                truth_flow2_t[start:end,0]  = torch.from_numpy(trueflow2)
            nfilled += batchlen[ib]

        return {"coord":coord_t,"src":srcpix_t,
                "tar1":tarpix_flow1_t,"tar2":tarpix_flow2_t,
                "flow1":truth_flow1_t,"flow2":truth_flow2_t}


if __name__== "__main__":

    "testing"
    inputfile = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    batchsize = 1
    nworkers  = 6
    tickbackward = True
    readonly_products=( ("wiremc",larcv.kProductImage2D),
                        ("larflow",larcv.kProductImage2D) )

    nentries = 10

    TEST_VANILLA = True

    if TEST_VANILLA:
        feeder = load_larflow_larcvdata( "larflowsparsetest", inputfile, batchsize, nworkers,
                                         tickbackward=tickbackward,readonly_products=readonly_products )
        tstart = time.time()

        print "TEST LARFLOW LARCVDATASET SERVER"
        for n in xrange(nentries):
            batch = feeder.get_batch_dict()
            print "ENTRY[",n,"] from ",batch["feeder"]
            for name,arr in batch.items():
                print "  ",name,": batch=",len(arr),"; ",
                if type(arr[0]) is np.ndarray:
                    print arr[0].shape
                else:
                    print arr
        tend = time.time()-tstart
        print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
        del feeder
