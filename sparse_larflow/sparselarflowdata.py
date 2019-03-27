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
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data

def load_sparse_larflowdata_sparseimg(io,remove_bg_labels=True):
    """
    we need the input data to be a pixel list
    however, the ground truth can be dense arrays

    products returned:
    pixplane[]
    flow[]2[]
    """

    nflows = 1
    threshold = 10.0
    data = {}

    # profiling variables
    tottime   = time.time()
    dtflow    = 0
    dtconvert = 0
    dtio      = 0
    dtnpmanip = 0

    tio = time.time()
    ev_sparse   = io.get_data(larcv.kProductSparseImage,"larflow_y2u")
    dtio += time.time()-tio

    sparsedata = ev_sparse.at(0)
    sparse_np  = larcv.as_ndarray( sparsedata, larcv.msg.kNORMAL )

    nfeatures = sparsedata.nfeatures()
    meta  = sparsedata.meta_v().front()

    if nflows==2:
        if nfeatures<=3:
            has_truth = False
        else:
            has_truth = True
    else:
        if nfeatures<=2:
            has_truth = False
        else:
            has_truth = True
        

    flows = [ ("yflow",sparse_np, meta) ]

    # cut on ADC values
    for (flowname,sparse_np,meta) in flows:

        tflowstart = time.time()
        #print "src+tar pix: shape=",srcandtarpix.shape
        #print "dense flowimg: shape=",flowimg.shape
        #print "pixellist: ",srcandtarpix[:,0:2].shape
        #print srcandtarpix[:20,0:2]

        tnpmanip  = time.time()
        if nflows==2:
            data["pix"+flowname] = sparse_np[:,0:5] # (row,col,src,tar1,tar2)
            if has_truth:
                data[flowname+"1"]   = sparse_np[:,5].astype(np.float32)
                data[flowname+"2"]   = sparse_np[:,6].astype(np.float32)
            else:
                data[flowname+"1"] = None
                data[flowname+"2"] = None
            dtnpmanip += time.time()-tnpmanip
        elif nflows==1:
            data["pix"+flowname] = sparse_np[:,0:4] # (row,col,src,tar)
            if has_truth:
                data[flowname+"1"] = sparse_np[:,4].astype(np.float32) # (truth1)
                data[flowname+"2"] = None                
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
    #print "io time: ",dtio
    #print "tot array manip time: ",tottime
    #print "  time for each flow: ",dtflow/len(flows)
    #print "    convert larcv2numpy per flow: ",dtconvert/len(flows)
    #print "    modify numpy arrays: ",(dtnpmanip)/len(flows)


    return data

def load_larflow_larcvdata( name, inputfile, batchsize, nworkers,
                            nflows=2,
                            tickbackward=False, readonly_products=None,
                            producer_name="larflow"):
    #feeder = LArCVServer(batchsize,name,load_sparse_larflowdata,inputfile,nworkers,
    #                     server_verbosity=0,worker_verbosity=2,
    #                     io_tickbackward=tickbackward,readonly_products=readonly_products)
    feeder = SparseLArFlowPyTorchDataset(inputfile, batchsize,
                                         tickbackward=tickbackward, nworkers=nworkers, nflows=nflows,
                                         producer_name=producer_name,
                                         readonly_products=readonly_products, feedername=name)
    return feeder

class SparseLArFlowPyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,batchsize,tickbackward=False,nworkers=4,
                 producer_name="larflow",
                 nflows=2,
                 readonly_products=None, feedername=None):
        super(SparseLArFlowPyTorchDataset,self).__init__()

        if type(inputfile) is str:
            self.inputfiles = [inputfile]
        elif type(inputfile) is list:
            self.inputfiles = inputfile

        if type(producer_name) is not str:
            raise ValueError("producer_name type must be str")
            
        # get length by querying the tree
        self.nentries  = 0
        tchain = rt.TChain("sparseimg_{}_tree".format(producer_name))
        for finput in self.inputfiles:
            tchain.Add(finput)
        self.nentries = tchain.GetEntries()
        print "nentries: ",self.nentries
        del tchain

        if feedername is None:
            self.feedername = "SparseLArFlowImagePyTorchDataset_%d"%\
                                (SparseImagePyTorchDataset.idCounter)
        else:
            self.feedername = feedername
        self.batchsize = batchsize
        self.nworkers  = nworkers
        self.nflows    = nflows
        self.feeder = LArCVServer(self.batchsize,self.feedername,
                                  #load_sparse_larflowdata,                                  
                                  load_sparse_larflowdata_sparseimg,
                                  self.inputfiles,self.nworkers,
                                  server_verbosity=-1,worker_verbosity=-1,
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
        if self.nflows==2:
            tarpix_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        else:
            tarpix_flow2_t = None
            
        # tensor for true flow
        if has_truth:
            truth_flow1_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
            if self.nflows==2:
                truth_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
            else:
                truth_flow2_t = None
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
            if self.nflows==2:
                tarpix_flow2_t[start:end,0] = torch.from_numpy(srcpix[:,4])
            if has_truth:
                truth_flow1_t[start:end,0]  = torch.from_numpy(trueflow1)
                if self.nflows==2:
                    truth_flow2_t[start:end,0]  = torch.from_numpy(trueflow2)
            nfilled += batchlen[ib]

        return {"coord":coord_t,"src":srcpix_t,
                "tar1":tarpix_flow1_t,"tar2":tarpix_flow2_t,
                "flow1":truth_flow1_t,"flow2":truth_flow2_t}


if __name__== "__main__":

    "testing"
    #inputfile = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    inputfile = "out_sparsified.root"
    batchsize = 1
    nworkers  = 2
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
