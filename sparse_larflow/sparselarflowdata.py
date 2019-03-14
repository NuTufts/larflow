import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer
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


    flows = [("u2v",0,1),("u2y",0,2),("v2u",1,0),("v2y",1,2),("y2u",2,0),("y2v",2,1)]

    
    # cut on ADC values
    for idx,flow in enumerate(flows):

        tflowstart = time.time()

        tconvert = time.time()
        flowname  = flow[0]
        src_plane = flow[1]
        tar_plane = flow[2]
        srcandtarpix = larcv.as_union_pixelarray( ev_wire.Image2DArray().at(src_plane),
                                                  ev_wire.Image2DArray().at(tar_plane),
                                                  threshold, larcv.msg.kNORMAL )
        flowimg = np.transpose( larcv.as_ndarray( ev_flow.Image2DArray().at(idx) ), (1,0) )
        dtconvert += time.time()-tconvert
        
        #print "src+tar pix: shape=",srcandtarpix.shape
        #print "dense flowimg: shape=",flowimg.shape
        #print "pixellist: ",srcandtarpix[:,0:2].shape
        #print srcandtarpix[:20,0:2]

        tnpmanip  = time.time()
        data["srcpix"+flowname] = srcandtarpix[:,0:3]
        data["tarpix"+flowname] = np.zeros( data["srcpix"+flowname].shape )
        data["tarpix"+flowname][:,0:2] = srcandtarpix[:,0:2]
        data["tarpix"+flowname][:,2]   = srcandtarpix[:,3]        
        data["flow"+flowname]   = flowimg[ srcandtarpix[:,0].astype(np.int), srcandtarpix[:,1].astype(int) ]
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
    feeder = LArCVServer(batchsize,name,load_sparse_larflowdata,inputfile,nworkers,
                         server_verbosity=0,worker_verbosity=2,
                         io_tickbackward=tickbackward,readonly_products=readonly_products)
    return feeder

class SparseImagePyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,batchsize,tickbackward=False,nworkers=4):
        super(SparseImagePyTorchDataset,self).__init__()

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
        
        self.feedername = "SparseImagePyTorchDataset_%d"%(SparseImagePyTorchDataset.idCounter)
        self.batchsize = batchsize
        self.nworkers  = nworkers
        self.feeder = LArCVServer(self.batchsize,self.feedername,
                                  load_sparse_ssnetdata,self.inputfiles,self.nworkers,
                                  server_verbosity=0,worker_verbosity=0,
                                  io_tickbackward=tickbackward)
        SparseImagePyTorchDataset.idCounter += 1

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
        
        

if __name__ == "__main__":
    
    "testing"
    inputfile = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    batchsize = 1
    nworkers  = 6
    tickbackward = True
    readonly_products=( ("wiremc",larcv.kProductImage2D),
                        ("larflow",larcv.kProductImage2D) )
    
    nentries = 10

    TEST_VANILLA = True
    TEST_PYTORCH_DATALOADER = False

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


    if TEST_PYTORCH_DATALOADER:
        print "TEST PYTORCH DATALOADER SERVER"
        print "DOES NOT WORK"
        params = {"batch_size":4,
                  "shuffle":True,
                  "pin_memory":True,
                  "num_workers":4}
        dataset = SparseImagePyTorchDataset(inputfile,tickbackward=True)
        pyloader = torchdata.DataLoader(dataset,**params)

        ientry = 0
        for batch in pyloader:
            print "entry[",n,"]: ",type(batch)
            ientry += 1
            if ientry>50:
                break
        tend = time.time()-tstart
        print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
        del pyloader


