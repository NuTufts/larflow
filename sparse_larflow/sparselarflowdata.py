import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata
from load_cropped_sparse_dualflow import load_cropped_sparse_dualflow



def load_larflow_larcvdata( name, inputfile, batchsize, nworkers,
                            nflows=2,
                            tickbackward=False, readonly_products=None,
                            producer_name="sparsecropdual"):
    feeder = SparseLArFlowPyTorchDataset(inputfile, batchsize,
                                         tickbackward=tickbackward, nworkers=nworkers, nflows=nflows,
                                         producer_name=producer_name,
                                         readonly_products=readonly_products,
                                         feedername=name)
    return feeder


###################################
## SparseLArFlowPyTorchDataset
###################################
class SparseLArFlowPyTorchDataset(torchdata.Dataset):
    idCounter = 0
    def __init__(self,inputfile,batchsize,tickbackward=False,nworkers=4,
                 producer_name="sparsecropdual",
                 nflows=2,
                 readonly_products=None,
                 feedername=None):
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
        #print "nentries: ",self.nentries
        del tchain

        if feedername is None:
            self.feedername = "SparseLArFlowImagePyTorchDataset_%d"%\
                                (SparseImagePyTorchDataset.idCounter)
        else:
            self.feedername = feedername
        self.batchsize = batchsize
        self.nworkers  = nworkers
        self.nflows    = nflows
        readonly_products = None
        params = {"producer":producer_name}
                  
        
        self.feeder = LArCVServer(self.batchsize,self.feedername,
                                  load_cropped_sparse_dualflow,
                                  self.inputfiles,self.nworkers,
                                  server_verbosity=-1,worker_verbosity=-1,
                                  io_tickbackward=tickbackward,
                                  func_params=params)

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

        output
        -------
        flowdatay2u [dict of torch tensors]
        flowdatay2v [dict of torch tensors]
        """

        # get batch using data loading functions above
        batch = self.feeder.get_batch_dict()

        flowdata = {}

        flow='dual'        

        ncoords  = 0
        batchlen = []

        # get reference tensor
        ref = batch["pixadc"]
        
        for ib,srcpix in enumerate(ref):
            #print "batch[{}] srcpix={}".format(ib,srcpix.shape)
            batchlen.append( srcpix.shape[1] )
            ncoords += batchlen[-1]

                
        if batch["flowy2u"][0] is not None:
            has_truth = True
        else:
            has_truth = False

        #print "get_tensor_batch: ncoords={} batchlen={}".format( ncoords, batchlen )

        # make tensor for coords (row,col,batch)
        coord_t = torch.zeros( (ncoords,3), dtype=torch.int ).to(device)

        # tensor for src pixel adcs
        srcpix_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        # tensor for target pixel adcs
        tarpix_flow1_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        if flow=='dual':
            tarpix_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
        else:
            tarpix_flow2_t = None
            
        # tensor for true flow
        if has_truth:
            truth_flow1_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
            if flow=='dual':
                truth_flow2_t = torch.zeros( (ncoords,1), dtype=torch.float).to(device)
            else:
                truth_flow2_t = None
        else:
            truth_flow1_t = None
            truth_flow2_t = None                

        if flow=='dual':
            data = zip(batch["pixadc"],batch["flowy2u"],batch["flowy2v"])
        else:
            raise ValueError("Separate flows not ready yet")

        nfilled = 0
        for ib,(srcpix,trueflow1,trueflow2) in enumerate(data):
            start = nfilled
            end   = nfilled+batchlen[ib]
            coord_t[start:end,0:2] \
                = torch.from_numpy( srcpix[ib,:,0:2].astype(np.int) )
            coord_t[start:end,2] = ib
            srcpix_t[start:end,0]       = torch.from_numpy(srcpix[ib,:,2])
            tarpix_flow1_t[start:end,0] = torch.from_numpy(srcpix[ib,:,3])
            tarpix_flow2_t[start:end,0] = torch.from_numpy(srcpix[ib,:,4])            

            if has_truth:
                truth_flow1_t[start:end,0]  = torch.from_numpy(trueflow1[ib,:,0])
                if truth_flow2_t is not None:
                    truth_flow2_t[start:end,0]  = torch.from_numpy(trueflow2[ib,:,0])
                
            nfilled += batchlen[ib]

        flowdata = {"coord":coord_t,"src":srcpix_t,
                    "tar1":tarpix_flow1_t,"tar2":tarpix_flow2_t,
                    "flow1":truth_flow1_t,"flow2":truth_flow2_t}
        
        return flowdata
            

if __name__== "__main__":

    "testing"
    #inputfile = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    #inputfile = "out_sparsified.root"
    inputfile = "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train.root"
    batchsize = 1
    nworkers  = 2
    tickbackward = True
    readonly_products=None
    nentries = 10

    TEST_VANILLA = True

    if TEST_VANILLA:
        feeder = load_larflow_larcvdata( "larflowsparsetest", inputfile, batchsize, nworkers,
                                         tickbackward=tickbackward,
                                         readonly_products=readonly_products,
                                         producer_name="sparsecropdual" )
        tstart = time.time()

        print "TEST LARFLOW LARCVDATASET SERVER"
        for n in xrange(nentries):
            print "=============================================="
            batch = feeder.get_tensor_batch(torch.device("cpu"))
            print "ENTRY[",n,"] from ",batch.keys()
            for name,arr in batch.items():
                print "  ",name," ",type(arr),": npts=",len(arr),"; ",
                if type(arr) is np.ndarray or type(arr) is torch.Tensor:
                    print arr.shape,
                print

        tend = time.time()-tstart
        print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
        del feeder
