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
                  
        # note, with way LArCVServer workers, must always use batch size of 1
        #   because larcvserver expects entries in each batch to be same size,
        #   but in sparse representations this is not true
        # we must put batches together ourselves for sparseconv operations
        self.feeder = LArCVServer(1,self.feedername,
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

        # we will fill this dict to return with batch
        datalen   = [] # store length of each sparse data instance
        ncoords   = 0  # total number of points over all batches
        flow='dual'    # flow type, turn into option later if needed        

        # first collect data
        data_v = []
        for ibatch in xrange(self.batchsize):
            batch = None
            ntries = 0
            while batch is None and ntries<10:
                batch = self.feeder.get_batch_dict()
                ntries += 1
            if batch is not None:
                data_v.append( batch )


        # now calc total points in each sparse image instance
        for data in data_v:
            datalen.append( data["pixadc"][0].shape[0] )
            ncoords += datalen[-1]
        #print "NCOORDS: ",ncoords

        if len(data_v)>0 and data_v[0]["flowy2u"][0] is not None:
            has_truth = True
        else:
            has_truth = False

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

        # fill tensors above
        nfilled = 0            
        for ib,batch in enumerate(data_v):
            srcpix    = batch["pixadc"][0]
            trueflow1 = batch["flowy2u"][0]
            trueflow2 = batch["flowy2v"][0]
            #print type(srcpix),
            #print srcpix.shape," "
            #print trueflow1.shape," "
            #print trueflow2.shape," "            
            
            start = nfilled
            end   = nfilled+datalen[ib]
            coord_t[start:end,0:2] \
                = torch.from_numpy( srcpix[:,0:2].astype(np.int) )
            coord_t[start:end,2]        = ib
            srcpix_t[start:end,0]       = torch.from_numpy(srcpix[:,2])
            tarpix_flow1_t[start:end,0] = torch.from_numpy(srcpix[:,3])
            tarpix_flow2_t[start:end,0] = torch.from_numpy(srcpix[:,4])

            if has_truth:
                truth_flow1_t[start:end,0]      = torch.from_numpy(trueflow1[:,0])
                if truth_flow2_t is not None:
                    truth_flow2_t[start:end,0]  = torch.from_numpy(trueflow2[:,0])
                
            nfilled += datalen[ib]

        flowdata = {"coord":coord_t,      "src":srcpix_t,
                    "tar1":tarpix_flow1_t,"tar2":tarpix_flow2_t,
                    "flow1":truth_flow1_t,"flow2":truth_flow2_t}
        
        return flowdata
            

if __name__== "__main__":

    "testing"
    #inputfile = "../testdata/mcc9mar_bnbcorsika/larcv_mctruth_ee881c25-aeca-4c92-9622-4c21f492db41.root"
    #inputfile = "out_sparsified.root"
    inputfiles = ["/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train1_v3.root",
                  "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train2_v3.root",
                  "/mnt/hdd1/twongj01/sparse_larflow_data/larflow_sparsify_cropped_train3_v3.root"]
    batchsize = 10
    nworkers  = 3
    tickbackward = True
    readonly_products=None
    nentries = 10

    TEST_VANILLA = True

    if TEST_VANILLA:
        feeder = load_larflow_larcvdata( "larflowsparsetest", inputfiles, batchsize, nworkers,
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
