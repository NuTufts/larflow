import os,sys,time
from pushworker import BasePushWorker
import numpy as np
import zlib
import zmq

from larcv import larcv

import msgpack
import msgpack_numpy as m
m.patch()

os.environ["GLOG_minloglevel"] = "1"

from larcvdataset import LArCVDataset


class LArCV2Feeder( BasePushWorker ):
    """ This worker simply receives data and replies with dummy string. prints shape of array. """

    def __init__( self,configfile,fillername,identity,pullsocket_address,port=0,batchsize=None):
        super( LArCV2Feeder, self ).__init__(identity,pullsocket_address,port=port)
        self.configfile = configfile
        self.fillername = fillername
        self.batchsize = batchsize
        self.larcvloader = LArCVDataset(self.configfile,fillername)
        self.products = {}
        self.compression_level = 4
        self.print_msg_size = False
        self.num_reads = 0
        if self.batchsize is not None:
            self.start_dataloader(self.batchsize)
        print "LArCV2Feeder[{}] is loaded.".format(self._identity)

        
    def start_dataloader(self,batchsize):
        print "LArCV2Feeder[{}] starting loader w/ batchsize={}".format(self._identity,self.batchsize)
        self.batchsize = batchsize
        self.larcvloader.start(self.batchsize)
        print "LArCV2Feeder[{}] dataloader ready, loading first product set".format(self._identity,self.batchsize)
        while not self.larcvloader.io._proc.manager_started():
            time.sleep(1.0)            
            print "LArCV2Feeder[{}] waiting for larcv_threadio".format(self._identity)
        self.post_reply() # get first batch
        print "LArCV2Feeder[{}] manager started. syncing with client".format(self._identity)
        self.sync() # we notify the client we are ready

    def generate_reply(self):
        """
        our job is to return our data set, then load another
        """
        
        reply = [self._identity]
        totmsgsize = 0.0
        totcompsize = 0.0        
        for key,arr in self.products.items():
                
            # encode
            x_enc = msgpack.packb( arr, default=m.encode )
            x_comp = zlib.compress(x_enc,self.compression_level)

            # for debug: inspect compression gains (usually reduction to 1% or lower of original size)
            if self.print_msg_size:
                encframe = zmq.Frame(x_enc)
                comframe = zmq.Frame(x_comp)
                totmsgsize  += len(encframe.bytes)
                totcompsize += len(comframe.bytes)
                
            reply.append( key.encode('utf-8') )
            reply.append( x_comp )

        if self.print_msg_size:
            print "LArCV2Feeder[{}]: size of array portion={} MB (uncompressed {} MB)".format(self._identity,totcompsize/1.0e6,totmsgsize/1.0e6)
        return reply

    def isready(self):
        return self.larcvloader.io._proc.manager_started()

    def post_reply(self):
        """ load up the next data set. we've already sent out the message. so here we try to hide latency while gpu running. """
        
        # get data
        self.products = self.larcvloader[0]        
        #print "[",self.num_reads,":{}] ".format(self._identity),self.products.keys()
        self.num_reads += 1
        return
            

if __name__ == "__main__":
    """ test if worker loads properly """

    from server import Server
    import multiprocessing
    ##
    ## Contents of test file
    ## 
    """
*         ../../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root  
  KEY: TTree    image2d_infillCropped_tree;1    infillCropped tree
  KEY: TTree    image2d_larflow_y2u_tree;1      larflow_y2u tree
  KEY: TTree    image2d_adc_tree;1      adc tree
  KEY: TTree    image2d_pixvisi_tree;1  pixvisi tree
  KEY: TTree    image2d_pixflow_tree;1  pixflow tree
  KEY: TTree    image2d_larflow_y2v_tree;1      larflow_y2v tree
  KEY: TTree    image2d_ssnetCropped_shower_tree;1      ssnetCropped_shower tree
  KEY: TTree    image2d_ssnetCropped_endpt_tree;1       ssnetCropped_endpt tree
  KEY: TTree    image2d_ssnetCropped_track_tree;1       ssnetCropped_track tree
"""
    
    def start_worker(ident):
        worker = LArCV2Feeder("tester_flowloader_dualflow_train.cfg","ThreadProcessorTrain",ident,"localhost")
        worker.do_work()
        
    process = multiprocessing.Process(target=start_worker,args=(0,))
    process.daemon = True
    process.start()
        
    broker = Server( "*" )
    broker.start(-1.0)
    broker.stop()


    
