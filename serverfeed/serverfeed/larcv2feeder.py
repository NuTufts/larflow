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
        if self.batchsize is not None:
            self.start_dataloader(self.batchsize)
        print "LArCV2Feeder[{}] is loaded.".format(self._identity)
        
        
    def process_message(self, frames ):
        """ 
        the client message is just a request for data. the batchsize requested is provide. it CANNOT change. there is really nothing for us to parse.
        """
        batchsize = int(frames[0].decode("ascii"))
        if self.batchsize is None:
            self.batchsize = batchsize
            self.larcvloader.start(self.batchsize)
            # get first batch
            self.post_reply()
        elif self.batchsize!=batchsize:
            raise RuntimeError("Cannot change batchsize!!")
            
        return

    def start_dataloader(self,batchsize):
        self.batchsize = batchsize
        self.larcvloader.start(self.batchsize)
        self.post_reply()

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

    def post_reply(self):
        """ load up the next data set. we've already sent out the message. so here we try to hide latency while gpu running. """
        
        # get data
        data = self.larcvloader[0]
        
        batchsize = self.batchsize
        # hack for now. how to do this?
        width  = 832 
        height = 512

        # make torch tensors from numpy arrays
        index = (0,1,2,3)
        self.products = {}
        #self.products['test'] = np.zeros((1),dtype=np.float32)
        self.products["source_t"]  = data["source"].reshape( (batchsize,1,width,height) ).transpose(index)  # source image ADC
        self.products["target1_t"] = data["target1"].reshape( (batchsize,1,width,height) ).transpose(index) # target image ADC
        self.products["target2_t"] = data["target2"].reshape( (batchsize,1,width,height)).transpose(index)  # target image ADC
        self.products["flow1_t"]   = data["pixflow1"].reshape( (batchsize,1,width,height)).transpose(index) # flow from source to target
        self.products["flow2_t"]   = data["pixflow2"].reshape( (batchsize,1,width,height)).transpose(index) # flow from source to target
        self.products["fvisi1_t"]  = data["pixvisi1"].reshape( (batchsize,1,width,height)).transpose(index) # vis at source (float)
        self.products["fvisi2_t"]  = data["pixvisi2"].reshape( (batchsize,1,width,height)).transpose(index) # vis at source (float)
        
        # apply threshold to source ADC values. returns a byte mask
        self.products["fvisi1_t"]  = self.products["fvisi1_t"].clip(0.0,1.0)
        self.products["fvisi2_t"]  = self.products["fvisi2_t"].clip(0.0,1.0)
        
        # make integer visi
        self.products["visi1_t"]   = self.products["fvisi1_t"].reshape( (batchsize,width,height) ).astype( np.int )
        self.products["visi2_t"]   = self.products["fvisi2_t"].reshape( (batchsize,width,height) ).astype( np.int )
        
        # image column origins
        self.products["source_x"]  = data["meta"].reshape((batchsize,3,1,4))[:,0,:,0].reshape((batchsize))
        self.products["target1_x"] = data["meta"].reshape((batchsize,3,1,4))[:,1,:,0].reshape((batchsize))
        self.products["target2_x"] = data["meta"].reshape((batchsize,3,1,4))[:,2,:,0].reshape((batchsize))

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


    
