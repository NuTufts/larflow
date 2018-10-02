import os,sys,time
from worker import WorkerService
import numpy as np
import zlib
import zmq

from larcv import larcv

import msgpack
import msgpack_numpy as m
m.patch()

os.environ["GLOG_minloglevel"] = "1"

from workermessages import decode_larcv1_metamsg
from larcvdataset import LArCVDataset

class LArCV2ThreadIOWorker( WorkerService ):
    """ This worker simply receives data and replies with dummy string. prints shape of array. """

    def __init__( self,configfile,fillername,identity,ipaddress,port=0,batchsize=None,verbosity=0):
        super( LArCV2ThreadIOWorker, self ).__init__(identity,ipaddress)
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
        print "LArCV2ThreadIOWorker[{}] is loaded.".format(self._identity)
        
    def process_message(self, frames ):
        """ just a request. nothing to parse
        """
        return True

    def fetch_data(self):
        """ load up the next data set. we've already sent out the message. so here we try to hide latency while gpu running. """
        
        # get data
        tstart = time.time()
        while self.larcvloader.io._proc.thread_running():
            #print "finish load"
            time.sleep(0.001)
        self.products = self.larcvloader[0]
        while self.larcvloader.io._proc.thread_running():
            #print "finish load"
            time.sleep(0.001)        
        #print "[",self.num_reads,":{}] ".format(self._identity),self.products.keys()
        self.num_reads += 1        
        print "LArCV2ThreadIOWorker[{}] fetched data. time={} secs. nreads={}".format(self._identity,time.time()-tstart,self.num_reads)
        return
    
    def generate_reply(self):
        """
        our job is to return our data set, then load another
        """
        self.fetch_data()
        
        reply = [self._identity]
        totmsgsize = 0.0
        totcompsize = 0.0
        tstart = time.time()
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
            print "LArCV2ThreadIOWorker[{}]: size of array portion={} MB (uncompressed {} MB)".format(self._identity,totcompsize/1.0e6,totmsgsize/1.0e6)
        print "LArCV2ThreadIOWorker[{}]: generate msg in {} secs".format(self._identity,time.time()-tstart)
        return reply
        
    def start_dataloader(self,batchsize):
        print "LArCV2ThreadIOWorker[{}] starting loader w/ batchsize={}".format(self._identity,self.batchsize)
        self.batchsize = batchsize
        self.larcvloader.start(self.batchsize)
        print "LArCV2ThreadIOWorker[{}] dataloader ready, loading first product set".format(self._identity,self.batchsize)
        while not self.larcvloader.io._proc.manager_started():
            time.sleep(1.0)            
            print "LArCV2ThreadIOWorker[{}] waiting for larcv_threadio".format(self._identity)
        #self.post_reply() # get first batch
        print "LArCV2ThreadIOWorker[{}] manager started. syncing with client".format(self._identity)
            

if __name__ == "__main__":
    pass


    
