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

class LArCV2ServerWorker2( WorkerService ):
    """ This worker simply receives data and replies with dummy string. prints shape of array. """

    def __init__( self,inputfile,identity,ipaddress,load_func,port=0,batchsize=None,worker_verbosity=0):
        super( LArCV2ServerWorker2, self ).__init__(identity,ipaddress)
        self.inputfile = inputfile
        self.io = larcv.IOManager(larcv.IOManager.kREAD)
        self.io.add_in_file(self.inputfile)
        self.io.initialize()
        self.nentries = self.io.get_n_entries()
        self.batchsize = batchsize
        self.products = {}
        self.compression_level = 4
        self.print_msg_size = False
        self.num_reads = 0
        self.load_func = load_func
        print "LArCV2ServerWorker2[{}] is loaded.".format(self._identity)
        
    def process_message(self, frames ):
        """ just a request. nothing to parse
        """
        return True

    def fetch_data(self):
        """ load up the next data set. we've already sent out the message. so here we try to hide latency while gpu running. """

        tstart = time.time()
        # get data
        indices = np.random.randint(0,high=self.nentries,size=(self.batchsize))
        batch = []
        keylist = None
        for ib,idx in enumerate(indices):
            self.io.read_entry(idx)
            batch.append( self.load_func( self.io ) )
            if keylist is None:
                keylist = batch[-1].keys()

        self.products = {}
        for k in keylist:
            nchannels = 1
            datashape = batch[0][k].shape
            if len(datashape)==3:
                nchannels = datashape[0]
            self.products[k] = np.zeros( (self.batchsize,nchannels,datashape[-2],datashape[-1]), dtype=batch[0][k].dtype )
        for ib,batchdata in enumerate(batch):
            for k in keylist:
                if self.products[k].shape[1]>1:
                    self.products[k][ib,:,:,:] = batchdata[k][:,:,:]
                else:
                    self.products[k][ib,0,:,:] = batchdata[k][:,:]
            
        self.num_reads += 1
        tload = time.time()-tstart
        print "LArCV2ServerWorker2[{}] fetched data. time={} secs. nreads={}".format(self._identity,tload,self.num_reads)
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
            print "LArCV2ServerWorker2[{}]: size of array portion={} MB (uncompressed {} MB)".format(self._identity,totcompsize/1.0e6,totmsgsize/1.0e6)
        print "LArCV2ServerWorker2[{}]: generate msg in {} secs".format(self._identity,time.time()-tstart)
        return reply
        
    def start_dataloader(self,batchsize):
        print "LArCV2ServerWorker2[{}] starting loader w/ batchsize={}".format(self._identity,self.batchsize)
        self.batchsize = batchsize
        self.larcvloader.start(self.batchsize)
        print "LArCV2ServerWorker2[{}] dataloader ready, loading first product set".format(self._identity,self.batchsize)
        while not self.larcvloader.io._proc.manager_started():
            time.sleep(1.0)            
            print "LArCV2ServerWorker2[{}] waiting for larcv_threadio".format(self._identity)
        #self.post_reply() # get first batch
        print "LArCV2ServerWorker2[{}] manager started. syncing with client".format(self._identity)
            

if __name__ == "__main__":
    pass


    
