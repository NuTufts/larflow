import os,sys,time

from pullclient import BasePullClient
from collections import OrderedDict
from functools import reduce
import numpy as np
import zmq
import zlib

import msgpack
import msgpack_numpy as m
m.patch()


# Inherist from client.SSNetClient
# Base class handles network stuff
# This handles processing of LArCV1 events
# Really, what I should use is composition ...

class LArCV2Sink( BasePullClient ):

    def __init__(self,batchsize,numworkers,identity,socketaddress,port=0,timeout_secs=30,max_tries=3):
        super( LArCV2Sink, self ).__init__(identity,numworkers,socketaddress,port=port,timeout_secs=timeout_secs,max_tries=max_tries )

        self.batchsize = batchsize
        self.products = {}

    def process_reply(self,frames):
        ## need to decode the products
        if len(frames)<=1:
            return False
        self.products = {}
        self.products["feeder"] = frames[0]
        data = frames[1:]
        numproducts = len(data)/2
        
        for i in xrange(numproducts):
            name   = data[2*i+0]
            x_comp = data[2*i+1]
            x_enc = zlib.decompress(x_comp)
            arr = msgpack.unpackb(x_enc,object_hook=m.decode)
            self.products[name] = arr

        return self.products
            

            
            
            

                
                    
    
