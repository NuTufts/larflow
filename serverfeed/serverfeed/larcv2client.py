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

    def __init__(self, batchsize, identity, broker_ipaddress, port=5559, timeout_secs=30, max_tries=3 ):
        super( LArCV2Sink, self ).__init__( identity, broker_ipaddress, port=port, timeout_secs=timeout_secs, max_tries=max_tries )

        self.batchsize = batchsize
        self.products = {}

    def get_batch(self):
        return True

    def make_outgoing_message(self):
        msg = ["{}".format(self.batchsize)]
        return msg
    
    def process_reply(self,frames):
        ## need to decode the products
        self.products = {}
        numproducts = len(frames)/2
        
        for i in xrange(numproducts):
            name   = frames[2*i+0]
            x_comp = frames[2*i+1]
            x_enc = zlib.decompress(x_comp)
            arr = msgpack.unpackb(x_enc,object_hook=m.decode)
            self.products[name] = arr
            
    def process_events(self, start=None, end=None):

        tprocess = time.time()
        if start is None:
            start = 0
        if end is None:
            end = self.nentries

        for ientry in range(start,end):
            ok = self.send_receive()
            if ok is False:
                raise RuntimeError("Trouble processing event")
            self.print_time_tracker()

        tprocess = time.time()-tprocess
        print "Time to run CaffeLArCV1Client[{}]::process_events: %.2f secs (%.2f secs/event)".format(self._identity)%(tprocess,tprocess/(end-start))
        

            
            
            

                
                    
    
