import os,time
import zmq
from collections import OrderedDict

"""
BaseClient

Behavior with respect to the server
1) the client requires the IP address of the server upon startup and a unique identifier
2) a timeout is defined to get a reply from the server
3) if the request times out, the client reconnects to the server and tries again
4) the client tries for self._max_times
5) if cannot finish, the client stops and return bad status

"""

class BasePullClient(object):

    def __init__( self, identity, numworkers, socket_address="ipc:///tmp/feed", port=0, timeout_secs=30, max_tries=3, do_compress=True ):
        #  Prepare our context and sockets
        self._identity = u"PullClient-{}".format(identity).encode("ascii")
        self._socket_address = socket_address
        self._port = port
        self._max_tries = max_tries
        self._timeout_secs=timeout_secs
        self._expected_shape = (24,1,512,512)
        self._compress = do_compress
        self.load_socket()
        self.nmsgs = 0
        self._numworkers = numworkers

        self._ttracker = OrderedDict()
        self._ttracker["send/receive::triptime"] = 0.0

        print "BasePullClient[{}] configured. childclass should call wait_for_workers() when finished with constructor".format(self._identity)
        
    def load_socket(self):
        # MAKE DIRECTORY IF IPC SOCKET
        if "ipc://" in self._socket_address:
            tmpdir = self._socket_address[5:]
            print "Make directory for IPC socket"
            os.system("mkdir -p %s"%(tmpdir))
        self._context  = zmq.Context()
        # PULL SOCKET
        self._socket   = self._context.socket(zmq.PULL)
        self._socket.identity = self._identity
        self._socket.bind("%s/%d"%(self._socket_address,self._port))
        # READY SOCKET
        self._readysocket = self._context.socket(zmq.REP)
        self._readysocket.bind("%s/ready%d"%(self._socket_address,self._port))
        print "BasePullClient[{}] sockets created".format(self._identity)

    def wait_for_workers(self):
        """ recv blocks on code """
        print "BasePullClient[{}] Waiting for Workers ({})".format(self._identity,self._numworkers)
        nworkers_ready = 0
        while nworkers_ready<self._numworkers:
            msg = self._readysocket.recv()
            print "BasePullClient[{}] worker {} ready".format( self._identity, msg )
            self._readysocket.send("OK")
            nworkers_ready += 1
            print "BasePullClient[{}] replied OK (to{}). ready workers {} of {}".format( self._identity,msg,nworkers_ready,self._numworkers)
        print "BasePullClient[{}] All workers accounted for".format(self._identity)
        return
        
    def receive(self):
        #print "PullClient[{}] request data.".format(self._identity)
        retries_left = self._max_tries

        troundtrip = time.time()

        reply = self._socket.recv_multipart()
                
        # process reply
        #print "PullClient[{}] received multi-part reply (len={}).".format(self._identity,len(reply))
        troundtrip = time.time()-troundtrip
        self._ttracker["send/receive::triptime"] += troundtrip
        self.nmsgs += 1
        return self.process_reply( reply )


    def process_reply(self):
        raise NotImplementedError('Must be implemented by the subclass.')
    
    def print_time_tracker(self):

        
        print "=============  TimeTracker =============="
        print "number of messages: ",self.nmsgs
        for k,v in self._ttracker.items():
            print k,": ",v," secs total"
        print "========================================="
