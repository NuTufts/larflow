import time
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

    def __init__( self, identity, socket_address="ipc:///tmp/feed", port=0, timeout_secs=30, max_tries=3, do_compress=True ):
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

        self._ttracker = OrderedDict()
        self._ttracker["send/receive::triptime"] = 0.0
        
    def load_socket(self):
        self._context  = zmq.Context()
        self._socket   = self._context.socket(zmq.PULL)
        self._socket.identity = self._identity
        self._socket.bind("%s/%d"%(self._socket_address,self._port))
        print "BasePullClient[{}] socket connected to server".format(self._identity)

    def receive(self):
        
        retries_left = self._max_tries

        troundtrip = time.time()

        reply = self._socket.recv_multipart()
                
        # process reply
        print "PullClient[{}] received reply.".format(self._identity)
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
