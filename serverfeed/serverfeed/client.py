import time
import zmq
from collections import OrderedDict

"""
ServerClient

This class runs SSNet through a inference server.
It loads a LArCV file and for each event:
1) cuts the region of interest in each plane into 8 pieces, 23 images total
2) prepares a numpy array with shape (24,1,512,512) containing the 8 pieces (50 MB)
3) serializes the data using messagepack-numpy
4) compresses the serialized bytestring
5) sends a request to the SSNet server to run SSNet on the images
6) receives the result
7) embeds the pieces back into a whole image for each plane
8) saves the ssnet image

Behavior with respect to the server
1) the client requires the IP address of the server upon startup and a unique identifier
2) a timeout is defined to get a reply from the server
3) if the request times out, the client reconnects to the server and tries again
4) the client tries for self._max_times
5) if cannot finish, the client stops and return bad status

"""

class ServerClient(object):

    def __init__( self, identity, broker_ipaddress, timeout_secs=30, max_tries=3, do_compress=True, verbosity=0 ):
        #  Prepare our context and sockets
        self._identity = u"Client-{}".format(identity).encode("ascii")
        self._broker_ipaddress = broker_ipaddress
        self._max_tries = max_tries
        self._timeout_secs=timeout_secs
        self._expected_shape = (24,1,512,512)
        self._compress = do_compress
        self._verbosity = verbosity        
        self.load_socket()
        self.nmsgs = 0


        self._ttracker = OrderedDict()
        self._ttracker["send/receive::triptime"] = 0.0
        
    def load_socket(self):
        self._context  = zmq.Context()
        self._socket   = self._context.socket(zmq.REQ)        
        self._socket.identity = self._identity
        self._socket.connect("%s/%d"%(self._broker_ipaddress,0))
        self._poller = zmq.Poller()
        self._poller.register( self._socket, zmq.POLLIN )
        print "ServerClient[{}] socket connected to server".format(self._identity)

    def send_receive(self):

        batchok = self.get_batch()
        if batchok==False:
            return False
        elif batchok is None:
            self.process_reply( None )
            return True
        msg = self.make_outgoing_message()
        
        retries_left = self._max_tries

        troundtrip = time.time()

        while retries_left>0:
        
            # send request
            for part in msg[:-1]:
                self._socket.send(part, zmq.SNDMORE)
            self._socket.send(msg[-1])

            socks = dict(self._poller.poll(self._timeout_secs*1000))
            if socks.get(self._socket) == zmq.POLLIN:
                # got message back
                reply = self._socket.recv_multipart()
                if not reply:
                    break
                
                # process reply
                if self._verbosity>1:
                    print "ServerClient[{}] received reply.".format(self._identity)
                troundtrip = time.time()-troundtrip
                self._ttracker["send/receive::triptime"] += troundtrip
                self.nmsgs += 1
                self.process_reply( reply )
                return True
                
            else:
                # timed out
                print "ServerClient[{}] no reply from server. retrying ...".format(self._identity)
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.close()
                self._poller.unregister(self._socket)
                retries_left -= 1
                if retries_left > 0:
                    self.load_socket()
                

        # should not get here
        troundtrip = time.time()-troundtrip
        self._ttracker["send/receive::triptime"] += troundtrip

        print "ServerClient[{}] Server seems to be offline, abandoning".format(self._identity)
        return False

    def get_batch( self ):
        raise NotImplementedError('Must be implemented by the subclass.')

    def make_outgoing_message( self ):
        raise NotImplementedError('Must be implemented by the subclass.')
    
    def print_time_tracker(self):

        
        print "=============  TimeTracker =============="
        print "number of messages: ",self.nmsgs
        for k,v in self._ttracker.items():
            print k,": ",v," secs total"
        print "========================================="
