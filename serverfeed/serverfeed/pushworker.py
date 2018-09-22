import sys,time
import zmq
from random import randint
from zmq import ssh

from workermessages import PPP_READY, PPP_HEARTBEAT

class BasePushWorker(object):

    def __init__(self,identity,broker_ipaddress, port=5560, timeout_secs=30, heartbeat_interval_secs=2, num_missing_beats=3, ssh_thru_server=None, worker_verbosity=0):
        self._identity = u"PushWorker-{}".format(identity).encode("ascii")
        self._broker_ipaddress = broker_ipaddress
        self._broker_port = port
        self._heartbeat_interval = heartbeat_interval_secs
        self._num_missing_beats  = num_missing_beats
        self._interval_init      = 1
        self._interval_max       = 32
        self._timeout_secs = timeout_secs
        self._worker_verbosity = worker_verbosity
        if ssh_thru_server is not None and type(ssh_thru_server) is not str:
            raise ValueError("ssh_thru_server should be a str with server address, e.g. user@server")
        self._ssh_thru_server = ssh_thru_server

        self._context = zmq.Context(1)
        self.connect_to_broker()

    def connect_to_broker(self):
        """ create new socket. connect to server. send READY message """

        self._socket   = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.IDENTITY, self._identity)
        
        if self._ssh_thru_server is None:
            # regular connection            
            #self._socket.connect("tcp://%s:%d"%(self._broker_ipaddress,self._broker_port))
            self._socket.connect("ipc:///tmp/feeds/0")
            if self._worker_verbosity>=0:
                print "BaseWorker[{}] socket connected".format(self._identity)
        else:
            ssh.tunnel_connection(self._socket, "tcp://%s:%d"%(self._broker_ipaddress,self._broker_port), self._ssh_thru_server )
            if self._worker_verbosity>=0:            
                print "BaseWorker[{}] socket connected via ssh-tunnel".format(self._identity)

        self._socket.send(PPP_READY)
        if self._worker_verbosity>=0:        
            print "BaseWorker[{}] sent PPP_READY".format(self._identity)        

    def do_work(self):

        liveness = self._num_missing_beats
        interval = self._interval_init
        heartbeat_at = time.time() + self._heartbeat_interval
        
        while True:

            if self._worker_verbosity>1:
                print "BasePushWorker[{}]: Generate Data".format(self._identity)
            # calling child function
            self.post_reply() # load data
            
            reply = self.generate_reply() # gen message
            # send back through the proxy
            self._socket.send_multipart(reply)
                
            # flush std out
            sys.stdout.flush()

        # end of while loop

        return True

    def process_message(self,frames):
        raise NotImplemented("Inherited classes must define this function")

    def generate_reply(self):
        raise NotImplemented("Inherited classes must define this function")

    def post_reply(self):
        raise NotImplemented("Inherited classes must define this function")
