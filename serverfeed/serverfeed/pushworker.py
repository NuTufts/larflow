import sys,time
import zmq
from random import randint
from zmq import ssh

from workermessages import PPP_READY, PPP_HEARTBEAT

class BasePushWorker(object):

    def __init__(self,identity,pullsocket_address, port=0, timeout_secs=30, heartbeat_interval_secs=2, num_missing_beats=3, ssh_thru_server=None, worker_verbosity=0):
        self._identity = u"PushWorker-{}".format(identity).encode("ascii")
        self._pullsocket_address = pullsocket_address
        self._port = port
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
        self.connect_to_pullsocket()
        self.event_queue = []

        # connect to sync socket
        self._sync_socket = self._context.socket(zmq.REQ)
        self._sync_socket.connect("%s/ready%d"%(self._pullsocket_address,self._port))
        
    def connect_to_pullsocket(self):
        """ create new socket. connect to server. send READY message """

        self._socket   = self._context.socket(zmq.PUSH)
        self._socket.setsockopt(zmq.IDENTITY, self._identity)
        
        if self._ssh_thru_server is None:
            # regular connection            
            #self._socket.connect("tcp://%s:%d"%(self._broker_ipaddress,self._broker_port))
            self._socket.connect("%s/%d"%(self._pullsocket_address,self._port))
            if self._worker_verbosity>=0:
                print "BaseWorker[{}] socket connected".format(self._identity)
        else:
            ssh.tunnel_connection(self._socket, "tcp://%s:%d"%(self._broker_ipaddress,self._broker_port), self._ssh_thru_server )
            if self._worker_verbosity>=0:            
                print "BaseWorker[{}] socket connected via ssh-tunnel".format(self._identity)

        
    def do_work(self):

        liveness = self._num_missing_beats
        interval = self._interval_init
        heartbeat_at = time.time() + self._heartbeat_interval

        while not self.isready():
            print "{} not ready".format(self._identity)
            time.sleep(0.01)
        print "{} Do Work".format(self._identity)

        nmsgs = 0
        while True:

            if self._worker_verbosity>1:
                print "BasePushWorker[{}]: Generate Data".format(self._identity)
            # calling child function
            self.post_reply() # load data

            if len(self.event_queue)<4:
                reply = self.generate_reply() # gen message
                self.event_queue.append(reply)
            # send back through the proxy
            #self._socket.send_multipart(reply,copy=False)
            if len(self.event_queue)>0:
                reply = self.event_queue.pop()
                self._socket.send_multipart(reply)
                nmsgs += 1

            if self._worker_verbosity>1:
                print "BasePushWorker[{}]: Queue={} sent={}".format(self._identity,len(self.event_queue),nmsgs)
            # flush std out
            sys.stdout.flush()
            #break
        # end of while loop

        return True

    def generate_reply(self):
        raise NotImplemented("Inherited classes must define this function")

    def post_reply(self):
        raise NotImplemented("Inherited classes must define this function")

    def isready(self):
        raise NotImplemented("Inherited classes must define this function")

    def sync(self):
        """  contact the sync socket, to say this worker is ready """
        print "BasePushWorker[{}] starting sync with client".format(self._identity)
        self._sync_socket.send(b'{}'.format(self._identity))
        msg = self._sync_socket.recv()
        print "BasePushWorker[{}] message from client received. Ready.".format(self._identity)
        return
