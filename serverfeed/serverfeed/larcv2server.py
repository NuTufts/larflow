import os,sys,time
from multiprocessing import Process

from server import Server
from larcv2serverworker import LArCV2ServerWorker
from larcv2serverworker2 import LArCV2ServerWorker2
from larcv2serverclient import LArCV2ServerClient

def __start_larcv2_server__(ipaddress,verbose):
    print "starting Server w/ ipaddress {}".format(ipaddress)
    server = Server(ipaddress,server_verbosity=verbose)
    server.start()

def __start_workers__(identity,cfg,fillername,address,port,batchsize):
    worker = LArCV2ServerWorker(cfg,fillername,identity,address,batchsize=batchsize)
    worker.do_work()

def __start_workers_v2__(identity,inputfile,loadfunc,address,port,batchsize):
    worker = LArCV2ServerWorker2(inputfile,identity,address,loadfunc,batchsize=batchsize)
    worker.do_work()
    
class LArCV2Server:

    def __init__(self,batchsize,identity,configurationfile,larcvfillername,nworkers,port=0,server_verbosity=0,load_func=None,inputfile=None):

        address = "ipc:///tmp/feeds" # client front end

        # start the server
        self.pserver = Process(target=__start_larcv2_server__,args=("ipc:///tmp/feeds",server_verbosity,))
        self.pserver.daemon = True
        self.pserver.start()

        # create the workers
        if load_func is None:
            self.cfgfile = configurationfile
            self.pworkers = [ Process(target=__start_workers__,
                                      args=("{}-{}".format(identity,n),self.cfgfile,larcvfillername,address,port,batchsize))
                              for n in xrange(nworkers) ]
        else:
            self.pworkers = [ Process(target=__start_workers_v2__,
                                      args=("{}-{}".format(identity,n),inputfile,load_func,address,port,batchsize))
                              for n in xrange(nworkers) ]
        for pworker in self.pworkers:
            pworker.daemon = True
            pworker.start()

        # client
        self.client = LArCV2ServerClient(identity,address)

        print "LArCV2Server. Workers initialized. Client synced with workers. Ready to feed data."


    def get_batch_dict(self):
        print "recieve one message"
        self.client.send_receive()
        return self.client.products

    def __len__(self):
        return len(self.client.larcvloader)


if __name__=="__main__":
    pass
