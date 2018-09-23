import os,sys,time
from multiprocessing import Process

from larcv2client import LArCV2Sink
from larcv2feeder import LArCV2Feeder

def __start_larcv2feeder_workers__(identity,configfile,fillername,address,port,batchsize):
    worker = LArCV2Feeder(configfile,fillername,identity,address,port=port,batchsize=batchsize)
    worker.do_work()
        
class LArCV2SocketFeeder:

    def __init__(self,batchsize,identity,configurationfile,larcvfillername,nworkers,port=0):

        address = "ipc:///tmp/feed"

        # create the client
        self.client = LArCV2Sink(batchsize,identity,address,port=port)

        # create the workers
        self.cfgfile = configurationfile
        self.pworkers = [ Process(target=__start_larcv2feeder_workers__,
                                  args=("{}-{}".format(identity,n),self.cfgfile,larcvfillername,address,port,batchsize))
                          for n in xrange(nworkers) ]
        for pworker in self.pworkers:
            pworker.daemon = True

    def start(self):
        print "LArCV2SocketFeeder: starting workers"
        for pworker in self.pworkers:
            # offset the start
            time.sleep(0.1)
            pworker.start()

    def get_batch_dict(self):
        return self.client.receive()
