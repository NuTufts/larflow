import os,sys,time

from serverfeed.server import Server
from serverfeed.larcv2feeder import LArCV2Feeder
from serverfeed.larcv2client import LArCV2Sink
import multiprocessing
import cProfile

##
## Contents of test file
##

"""
*         ../../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root  
  KEY: TTree    image2d_infillCropped_tree;1    infillCropped tree
  KEY: TTree    image2d_larflow_y2u_tree;1      larflow_y2u tree
  KEY: TTree    image2d_adc_tree;1      adc tree
  KEY: TTree    image2d_pixvisi_tree;1  pixvisi tree
  KEY: TTree    image2d_pixflow_tree;1  pixflow tree
  KEY: TTree    image2d_larflow_y2v_tree;1      larflow_y2v tree
  KEY: TTree    image2d_ssnetCropped_shower_tree;1      ssnetCropped_shower tree
  KEY: TTree    image2d_ssnetCropped_endpt_tree;1       ssnetCropped_endpt tree
  KEY: TTree    image2d_ssnetCropped_track_tree;1       ssnetCropped_track tree
"""


if __name__ == "__main__":

    NWORKERS = 5
    NENTRIES = 100

    # start client
    ident = 0
    client = LArCV2Sink(4,ident,"localhost")        
    
    def start_worker(ident):
        worker = LArCV2Feeder("tester_flowloader_dualflow_train.cfg","ThreadProcessorTrain",ident,"localhost")
        cProfile.runctx('worker.do_work()',globals(),locals(),"profworker_{}.prof".format(ident))

    def start_server():
        print "START SERVER"
        broker = Server("*",poller_timeout_secs=0.01)
        cProfile.runctx('broker.start()',globals(),locals(),"profserver.prof")

    #pserver = multiprocessing.Process(target=start_server)
    #pserver.daemon = True
    #pserver.start()

    workers = []
    for i in xrange(NWORKERS):
        process = multiprocessing.Process(target=start_worker,args=(i,))
        process.daemon = True
        process.start()
        workers.append(process)

    time.sleep(1)

    # start recieving the feeds
    startfeed = time.time()
    for i in xrange(NENTRIES):
        print "entry[",i,"]"
        ok = client.receive()
        print client.products.keys()
    endfeed = time.time()

    print "Test time: ",(endfeed-startfeed)/float(NENTRIES)," secs"

    print "shutting down..."
    for worker in workers:
        worker.terminate()

    #pserver.terminate()

    print "Test time: ",(endfeed-startfeed)/float(NENTRIES)," secs"

    

    
    
    

    
