import os,sys,time

from serverfeed.larcv2socketfeed import LArCV2SocketFeeder


if __name__ == "__main__":

    batchsize = 4
    nworkers  = 3
    print "start feeders"
    feeder = LArCV2SocketFeeder(batchsize,"test","tester_flowloader_dualflow_train.cfg","ThreadProcessorTrain",nworkers)
    
    print "start receiving"
    nentries = 100
    tstart = time.time()
    for n in xrange(nentries):
        batch = feeder.get_batch_dict()
        print "entry[",n,"] from ",batch["feeder"],": ",batch.keys()
    tend = time.time()-tstart
    print "elapsed time, ",tend,"secs ",tend/float(nentries)," sec/batch"
    # batchsize 4 with 4 workers, about 200 ms for meitner
