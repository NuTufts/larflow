#!/usr/bin/env python
import os,sys
import argparse

import six
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import ROOT as rt
from larcv import larcv
from larlite import larlite
from ROOT import larutil

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from matplotlib import colors


def dataIO(inputfile,outputfile):
    # import data
    io = larlite.storage_manager()
    rw = io.kREAD
    if len(outputfile) !=0:
        rw = io.kBOTH
        io.set_out_filename( outputfile )
    io.set_io_mode(rw)
    io.add_in_filename( inputfile )
    #io.set_in_rootdir("")
    io.open()
    return io

def read_event(io,event=0,product="",producer=""):    
    io.go_to(event)
    ev = None
    myString  ='%s.get_data(larlite.data.%s,"%s")'%("io",product,producer)
    loc = {"io":io}
    ev = eval(myString,globals(),loc)
    #print "ev var: ",ev
    #exec("print ev")
    return ev

def save_event():
    print "saving"

def fill_data(ev):
    nhits = ev.size()
    pos_np = np.zeros( (nhits,3) )
    id_np  = np.zeros( (nhits) )
    for ihit in xrange(nhits):
        hit = ev.at(ihit)
        pos_np[ihit,0] = hit.at(0)
        pos_np[ihit,1] = hit.at(1)
        pos_np[ihit,2] = hit.at(2)
        id_np[ihit]    = hit.trackid
    return pos_np,id_np

def set_params(pars):
    default_base = {'quantile': .3,
                    'eps': .18,
                    'damping': .9,
                    'preference': -200,
                    'n_neighbors': 10,
                    'n_clusters': 20}
    params = default_base.copy()
    params.update(pars)
    return params
    
def call_algo(name,params):     
    algo=None
    if name=="dbscan" or name=="DBSCAN":
        algo = cluster.DBSCAN(eps=params['eps'],min_samples=params['n_neighbors'])
    elif name=="spectral":
        algo = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack',affinity="nearest_neighbors")
    elif name=="birch" or name=="Birch":
        algo = cluster.Birch(n_clusters=params['n_clusters'])
    elif name=="gmm" or name=="GMM":
        algo = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
    else:
        print "unknown algo; exit"
        exit(0)
    return algo

def main():

    # ARGUMENT DEFINTIONS
    argparser = argparse.ArgumentParser(description="scikit learn clustering algo application")
    argparser.add_argument("-i", "--input",    required=True,  type=str, help="location of input larlite file with larflow3dhit tree")
    argparser.add_argument("-o","--output",    default=None,   type=str, help="location of output file; if none will not save")
    argparser.add_argument("-a", "--algo",     required=True,  type=str, help="algo name(s) separated by comma")
    argparser.add_argument("-p","--plot",      default=False,  type=bool, help="plot event? default False ")

    args = argparser.parse_args(sys.argv[1:])

    # IO
    inputfile = args.input
    outputfile = ""
    if args.output is not None:
        outputfile = args.output
    io = dataIO(inputfile,outputfile)

    # DEFINE CLUSTERING ALGOS
    algonames= [x.strip() for x in args.algo.split(',')]
    alg_list = []

    params=set_params({'eps': .3})
    for name in algonames:
        algo=call_algo(name,params)
        pair = (name,algo)
        alg_list.append(pair)

    # EVENT LOOP
    nevents = 1
    for i in xrange(nevents):
        ev_larflow = read_event(io,i,"kLArFlow3DHit","flowhits")
        print "event: ",i, " 3D hits: ",ev_larflow.size()
        # fill the data and truth
        X,y = fill_data(ev_larflow)
        # normalize dataset for easier parameter selection
        X_norm = StandardScaler().fit_transform(X)
        # cluster
        fignum=1
        for (name,algo) in alg_list:
            algo.fit(X_norm)
            if hasattr(algo, 'labels_'):
                y_pred = algo.labels_.astype(np.int)
            else:
                y_pred = algo.predict(X_norm)

            if args.plot:
                plot_event(fignum,X,y_pred,name)
            if args.output is not None:
                save_event() # need ev_cluster per algo
        #raw_input()
        io.set_id( io.run_id(), io.subrun_id(), io.event_id() );
        io.next_event()
        
    #finalize
    io.close()
    exit(0)

def plot_event(fignum,X,y,algoname):    
    # get colors
    colors_ = list(six.iteritems(colors.cnames))
    for i, (n,h) in enumerate(colors_):
            colors_[i] = h
    colors_ = np.array(colors_)        
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                              s=10, c=colors_[y], depthshade=False, edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(algoname)
    ax.dist = 12
    fignum = fignum + 1    
    fig.show()
                                            
if __name__ == '__main__':
    main()

    
