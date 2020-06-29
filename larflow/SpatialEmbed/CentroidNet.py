import os,sys
import ROOT as rt
from ROOT import std
import numpy

from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow


iolcv = larcv.IOManager( larcv.IOManager.kREAD )
iolcv.add_in_file( "/home/taritree/larcvtruth-Run000001-SubRun000001.root" )
iolcv.initialize()
iolcv.set_verbosity(1)


ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( "/home/taritree/mcinfo-Run000001-SubRun000001.root" )
ioll.set_verbosity(1)
ioll.open()

nentries = iolcv.get_n_entries()


for ientry in xrange(nentries):

    iolcv.read_entry(ientry)
    ioll.go_to(ientry)

    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.set_adc_treename( "wiremc" )
    mcpg.buildgraph( iolcv, ioll )
    mcpg.printGraph()

    preptriplet = larflow.PrepMatchTriplets()
    prepembed = larflow.spatialembed.PrepMatchEmbed()

    prepembed.process( iolcv, ioll, preptriplet )

    print "============================================="
    print prepembed.get_id_list(0)
    print "============================================="


    tids_from_neutrino = mcpg.getNeutrinoPrimaryParticles()

    # for node in mcpg.getNeutrinoPrimaryParticles():
    #     print "Origin:", node.origin
    #     print "pid:", node.pid
    #     print "tid:", node.tid

    rows = []
    cols = []
    for plane in [0]:
        print "plane: ", plane
        for instance_node in tids_from_neutrino:
            print "tid: ", instance_node.tid, ", pid: ", instance_node.pid
            pixels = prepembed.get_instance_pixlist(plane, instance_node.tid)
            rows.append(map(lambda node: instance_node.row, pixels))
            cols.append(map(lambda anode: instance_node.col, pixels))          

    print rows
    print cols

    if ientry == 0:
        break

    # for p in xrange(3):
    #     print "plane: ",p
    #     id_v = prepembed.get_id_list(p)

    #     print " num instances: ",id_v.size()

    #     for iid in xrange(id_v.size()):
    #         aid = id_v[iid]
    #         pix_v = prepembed.get_instance_pixlist( p, aid )
            
    #         print "ancestor id: ",aid," npixels=",pix_v.size()
    #         raw_input()