#!/usr/bin/env python
import os,sys

# more involved, pyqtgraph visualization

import argparse
# ARGUMENT DEFINTIONS

argparser = argparse.ArgumentParser(description="pyqtgraph visualization for DL cosmic tagging")
argparser.add_argument("-i", "--input",    required=True,  type=str, help="location of input larlite file with larflow3dhit tree")
#argparser.add_argument("-mc","--mctruth",  default=None,   type=str, help="location of input larlite file with mctrack and mcshower objects")
argparser.add_argument("-e", "--entry",    required=True,  type=int, help="entry number")
#argparser.add_argument("-c", "--color",    required=True,  type=str, help="colorscheme. options: [ssnet,quality,flowdir,infill,3ddist,hastruth,dwall]")
args = argparser.parse_args(sys.argv[1:])

# Setup pyqtgraph/nump
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from vis.detectordisplay import DetectorDisplay

import ROOT as rt
from larcv import larcv
from larlite import larlite
from ROOT import larutil

# create app and 3D viewer widget that shows MicroBooNE Mesh scene
app = QtGui.QApplication([])
w = DetectorDisplay()
w.show() # bring main 3D screen
w.solidswidget.show() # checkbox for parts of detector
w.setWindowTitle('LArFlow Visualization')

# default, turn off cryostat
w.changeComponentState("OuterCryostatPC",False)
w.changeComponentState("InnerCryostatPC",False)
w.changeComponentState("TPCFrame",False)
w.changeComponentState("CathodePlate",False)
w.changeComponentState("TPCPlaneVert",False)
w.changeComponentState("TPCPlane",False)
w.changeComponentState("FieldCageTubeZ",False)
w.changeComponentState("FieldCageTubeY",False)
w.changeComponentState("TPCActive",False)

# import data
inputfile = args.input
io = larlite.storage_manager(larlite.storage_manager.kREAD)
io.add_in_filename( inputfile )
io.open()

# color scheme
#schemes = ["colorbyssnet","colorbyquality","colorbyflowdir","colorbyinfill","colorby3ddist","colorbyhastruth","colorbydwall","colorbyrecovstruth"]
#shortschemes = ["ssnet","quality","flowdir","infill","3ddist","hastruth","dwall","recovstruth"]

# colorscheme = args.color
# if colorscheme in shortschemes:
#     colorscheme = "colorby"+colorscheme
# if colorscheme not in schemes:
#     raise ValueError("Invalid color scheme. Choices: {}".format(schemes))

# get larflow clusters
io.go_to(args.entry)
ev_larflow = io.get_data(larlite.data.kLArFlowCluster, "flowtruthclusters" )
nclusters = ev_larflow.size()
print "Number of larflow clusters: ",nclusters

#if colorscheme=="colorbyrecovstruth":
#    # we also need to count points with truth
#    for ihit in xrange(nhits):
#        hit = ev_larflow.at(ihit)
#        if hit.truthflag>0:
#            ntruth += 1


class clustercolorgen:

    def __init__(self):
        self.generated = 0
        self.current = 1.0
        self.current_gen = 0

        self.template  = np.array( ( (0,0,1),(0,1,0),(1,0,0),(0,1,1),(1,0,1),(1,1,0) ) )

    def gencolor(self):
        
        color = (self.template*self.current)[self.current_gen]
        color += (1.0-self.current)
        self.current_gen += 1
        if self.current_gen==6:
            # make a new calculation for the number to multiply by
            self.current_gen = 0
            #self.current /= 2.0           
        #if self.current<(1.0/4.0):
        #    # restart the loop
        #    self.current = 1.0
        #    self.current_gen = 0
        return color

colorgen = clustercolorgen()

itruthhit = 0
clusterplotitems = []
for icluster in xrange(nclusters):

    cluster = ev_larflow.at(icluster)
    
    # we get positions, and also set color depending on type
    # note ntruth=0 unless colorscheme is recovstruth
    nhits = cluster.size()
    pos_np = np.zeros( (nhits,3) )
    colors = np.zeros( (nhits,4) )

    # generate a color for this cluster
    clustercol = colorgen.gencolor()

    if icluster+1==nclusters:
        # last one, reserve white
        clustercol = np.ones( (3,) )

    print clustercol

    for ihit in xrange(nhits):
    
        hit = cluster.at(ihit)
        pos_np[ihit,0] = hit.at(0)-130.0
        pos_np[ihit,1] = hit.at(1)
        pos_np[ihit,2] = hit.at(2)-500.0
        colors[ihit,0] = clustercol[0]
        colors[ihit,1] = clustercol[1]
        colors[ihit,2] = clustercol[2]
        colors[ihit,3] = 0.5
        
        
    hitplot = gl.GLScatterPlotItem(pos=pos_np, color=colors, size=2.0, pxMode=False)
    clusterplotitems.append( hitplot )

        
# make the plot
for n,hitplot in enumerate(clusterplotitems):
    w.addVisItem( "cluster%d"%(n), hitplot )
    
w.plotData()

# start the app. close windows to end program
sys.exit(app.exec_())
