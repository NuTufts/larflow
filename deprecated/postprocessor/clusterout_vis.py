#!/usr/bin/env python
import os,sys
from math import sqrt

# more involved, pyqtgraph visualization

import argparse
# ARGUMENT DEFINTIONS

argparser = argparse.ArgumentParser(description="pyqtgraph visualization for DL cosmic tagging")
argparser.add_argument("-i", "--input",    required=True,  type=str, help="location of input larlite file with larflow3dhit tree")
argparser.add_argument("-t", "--treename", required=True,  type=str, help="tree (or producer name) to get larflowclusters from")
argparser.add_argument("-e", "--entry",    required=True,  type=int, help="entry number")
argparser.add_argument("-p", "--pca",      action='store_true',      help="plot pca")
argparser.add_argument("-m", "--mode",     default="",     type=str, help="plotting mode")
argparser.add_argument("-mc","--mctruth",  default=None,   type=str, help="location of input larlite file with mctrack and mcshower objects")
argparser.add_argument("-s", "--step",     action='store_true',      help="step through individual clusters")
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
#w = DetectorDisplay(background=(100,100,100,0.1) )
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

# modes
modes = ["core","core-only","spheres"]

# get larflow clusters
io.go_to(args.entry)
ev_larflow = io.get_data(larlite.data.kLArFlowCluster, args.treename )
nclusters = ev_larflow.size()
print "Number of larflow clusters: ",nclusters

ev_pca = None
if args.pca:
    ev_pca = io.get_data(larlite.data.kPCAxis, args.treename)

## ================================================================================
## truth plotting
cm_per_tick = larutil.LArProperties.GetME().DriftVelocity()*(larutil.DetectorProperties.GetME().SamplingRate()*1.0e-3)
print "driftv: ",larutil.LArProperties.GetME().DriftVelocity()
print "sampling rate:",larutil.DetectorProperties.GetME().SamplingRate()
print "cm_per_tick=",cm_per_tick

def extract_trackpts( mctrack, sce ):
    # convert mctrack points to image pixels
    steps_np  = np.zeros( (mctrack.size(),3) )
    colors_np = np.zeros( (mctrack.size(),4) )
    colors_np[:,3] = 1
    for istep in xrange(mctrack.size()):
        step = mctrack.at(istep)
        t = step.T()
        
        tick = larutil.TimeService.GetME().TPCG4Time2Tick(t) + step.X()/(cm_per_tick)
        #print "t=",t," -> tick(t)=",larutil.TimeService.GetME().TPCG4Time2Tick(t),"  tick(t)+pos/cmpertick=",tick
        #x = (tick + larutil.TimeService.GetME().TriggerOffsetTPC()/(larutil.DetectorProperties.GetME().SamplingRate()*1.0e-3))*cm_per_tick
        #x = tick
        x  = (tick - 3200)*cm_per_tick
        
        steps_np[istep,:] = (x,step.Y(),step.Z()-500)
        if tick<2400-2400 or tick>8448-2400:
            colors_np[istep,2] = 1
        else:
            colors_np[istep,0] = 1
    if mctrack.Origin()==2:
        mcplot = gl.GLLinePlotItem(pos=steps_np,color=colors_np,width=1.0) # cosmic
    else:
        mcplot = gl.GLLinePlotItem(pos=steps_np,color=(1,0,1,1),width=1.0) # neutrino
    #sys.exit(-1)
    return mcplot

mcplots = []
if args.mctruth is not None:
    sce = larutil.SpaceChargeMicroBooNE()
    iomc = larlite.storage_manager( larlite.storage_manager.kREAD )
    iomc.add_in_filename( args.mctruth )
    iomc.open()
    iomc.go_to( args.entry )
    evmctrack = iomc.get_data( larlite.data.kMCTrack, "mcreco" )
    nmctracks = evmctrack.size()
    for itrk in xrange(nmctracks):
        mctrack = evmctrack.at(itrk)
        mcplot = extract_trackpts( mctrack, sce )
        #w.addVisItem( "mctrackidx%d"%(itrk), mcplot )
        mcplots.append( mcplot )
    iomc.close()
## ================================================================================


## Example 3:
## sphere
md = gl.MeshData.sphere(rows=10, cols=20, radius=3)
#colors = np.random.random(size=(md.faceCount(), 4))
#colors[:,3] = 0.3
#colors[100:] = 0.0
colors = np.ones((md.faceCount(), 4), dtype=float)
colors[::2,0] = 0
colors[:,1] = np.linspace(0, 1, colors.shape[0])
md.setFaceColors(colors)
#m3 = gl.GLMeshItem(meshdata=md, smooth=False)#, shader='balloon')
#m3.translate(5, -5, 0)
#w.addItem(m3)

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
    print "[cluster %d] number of hits %d"%(icluster,nhits)

    if args.mode == "sphere":
        for ihit in xrange(nhits):
            hit = cluster.at(ihit)
            s = gl.GLMeshItem(meshdata=md,smooth=False)
            s.translate( hit.at(0)-130.0, hit.at(1), hit.at(2)-500.0 )
            clusterplotitems.append(s)
    else:
        
        pos_np = np.zeros( (nhits,3) )
        colors = np.zeros( (nhits,4) )

        # generate a color for this cluster
        clustercol = colorgen.gencolor()

        if icluster+1==nclusters:
            # last one, reserve white
            clustercol = np.ones( (3,) )

        for ihit in xrange(nhits):
            hit = cluster.at(ihit)
            pos_np[ihit,0] = hit.at(0)-130.0
            pos_np[ihit,1] = hit.at(1)
            pos_np[ihit,2] = hit.at(2)-500.0
            colors[ihit,0] = clustercol[0]
            colors[ihit,1] = clustercol[1]
            colors[ihit,2] = clustercol[2]
            colors[ihit,3] = 1.0
        

        # cluster
        hitplot = gl.GLScatterPlotItem(pos=pos_np, color=colors, size=3.0, pxMode=False)
        clusterplotitems.append( hitplot )

    # pca
    if args.pca and icluster+1<nclusters:
        # pca of all clusters, except last one, which is unassigned hits
        
        pcapoints = np.zeros( (3,3) ) # start, center, end of main eigenvector
        pcacolor  = np.ones( (3,4) )
        pcaxis  = ev_pca.at(icluster)
        eigval  = pcaxis.getEigenValues()
        #print eigval[0],eigval[1],eigval[2]
        meanpos = pcaxis.getAvePosition()
        eigvec  = pcaxis.getEigenVectors()
        eiglen  = sqrt(eigval[0])

        for j in xrange(3):
            pcapoints[0,j] = meanpos[j] - eigvec[j][0]*2*eiglen
            pcapoints[1,j] = meanpos[j]
            pcapoints[2,j] = meanpos[j] + eigvec[j][0]*2*eiglen
        for i in xrange(3):
            pcapoints[i,0] += -130.0
            pcapoints[i,2] += -500.0
            #for j in xrange(3):
            #    pcacolor[i,j] = clustercol[j]
        
        pcaplot = gl.GLLinePlotItem(pos=pcapoints,color=pcacolor,width=1.0)
        clusterplotitems.append( pcaplot )

core_plots = []
if args.mode in ["core","core-only"]:
    print "PLOTTING CORE CLUSTERS"
    if args.mode=="core-only":
        clusterplotitems = []
    
    # plot core points as well
    ev_core = io.get_data(larlite.data.kLArFlowCluster, "flowtruthcore" )
    ev_corepca = io.get_data(larlite.data.kPCAxis, "flowtruthcore" )    
    nclusters = ev_core.size()
    print "Number of core clusters: ",nclusters

    for iclust in xrange(nclusters):
        cluster = ev_core.at(iclust)
        nhits = cluster.size()
        print "  corecluster[",iclust,"] nhits=",nhits
        
        pos_np = np.zeros( (nhits,3) )
        colors = np.ones( (nhits,4) ) # white

        for ihit in xrange(nhits):            
            hit = cluster.at(ihit)
            pos_np[ihit,0] = hit.at(0)-130.0
            pos_np[ihit,1] = hit.at(1)
            pos_np[ihit,2] = hit.at(2)-500.0
            colors[ihit,0:3] = 255.0

        # pca
        if args.pca:
            pcapoints = np.zeros( (3,3) ) # start, center, end of main eigenvector
            pcacolor  = np.zeros( (3,4) )
            pcacolor[:,0] = 255.0
            pcacolor[:,3] = 1.0
            pcaxis  = ev_corepca.at(iclust)
            eigval  = pcaxis.getEigenValues()
            tot = eigval[0]+eigval[1]+eigval[2]
            print "eigenvals[",iclust,"]: ",eigval[0]/tot,eigval[1]/tot,eigval[2]/tot
            #if eigval[1]/eigval[0]>0.001:
            #    continue
            
            meanpos = pcaxis.getAvePosition()
            eigvec  = pcaxis.getEigenVectors()
            eiglen = sqrt(eigval[0])
        
            for j in xrange(3):
                pcapoints[0,j] = meanpos[j] - eigvec[j][0]*2*eiglen
                pcapoints[1,j] = meanpos[j]
                pcapoints[2,j] = meanpos[j] + eigvec[j][0]*2*eiglen
            for i in xrange(3):
                pcapoints[i,0] += -130.0
                pcapoints[i,2] += -500.0
        
            pcaplot = gl.GLLinePlotItem(pos=pcapoints,color=pcacolor,width=1.0)
            clusterplotitems.append( pcaplot )
            
        # cluster
        hitplot = gl.GLScatterPlotItem(pos=pos_np, color=colors, size=2.0, pxMode=False)        
        clusterplotitems.append( hitplot )
    
# step through individual clusters
if args.step:
    for n,hitplot in enumerate(clusterplotitems):
        w.addVisItem( "cluster%d"%(n), hitplot )
        w.plotData()
        print "[enter] to go to next cluster"
        raw_input()
        w.clearVisItems()
        hitplot.__pylard_transformed = True # hack

# all clusters at once
w.clearVisItems()    
for n,hitplot in enumerate(clusterplotitems):
    w.addVisItem( "cluster%d"%(n), hitplot )
for itrk,mcplot in enumerate(mcplots):
    w.addVisItem( "mctrackidx%d"%(itrk), mcplot )    
w.plotData()



# start the app. close windows to end program
sys.exit(app.exec_())
