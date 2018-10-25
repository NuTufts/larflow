#!/usr/bin/env python
import os,sys

# more involved, pyqtgraph visualization

import argparse
# ARGUMENT DEFINTIONS

argparser = argparse.ArgumentParser(description="pyqtgraph visualization for DL cosmic tagging")
argparser.add_argument("-i", "--input",    required=True,  type=str, help="location of input larlite file with larflow3dhit tree")
argparser.add_argument("-mc","--mctruth",  default=None,   type=str, help="location of input larlite file with mctrack and mcshower objects")
argparser.add_argument("-e", "--entry",    required=True,  type=int, help="entry number")
argparser.add_argument("-c", "--color",    default="default", type=str, help="colorscheme. options: [ssnet,quality,flowdir,infill,3ddist,hastruth,dwall,recovstruth]")
argparser.add_argument("-l", "--light",    action='store_true',      help="use light background")
args = argparser.parse_args(sys.argv[1:])

# Setup pyqtgraph/nump
import pyqtgraph as pg
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
#w = DetectorDisplay(background=(150,150,150,0.1))
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
schemes = ["colorbydefault","colorbyssnet","colorbyquality","colorbyflowdir","colorbyinfill","colorby3ddist","colorbyhastruth","colorbydwall","colorbyrecovstruth"]
shortschemes = ["default","ssnet","quality","flowdir","infill","3ddist","hastruth","dwall","recovstruth"]

colorscheme = args.color
if colorscheme in shortschemes:
    colorscheme = "colorby"+colorscheme
if colorscheme not in schemes:
    raise ValueError("Invalid color scheme. Choices: {}".format(schemes))

# get larflow hits
io.go_to(args.entry)
ev_larflow = io.get_data(larlite.data.kLArFlow3DHit, "flowhits" )
nhits = ev_larflow.size()
ntruth = 0
print "Number of larflow hits: ",nhits

if colorscheme=="colorbyrecovstruth":
    # we also need to count points with truth
    for ihit in xrange(nhits):
        hit = ev_larflow.at(ihit)
        if hit.truthflag>0:
            ntruth += 1

# we get positions, and also set color depending on type
# note ntruth=0 unless colorscheme is recovstruth
pos_np = np.zeros( (nhits+ntruth,3) )
colors = np.zeros( (nhits+ntruth,4) )
#hitplot = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.1, pxMode=False)

itruthhit = 0
for ihit in xrange(nhits):
    
    hit = ev_larflow.at(ihit)
    pos_np[ihit,0] = hit.at(0)-130.0
    pos_np[ihit,1] = hit.at(1)
    pos_np[ihit,2] = hit.at(2)-500.0

    if colorscheme=="colorbyquality":
        if hit.matchquality==0:
            # on charge in cluster
            colors[ihit,:] = (1.0,1.0,1.0,1.0)
        elif hit.matchquality==1:
            # in cluster, moved to nearest charge
            colors[ihit,:] = (1.0,0.0,0.0,1.0)
        elif hit.matchquality==2:
            # outside cluster, moved to nearest cluster
            colors[ihit,:] = (0.0,1.0,0.0,1.0)
        else:
            # no match
            colors[ihit,:] = (0.0,0.0,0.0,1.0)
    elif colorscheme=="colorbyssnet":
        if hit.endpt_score>0.8:
            colors[ihit,:] = (1.0,0.0,0.0,1.0)
        elif hit.renormed_track_score>0.5:
            colors[ihit,:] = (1.0,1.0,1.0,1.0)
        elif hit.renormed_shower_score>0.5:
            colors[ihit,:] = (0.0,1.0,0.0,1.0)
        else:
            colors[ihit,:] = (0.0,0.0,1.0,1.0)

    elif colorscheme=="colorbyflowdir":
        if hit.flowdir==larlite.larflow3dhit.kY2U:
            colors[ihit,:] = (1.,1.,1.,1.)
        else:
            colors[ihit,:] = (1.,0.,0.,1.)
        
    elif colorscheme=="colorbyinfill":
        if hit.src_infill!=0 and hit.tar_infill[0]==0 and hit.tar_infill[1]==0:
            # src on infill blue
            colors[ihit,:] = (0.0,0.0,1.0,1.0)
        elif (hit.tar_infill[0]!=0 or hit.tar_infill[1]!=0) and hit.src_infill==0:
            # target on infill red
            colors[ihit,:] = (1.0,0.0,0.0,1.0)
        elif (hit.src_infill!=0 and (hit.tar_infill[0]!=0 or hit.tar_infill[1]!=0)):
            # both src and tar on infill green
            colors[ihit,:] = (0.0,1.0,0.0,1.0)
        elif hit.src_infill==0 and hit.tar_infill[0]==0 and hit.tar_infill[1]==0:
            # neither src nor tar on infill white
            colors[ihit,:] = (1.0,1.0,1.0,1.0)

    elif colorscheme=="colorby3ddist":
        if hit.consistency3d==0:
            colors[ihit,:] = (1.,1.,1.,1.)
        elif hit.consistency3d==1:
            colors[ihit,:] = (1.,0.,0.,1.)
        elif hit.consistency3d==2:
            colors[ihit,:] = (0.,1.,0.,1.)
        elif hit.consistency3d==3:
            colors[ihit,:] = (0.,0.,1.,1.)
        else:
            colors[ihit,:] = (1.,0.,1.,1.)

    elif colorscheme=="colorbyhastruth":
        if hit.truthflag==1:
            colors[ihit,:] = (1.,1.,1.,1.) # mctrack matched
        elif hit.truthflag==2:
            colors[ihit,:] = (1.,0.,0.,1.) # mctrack tails matched
        elif hit.truthflag==0:
            colors[ihit,:] = (0.,1.,0.,1.) # mctrack not matched
        else:
            colors[ihit,:] = (1.,1.,1.,1.) # kUnknown

    elif colorscheme=="colorbydwall":
        if hit.dWall>15.:
            colors[ihit,:] = (1.,1.,1.,1.)
        elif hit.dWall==-1:
            colors[ihit,:] = (0.,1.,0.,1.)
        else:
            colors[ihit,:] = (1.,0.,0.,1.)
        if hit.endpt_score>0.8:
            colors[ihit,:] = (0.,0.,1.,1.)

    elif colorscheme=="colorbyrecovstruth":
        colors[ihit,:] = (1.,1.,1.,0.5) # color for reco points is white        
        if hit.truthflag>0:
            # create a truth hit that is red
            pos_np[nhits+itruthhit,0] = hit.X_truth[0]-130
            pos_np[nhits+itruthhit,1] = hit.X_truth[1]
            pos_np[nhits+itruthhit,2] = hit.X_truth[2]-500.0
            if hit.truthflag==1:
                colors[nhits+itruthhit,:] = (1.,0.,0.,1.0) # core truth hit is red
            else:
                colors[nhits+itruthhit,:] = (0.,0.,1.,1.0) # edge truth hit is bleed                
            itruthhit+=1
    elif colorscheme=="colorbydefault":
        colors[ihit,:] = (1.,1.,1.,1.)
            

hitplot = gl.GLScatterPlotItem(pos=pos_np, color=colors, size=2.0, pxMode=False)

# truth plot
cm_per_tick = larutil.LArProperties.GetME().DriftVelocity()*(larutil.DetectorProperties.GetME().SamplingRate()*1.0e-3)
print "driftv: ",larutil.LArProperties.GetME().DriftVelocity()
print "sampling rate:",larutil.DetectorProperties.GetME().SamplingRate()
print "cm_per_tick=",cm_per_tick

def extract_trackpts( mctrack, sce ):
    # convert mctrack points to image pixels
    steps_np = np.zeros( (mctrack.size(),3) )
    for istep in xrange(mctrack.size()):
        step = mctrack.at(istep)
        t = step.T()
        
        tick = larutil.TimeService.GetME().TPCG4Time2Tick(t) + step.X()/(cm_per_tick)
        #print "t=",t," -> tick(t)=",larutil.TimeService.GetME().TPCG4Time2Tick(t),"  tick(t)+pos/cmpertick=",tick
        #x = (tick + larutil.TimeService.GetME().TriggerOffsetTPC()/(larutil.DetectorProperties.GetME().SamplingRate()*1.0e-3))*cm_per_tick
        #x = tick
        x  = (tick - 3200)*cm_per_tick
        
        steps_np[istep,:] = (x,step.Y(),step.Z()-500)
    if mctrack.Origin()==2:
        mcplot = gl.GLLinePlotItem(pos=steps_np,color=(1,0,0,1),width=1.0)
    else:
        mcplot = gl.GLLinePlotItem(pos=steps_np,color=(0,1,0,1),width=1.0)
    #sys.exit(-1)
    return mcplot

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
        w.addVisItem( "mctrackidx%d"%(itrk), mcplot )
    iomc.close()


#print sce
        
# make the plot
w.addVisItem( "flowhits", hitplot )
w.plotData()

# start the app. close windows to end program
sys.exit(app.exec_())
