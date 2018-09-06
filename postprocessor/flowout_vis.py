import os,sys

# more involved, pyqtgraph visualization

# Setup pyqtgraph/nump
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from vis.detectordisplay import DetectorDisplay

import ROOT as rt
from larcv import larcv
from larlite import larlite

colorscheme = "colorbyssnet"

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
inputfile = sys.argv[1]
io = larlite.storage_manager(larlite.storage_manager.kREAD)
io.add_in_filename( inputfile )
io.open()

# get larflow hits
io.go_to(0)
ev_larflow = io.get_data(larlite.data.kLArFlow3DHit, "flowhits" )
nhits = ev_larflow.size()

print "Number of larflow hits: ",nhits

# we get positions, and also set color depending on type
pos_np = np.zeros( (nhits,3) )
colors = np.zeros( (nhits,4) ) # (r,g,b,alpha)
#hitplot = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.3), size=0.1, pxMode=False)

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
        

hitplot = gl.GLScatterPlotItem(pos=pos_np, color=colors, size=2.0, pxMode=False)
w.addVisItem( "flowhits", hitplot )
w.plotData()

# start the app. close windows to end program
sys.exit(app.exec_())
