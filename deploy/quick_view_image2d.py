import os,sys
import ROOT as rt
from larcv import larcv

rt.gStyle.SetOptStat(0)

rfile = sys.argv[1]
tree  = sys.argv[2]
entry = int(sys.argv[3])
index = int(sys.argv[4])

io = larcv.IOManager( larcv.IOManager.kREAD )
io.add_in_file( rfile )
io.initialize()

io.read_entry( entry )

evimg2d = io.get_data( "image2d", tree )

img = evimg2d.as_vector().at(index)

c = rt.TCanvas("c","c",1000,400)
c.Draw()

himg = larcv.as_th2d( img, "tree_%d_%d"%(entry,index))
himg.Draw("COLZ")
c.Update()

print "Printed"
raw_input()
