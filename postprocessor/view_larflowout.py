import os,sys
import ROOT as rt
from larcv import larcv
from ublarcvapp import ublarcvapp
larcv.load_rootutil()

rt.gStyle.SetOptStat(0)

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file( "out_dense_ublarflow_test.root" )
#io.add_in_file( "flowtest.root" )
io.initialize()

io.read_entry(0)

#cfg = larcv.CreatePSetFromFile( "ubsplit.cfg","UBSplitDetector")
#splitalgo = ublarcvapp.UBSplitDetector()
#splitalgo.configure(cfg)
#splitalgo.initialize()

ev_larflow_y2u = io.get_data(larcv.kProductImage2D,"larflow_y2u")
y2u_v = ev_larflow_y2u.as_vector()
ncrops = y2u_v.size()
print "num crops: ",ncrops

c = rt.TCanvas("c","c",800,600)
c.Draw()

for iimg in xrange(ncrops):
    print "IMG: ",iimg
    img = y2u_v.at(iimg)
    c.Clear()
    
    hist = larcv.as_th2d( img, "y2u_%02d"%(iimg) )
    hist.Draw("COLZ")
    c.Update()
    raw_input()


