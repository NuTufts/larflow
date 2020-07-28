import os,sys
import ROOT as rt
from ROOT import std
import numpy

from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow


rt.gStyle.SetOptStat(0)

iolcv = larcv.IOManager( larcv.IOManager.kREAD )
iolcv.add_in_file( "../../../../testdata/mcc9_v13_bnbnue_corsika/larcvtruth-Run000001-SubRun000001.root" )
iolcv.initialize()
iolcv.set_verbosity(1)


ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( "../../../../testdata/mcc9_v13_bnbnue_corsika/mcinfo-Run000001-SubRun000001.root" )
ioll.set_verbosity(1)
ioll.open()

nentries = iolcv.get_n_entries()

c = rt.TCanvas("c","c",800,600)
c.Draw()

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

    ev_adc = iolcv.get_data( larcv.kProductImage2D, "wiremc" )
    adc_v = ev_adc.Image2DArray()

    hadc_v = larcv.rootutils.as_th2d_v( adc_v, "ev_adc_%d"%(ientry) )

    for p in xrange(3):
        print "plane: ",p
        hadc_v[p].Draw("colz")
        id_v = prepembed.get_id_list(p)
        print " num instances: ",id_v.size()
        meta = adc_v[p].meta()
        for iid in xrange(id_v.size()):
            aid = id_v[iid]
            a_node = mcpg.findTrackID( aid )
            pix_v = prepembed.get_instance_pixlist( p, aid )
            print "ancestor id: ",aid," npixels=",pix_v.size()," pid=",a_node.pid," origin=",a_node.origin
            
            tg = rt.TGraph( pix_v.size() )
            for ipt in xrange(pix_v.size()):
                tg.SetPoint( ipt, meta.pos_x(pix_v[ipt].col), meta.pos_y(pix_v[ipt].row) )
            tg.SetMarkerStyle(20)
            tg.SetMarkerColor(rt.kRed)
            tg.Draw("P")
            c.Update()
            raw_input()

