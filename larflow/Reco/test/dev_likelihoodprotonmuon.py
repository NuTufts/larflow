from __future__ import print_function
import os,sys,argparse,time
"""
Runs the larflow::reco::LikelihoodProtonMuon algorithm
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-i','--input-kpsana',type=str,required=True,help="Input file containing the Ana products of KPSRecoManager")
#parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

print("[INPUT: KPSRecoManager ana output]  ",args.input_kpsana)
#print("[OUTPUT]    ",args.output)

# OPEN VERTEX RECO FILE
anafile = rt.TFile( args.input_kpsana )
tree = anafile.Get("KPSRecoManagerTree")
nentries = tree.GetEntries()

algo = larflow.reco.LikelihoodProtonMuon()
algo.set_verbosity( larcv.msg.kDEBUG )
#algo.set_verbosity( larcv.msg.kINFO )
#mcdata = ublarcvapp.mctools.LArbysMC()

#tfana = rt.TFile( args.output.replace(".root","_ana.root"), "recreate" )
#tfana.cd()
#vatree = rt.TTree("vtxactivityana","Vertex Activity Ana")
#mcdata.bindAnaVariables( vatree )
#algo.bind_to_tree( vatree )

c = rt.TCanvas("c","c",1000,500)

start_entry = 22
for ientry in range(start_entry,nentries):

    tree.GetEntry(ientry)
    
    rse = ( tree.run, tree.subrun, tree.event )
    print("[ENTRY ",ientry,"]: ",rse)

    vertex_v = tree.nufitted_v
    curve = []
    maxlen = 0
    for ivtx in range( tree.nufitted_v.size() ):
        nuvtx = vertex_v.at(ivtx)
        ntracks  = nuvtx.track_v.size()
        nshowers = nuvtx.shower_v.size()
        print(" VTX %d (%.2f) ntracks=%d nshowers=%d"%(ivtx,vertex_v.at(ivtx).score,ntracks,nshowers))

        for itrack in range(ntracks):
            lltrack = nuvtx.track_v.at(itrack)
            llpid   = algo.calculateLL( lltrack, nuvtx.pos )
            npts    = lltrack.NumberTrajectoryPoints()
            lastpt  = lltrack.LocationAtPoint(0)
            endpt   = ( lastpt[0], lastpt[1], lastpt[2] )
            resrange = 0.0
            g = rt.TGraph(npts)
            for ipt in range(npts):
                dqdx = lltrack.DQdxAtPoint( ipt, 2 )
                currentpt = lltrack.LocationAtPoint(ipt)
                #print("    track[%d] %d = %.1f"%(itrack,npts,dqdx))
                g.SetPoint(ipt,resrange,dqdx)
                resrange += (currentpt-lastpt).Mag()
                lastpt = currentpt
                
            if llpid<0:
                g.SetLineColor(rt.kRed)
            curve.append(g)
            if resrange>maxlen:
                maxlen = resrange
            data = (itrack,npts,endpt[0],endpt[1],endpt[2],lastpt[0],lastpt[1],lastpt[2],resrange,llpid)
            print("    :: [track %d] npts=%d end=(%.1f,%.1f,%.1f) start=(%.1f,%.1f,%.1f) len=%.2f cm likelihood ratio=%.2f"%data)
    h = rt.TH2D("h",";residual range (cm); dQ/dx",10,0,maxlen,10,0,500)
    c.Clear()
    c.Draw()
    h.Draw()
    if len(curve)>0:
        for g in curve:
            g.Draw("Lsame")
    c.Update()
    print("[ENTER] to continue")
    input()
    
    break


print("[END]")
