from __future__ import print_function
import os,sys,argparse,time
"""
Runs the larflow::reco::LikelihoodProtonMuon algorithm
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-i','--input-kpsana',type=str,required=True,help="Input file containing the Ana products of KPSRecoManager")
parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="Input file containing the Ana products of KPSRecoManager")
#parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from ROOT import larutil
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

print("[INPUT: KPSRecoManager ana output]  ",args.input_kpsana)
print("[INPUT: DL-MERGED] ",args.input_dlmerged)
#print("[OUTPUT]    ",args.output)

# ========================================
# LOAD dq/dx expectation curves
q2adc = 4.0*93.0/2.2
splinefile = rt.TFile( "../data/Proton_Muon_Range_dEdx_LAr_TSplines.root" )
sMuonRange2dEdx = splinefile.Get("sMuonRange2dEdx")
sProtonRange2dEdx = splinefile.Get("sProtonRange2dEdx")
xend = 100
mu_curve = rt.TGraph( int(xend) )
for i in xrange(int(xend)):
    x = i*1.0+0.5
    y = sMuonRange2dEdx.Eval(x)
    y2 = q2adc*y/(1+y*0.0486/0.273/1.38)    
    mu_curve.SetPoint(i,x,y2)
mu_curve.SetLineColor(rt.kBlack)
mu_curve.SetLineWidth(2)

pi_curve = rt.TGraph( int(xend) )
for i in xrange(int(xend)):
    x = i*1.0+0.5
    y = sMuonRange2dEdx.Eval(x)
    y2 = q2adc*y/(1+y*0.0486/0.273/1.38)    
    pi_curve.SetPoint(i,x,y2)
pi_curve.SetLineColor(rt.kBlack)
pi_curve.SetLineWidth(2)

p_curve = rt.TGraph( int(xend) )
for i in xrange(int(xend)):
    x = i*1.0+0.5
    y = sProtonRange2dEdx.Eval(x)
    y2 = q2adc*y/(1+y*0.0486/0.273/1.38)
    p_curve.SetPoint(i,x,y2)
p_curve.SetLineColor(rt.kBlack)
p_curve.SetLineWidth(2)


io = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )
#io = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
io.add_in_file( args.input_dlmerged )
io.specify_data_read( larcv.kProductImage2D, "wire" )
io.specify_data_read( larcv.kProductChStatus, "wire" )
io.reverse_all_products()
io.initialize()

# OPEN VERTEX RECO FILE
anafile = rt.TFile( args.input_kpsana )
tree = anafile.Get("KPSRecoManagerTree")
nentries = tree.GetEntries()

algo = larflow.reco.LikelihoodProtonMuon()
algo.set_verbosity( larcv.msg.kDEBUG )
#algo.set_verbosity( larcv.msg.kINFO )
#mcdata = ublarcvapp.mctools.LArbysMC()
splitter = larflow.reco.TrackFindBadConnection()
splitter.set_verbosity( larcv.msg.kDEBUG )
dqdx_algo = larflow.reco.TrackdQdx()

#tfana = rt.TFile( args.output.replace(".root","_ana.root"), "recreate" )
#tfana.cd()
#vatree = rt.TTree("vtxactivityana","Vertex Activity Ana")
#mcdata.bindAnaVariables( vatree )
#algo.bind_to_tree( vatree )

c = rt.TCanvas("c","c",1500,1000)
        
start_entry = 8
for ientry in range(start_entry,nentries):

    tree.GetEntry(ientry)
    io.read_entry(ientry)
    
    rse = ( tree.run, tree.subrun, tree.event )
    print("[ENTRY ",ientry,"]: ",rse)

    ev_adc = io.get_data( larcv.kProductImage2D, "wire" )
    adc_v = ev_adc.as_vector()

    hist_v = larcv.rootutils.as_th2d_v( adc_v, "histentry%d"%ientry )    
    
    vertex_v = tree.nufitted_v
    curve = []
    maxlen = 0

    c.Clear()
    
    c.Divide(3,2)
    for p in range(3):
        c.cd(1+p)
        hist_v[p].Draw("colz")
        hist_v[p].GetZaxis().SetRangeUser(0,150.0)
    c.Update()
    
    for ivtx in range( tree.nufitted_v.size() ):
        
        vtx_markers = []
        
        nuvtx = vertex_v.at(ivtx)
        vtxtype = nuvtx.keypoint_type
        ntracks  = nuvtx.track_v.size()
        nshowers = nuvtx.shower_v.size()
        print(" VTX %d (%.2f) type=%d ntracks=%d nshowers=%d"%(ivtx,vertex_v.at(ivtx).score,vtxtype,ntracks,nshowers))
        if vtxtype != 0:
            print("Not Nu-Vtx skipping.")
            continue

        # first draw canvas
        for p in range(3):
            gvtx = rt.TMarker(nuvtx.col_v[p],nuvtx.tick,24)
            gvtx.SetMarkerSize(2)
            gvtx.SetMarkerColor(rt.kMagenta)
            print(nuvtx.col_v[p],nuvtx.tick)
            c.cd(1+p)
            hist_v[p].GetXaxis().SetRangeUser( nuvtx.col_v[p]-100, nuvtx.col_v[p]+100 )
            hist_v[p].GetYaxis().SetRangeUser( nuvtx.tick-100, nuvtx.tick+100 )
            gvtx.Draw()
            vtx_markers.append(gvtx)
        c.Update()
        c.Draw()
            

        for itrack in range(ntracks):

            g_v = {}
            
            lltrack = nuvtx.track_v.at(itrack)
            llpid   = algo.calculateLL( lltrack, nuvtx.pos )
            npts    = lltrack.NumberTrajectoryPoints()
            lastpt  = lltrack.LocationAtPoint(npts-1)
            endpt   = ( lastpt[0], lastpt[1], lastpt[2] )
            resrange = 0.0

            split_v = splitter.splitBadTrack( lltrack, adc_v, 5.0 )
            print (" nsplits=",split_v.size())

            plane_dqdx_vv = dqdx_algo.calculatedQdx2D( lltrack, adc_v, 0.3 )
            print ("ran calculatedQdx2D!")

            for p in range(3):
                gp = rt.TGraph(npts)
                gv = rt.TGraph(1)                
                for ipt in range(npts):
                    y = lltrack.LocationAtPoint(ipt)[0]/larutil.LArProperties.GetME().DriftVelocity()/0.5+3200
                    x = larutil.Geometry.GetME().WireCoordinate( lltrack.LocationAtPoint(ipt), p )
                    gp.SetPoint(ipt,x,y)
                    if ipt==0:
                        gv.SetPoint(ipt,x,y)
                
                gp.SetLineColor(rt.kRed)
                gp.SetLineWidth(2)
                gp.SetMarkerStyle(21)
                gp.SetMarkerSize(1)
                
                gv.SetLineColor(rt.kRed)
                gv.SetMarkerColor(rt.kRed)
                gv.SetMarkerSize(2)
                gv.SetMarkerStyle(22)

                c.cd(p+1)
                
                g_v[p] = gp
                g_v[(p,200)] = gv
                g_v[p].Draw("LP")
                gv.Draw("P")

                if split_v.size()>0:
                    t = split_v.at(0)
                    nt = t.NumberTrajectoryPoints()
                    gs = rt.TGraph(nt)
                    for ipt in range(nt):
                        y = t.LocationAtPoint(ipt)[0]/larutil.LArProperties.GetME().DriftVelocity()/0.5+3200
                        x = larutil.Geometry.GetME().WireCoordinate( t.LocationAtPoint(ipt), p )
                        gs.SetPoint(ipt,x,y)
                    gs.SetLineColor(rt.kMagenta)
                    gs.SetLineWidth(2)
                    gs.SetMarkerStyle(21)
                    gs.SetMarkerSize(1)
                    gs.SetMarkerColor(rt.kMagenta)
                    gs.Draw("LP")
                    g_v[(p,2)] = gs

                if npts>0 and False:
                    g = rt.TGraph(npts)
                    resrange = 0.0
                    lastpt = lltrack.LocationAtPoint(npts-1)
                    for ipt in range(npts):
                        dqdx = lltrack.DQdxAtPoint( npts-ipt-1, p )
                        currentpt = lltrack.LocationAtPoint(npts-ipt-1)
                        #print("    track[%d] %d = %.1f"%(itrack,npts,dqdx))
                        g.SetPoint(ipt,resrange,dqdx)
                        resrange += (currentpt-lastpt).Mag()
                        lastpt = currentpt
                    if llpid<0:
                        g.SetLineColor(rt.kRed)
                    c.cd(3+p+1)                    
                    g.Draw("AL")
                    g_v[(p,1)] = g                    

                print(" plane_dqdx_vv[%d].size()=%d"%(p,plane_dqdx_vv[p].size()))
                if plane_dqdx_vv[p].size()>0:
                    # need the length of the track
                    plane_dqdx_len = plane_dqdx_vv[3+p][ plane_dqdx_vv[3+p].size()-1 ]
                    gdqdx = rt.TGraph( plane_dqdx_vv.at(p).size() )
                    for ipt in range( plane_dqdx_vv.at(p).size() ):
                        lenx = plane_dqdx_len - plane_dqdx_vv[3+p][ipt]
                        gdqdx.SetPoint(ipt, lenx, plane_dqdx_vv[p][ipt])
                    gdqdx.SetLineColor(rt.kMagenta)
                    gdqdx.SetMarkerStyle(22)
                    gdqdx.SetMarkerColor(rt.kMagenta)                
                
                    c.cd(3+p+1)
                    if npts>0:
                        gdqdx.Draw("APL")
                    else:
                        gdqdx.Draw("APL")
                    gdqdx.GetYaxis().SetRangeUser(0,3000)
                    g_v[(p,10)] =  gdqdx

            data = (itrack,npts,endpt[0],endpt[1],endpt[2],lastpt[0],lastpt[1],lastpt[2],resrange,llpid)
            print("    :: [track %d] npts=%d end=(%.1f,%.1f,%.1f) start=(%.1f,%.1f,%.1f) len=%.2f cm likelihood ratio=%.2f"%data)

            for p in range(3):
                c.cd(3+p+1)
                p_curve.Draw("L")
                mu_curve.Draw("L")
            
            c.Update()
            c.Draw()
            raw_input()

    print("[ENTER] to continue")
    raw_input()
    

print("[END]")
