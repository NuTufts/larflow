from __future__ import print_function
import os,sys,argparse,time
"""
Runs the larflow::reco::LikelihoodProtonMuon algorithm
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-kps','--input-kpsana',type=str,required=True,help="Input file containing the Ana products of KPSRecoManager")
parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="Input file containing larcv images")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from ROOT import larutil
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

try:
    #in python 2 raw_input exists, so use that
    input = raw_input
except NameError:
    #in python 3 we hit this case and input is already raw_input
    pass

rt.gStyle.SetOptStat(0)

PLOTME=False

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

print("[INPUT: DL MERGED] ",args.input_dlmerged)
print("[INPUT: RECO OUT]  ",args.input_kpsana)
#print("[INPUT (optional): MCINFO]  ",args.input_mcinfo)
#print("[OUTPUT]    ",args.output)

iolcv.add_in_file(   args.input_dlmerged )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
#iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
#iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
#iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
#iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
#iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
#iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
#iolcv.addto_storeonly_list( ... )
iolcv.reverse_all_products()
iolcv.initialize()

# OPEN VERTEX RECO FILE
anafile = rt.TFile( args.input_kpsana )
tree = anafile.Get("KPSRecoManagerTree")
nentries = tree.GetEntries()

# Setup algorithm to run on reco showers
reco_algo = larflow.reco.ShowerBilineardEdx()
reco_algo.set_verbosity( larcv.msg.kDEBUG )
# Setup algorithm to run on perfect reconstruction showers
perfect_algo = larflow.reco.ShowerBilineardEdx()
perfect_algo.set_verbosity( larcv.msg.kDEBUG )
# Truth data for analysis
mcdata = ublarcvapp.mctools.LArbysMC()

tfana = rt.TFile( args.output, "recreate" )
tfana.cd()
algotree = rt.TTree("showerbilinear","ShowerBilineardEdx output variables")
reco_algo.bindVariablesToTree( algotree )
perfect_tree = rt.TTree("perfectreco","ShowerBilineardEdx output variables on Perfect reco")
perfect_algo.bindVariablesToTree( perfect_tree )

start_entry = 0
for ientry in range(start_entry,nentries):

    tree.GetEntry(ientry)
    iolcv.read_entry(ientry)
    
    rse = ( tree.run, tree.subrun, tree.event )
    print("[ENTRY ",ientry,"]: ",rse)

    ev_adc = iolcv.get_data(larcv.kProductImage2D,"wire")
    adc_v = ev_adc.as_vector()

    if PLOTME:
        hist_v = larcv.rootutils.as_th2d_v( adc_v, "histentry%d"%ientry )
        mask_v = {}
        graphs = []        
        for p in range(3):
            m = hist_v[p].Clone( "maskentry%d_plane%d"%(ientry,p))
            m.Reset()
            mask_v[p] = m

        c = rt.TCanvas("c","",1500,1200)
        c.Divide(3,3)
        for p in range(3):
            c.cd(p+1)
            hist_v[p].Draw("colz")
            c.cd(3+1+p)
            mask_v[p].Draw("colz")
    
    vertex_v = tree.nufitted_v
    print(" Number of vertices: ",vertex_v.size())
    nuperfect_v = tree.nu_perfect_v
    print(" Number of perfect vertices: ",vertex_v.size())    

    maxlen = 0

    for (name,vertices,algo,outtree) in [("RECO",vertex_v,reco_algo,algotree),("PERFECT",nuperfect_v,perfect_algo,perfect_tree)]:
        nvertices = vertices.size()
        print("=====================")
        print("%s VERTICES"%(name),": n=",nvertices)
        if nvertices==0:
            continue
        for ivtx in range( nvertices ):

            nuvtx = vertices.at(ivtx)        
            ntracks  = nuvtx.track_v.size()
            nshowers = nuvtx.shower_v.size()

            # cuts
            if name in ("RECO"):                            
                nusel = tree.nu_sel_v[ivtx]
                if nusel.ntracks==0 or nusel.ntracks>2:
                    continue
                if nusel.max_shower_nhits<500:
                    continue
                if nusel.nshowers==0 or nusel.nshowers>2:
                    continue

                if nusel.dist2truevtx<3.0:
                    goodvtx = "good"
                else:
                    goodvtx = "bad"
                
                print(" VTX %d (%.2f) dist2true=%.2f ntracks=%d nshowers=%d"%(ivtx,vertex_v.at(ivtx).score,nusel.dist2truevtx,ntracks,nshowers))

            tvtx = rt.TVector3()
            for i in range(3):
                tvtx[i] = nuvtx.pos[i]

            for ishower in range(nshowers):
                shower = nuvtx.shower_v[ishower]
                trunk  = nuvtx.shower_trunk_v[ishower]
                shpca  = nuvtx.shower_pcaxis_v[ishower]
                algo.processShower( shower, trunk, shpca, adc_v )

                if PLOTME:
                    for p in range(3):
                        c.cd(p+1)
                        for ipath in range(algo.bilinear_path_vv.at(p).size()):
                            path = algo.bilinear_path_vv.at(p).at(ipath)
                            path.SetLineWidth(2)
                            if name in ["PERFECT"]:
                                path.SetLineColor(rt.kRed)
                            else:
                                path.SetLineColor(rt.kMagenta)                                
                            path.Draw("Lsame")
                            graphs.append(path)
                        c.Update()

                outtree.Fill()
                    
                if False:
                    print("[ENTER] to continue")
                    x = input()
            # end of reoc vertex loop
            if False:
                break
    
            if name in ["PERFECT"] and PLOTME:
                tick1 = nuvtx.tick - 6*50
                tick2 = nuvtx.tick + 6*50
                dpos = std.vector("double")(3)
                for i in range(3):
                    dpos[i] = nuvtx.pos[i]
                for p in range(3):
                    c.cd(3+p+1)
                    #algo.maskPixels( p, mask_v[p] )
                    algo._debug_crop_v[p].Draw("colz")
                    c.cd(p+1)                    
                    wire = larutil.Geometry.GetME().WireCoordinate(dpos,p)
                    col1 = wire-50
                    col2 = wire+50
                    hist_v[p].GetXaxis().SetRangeUser(col1,col2)
                    hist_v[p].GetYaxis().SetRangeUser(tick1,tick2)
                    hist_v[p].GetZaxis().SetRangeUser(0,200.0)
                    mask_v[p].GetXaxis().SetRangeUser(col1,col2)
                    mask_v[p].GetYaxis().SetRangeUser(tick1,tick2)
                    mask_v[p].GetZaxis().SetRangeUser(0,10.0)
                    c.cd(6+p+1)
                    g = algo.makeSegdQdxGraphs(p)
                    g.Draw("APL")                    
                    graphs.append(g)
                    g_reco = reco_algo.makeSegdQdxGraphs(p)
                    if g_reco.GetN()>1:
                        g_reco.SetLineColor(rt.kRed)
                        g_reco.Draw("PL")                        
                        graphs.append(g_reco)                        
                    c.Update()
                    
    if PLOTME:
        c.Update()
        c.Draw()
        x = input("[ENTER] to go to next event")
    if False:
        break
    # end of entry

print("WRITE")
tfana.cd()
algotree.Write()
perfect_tree.Write()

if True:
    sys.exit(0)
    
#canv["curve"].Draw()
#hbg.Draw()
#for c in curve:
#    c.Draw("Lsame")
canv["curve"].Update()

for p in range(4):
    canv[p].Draw()
    canv[p].cd(1)
    hist_2d[(p,"good")].Draw("colz")
    canv[p].cd(2)
    hist_dqdx_ave[(p,"good")].Draw("hist")
    canv[p].cd(3)
    hist_ll[(p,"good")].Draw("hist")

    canv[p].cd(4)
    hist_2d[(p,"bad")].Draw("colz")
    canv[p].cd(5)
    hist_dqdx_ave[(p,"bad")].Draw("hist")
    canv[p].cd(6)
    hist_ll[(p,"bad")].Draw("hist")
    
    canv[p].Update()
    
raw_input()


print("[END]")
