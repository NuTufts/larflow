from __future__ import print_function
import os,sys,argparse,time
"""
Runs the larflow::reco::LikelihoodProtonMuon algorithm
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-kps','--input-kpsana',type=str,required=True,help="Input file containing the Ana products of KPSRecoManager")
parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="Input file containing larcv images")
#parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
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

algo = larflow.reco.ShowerBilineardEdx()
algo.set_verbosity( larcv.msg.kDEBUG )
#algo.set_verbosity( larcv.msg.kINFO )
#mcdata = ublarcvapp.mctools.LArbysMC()

#tfana = rt.TFile( args.output.replace(".root","_ana.root"), "recreate" )
#tfana.cd()
#vatree = rt.TTree("vtxactivityana","Vertex Activity Ana")
#mcdata.bindAnaVariables( vatree )
#algo.bind_to_tree( vatree )

curve = []
hist_dqdx_ave = {}
hist_ll = {}
hist_2d = {}
canv = {}
for p in range(4):
    c = rt.TCanvas("c_%d"%(p),"Plane %d"%(p),1500,800)
    #c.Divide(3,2)
    canv[p] = c    
    for vtx in ["good","bad"]:
        hist_dqdx_ave[(p,vtx)] = rt.TH1D( "hdqdx_ave_%d_%s"%(p,vtx), "", 100, 0, 200)
        hist_ll[(p,vtx)] = rt.TH1D( "hll_%d_%s"%(p,vtx), "", 50, -25, 25 )
        hist_2d[(p,vtx)] = rt.TH2D( "h2d_%d_%s"%(p,vtx), "", 20, 0, 10, 25, 0, 300 )

canv["curve"] = rt.TCanvas("ccurve","Curves",1200,500)
canv["curve"].Divide(2,1)

hbg1 = rt.TH2D("hbg1","", 50, 0, 10, 50, 0, 300)
hbg2 = rt.TH2D("hbg2","", 50, 0, 10, 50, 0, 300)
canv["curve"].cd(1)
hbg1.Draw()
canv["curve"].cd(2)
hbg2.Draw()

start_entry = 0
for ientry in range(start_entry,nentries):

    tree.GetEntry(ientry)
    iolcv.read_entry(ientry)
    
    rse = ( tree.run, tree.subrun, tree.event )
    print("[ENTRY ",ientry,"]: ",rse)

    ev_adc = iolcv.get_data(larcv.kProductImage2D,"wire")
    adc_v = ev_adc.as_vector()
    hist_v = larcv.rootutils.as_th2d_v( adc_v, "histentry%d"%ientry )
    
    vertex_v = tree.nufitted_v
    print(" Number of vertices: ",vertex_v.size())
    nuperfect_v = tree.nu_perfect_v
    print(" Number of perfect vertices: ",vertex_v.size())    

    maxlen = 0

    for (name,vertices) in [("RECO",vertex_v),("PERFECT",nuperfect_v)]:
        print("=====================")
        print("%s VERTICES"%(name))
        for ivtx in range( vertices.size() ):

            nuvtx = vertices.at(ivtx)        
            ntracks  = nuvtx.track_v.size()
            nshowers = nuvtx.shower_v.size()
            nusel = tree.nu_sel_v[ivtx]

            # cuts
            if name in ("RECO"):
                if nusel.ntracks==0 or nusel.ntracks>2:
                    continue
                if nusel.max_shower_nhits<500:
                    continue
                if nusel.nshowers==0 or nusel.nshowers>2:
                    continue
        
            print(" VTX %d (%.2f) dist2true=%.2f ntracks=%d nshowers=%d"%(ivtx,vertex_v.at(ivtx).score,nusel.dist2truevtx,ntracks,nshowers))

            tvtx = rt.TVector3()
            for i in range(3):
                tvtx[i] = nuvtx.pos[i]

            if nusel.dist2truevtx<3.0:
                goodvtx = "good"
            else:
                goodvtx = "bad"

            for ishower in range(nshowers):
                shower = nuvtx.shower_v[ishower]
                trunk  = nuvtx.shower_trunk_v[ishower]
                shpca  = nuvtx.shower_pcaxis_v[ishower]
                algo.processShower( shower, trunk, shpca, adc_v )

                for p in range(3):
                    print(canv[p])
                    canv[p].Draw()
                    print(hist_v[p])
                    hist_v[p].Draw("colz")
                    for ipath in range(algo.bilinear_path_vv.at(p).size()):
                        path = algo.bilinear_path_vv.at(p).at(ipath)
                        path.Draw("Lsame")
                    canv[p].Draw()
                    canv[p].Update()
                    if True:
                        print("[ENTER] to continue to next plane for shower")
                        x = input()


                if True:
                    print("[ENTER] to continue")
                    x = input()
            if True:
                break
    
    if True:
        break


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
