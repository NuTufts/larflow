from __future__ import print_function
import os,sys,argparse,time
"""
Runs the larflow::reco:: algorithm
"""

parser = argparse.ArgumentParser("Run larflow3dhit clustering algorith")
# required
parser.add_argument('-kps','--input-kpsana',type=str,required=True,help="Input file containing the Ana products of KPSRecoManager")
parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="Input file containing larcv images")
parser.add_argument('-o','--output',type=str,required=True,help="Name of output file. Will not overwrite")
parser.add_argument("-n",'--nentries',default=None,type=int,help="Number of entries to run")
parser.add_argument("-tf","--tick-forward",default=False,action='store_true',help="If true, run in tick-forward mode")

args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

if not args.tick_forward:
    iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickBackward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kBOTH, "larcv", larcv.IOManager.kTickForward )
#iolcv.set_verbosity( larcv.msg.kDEBUG )

print("[INPUT: DL MERGED] ",args.input_dlmerged)
print("[INPUT: RECO OUT]  ",args.input_kpsana)
#print("[INPUT (optional): MCINFO]  ",args.input_mcinfo)
print("[OUTPUT]    ",args.output)

iolcv.add_in_file(  args.input_dlmerged )
iolcv.set_out_file( args.output )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
#iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
#iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
#iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
#iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
#iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
#iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
#iolcv.addto_storeonly_list( ... )
if not args.tick_forward:
    iolcv.reverse_all_products()
iolcv.initialize()

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename(  args.input_dlmerged )
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )
ioll.open()


# OPEN VERTEX RECO FILE
anafile = rt.TFile( args.input_kpsana )
tree = anafile.Get("KPSRecoManagerTree")
nentries = tree.GetEntries()

algo = larflow.reco.NuSelUnrecoCharge()
algo.set_verbosity( larcv.msg.kDEBUG )
#algo.set_verbosity( larcv.msg.kINFO )

mcdata = ublarcvapp.mctools.LArbysMC()

tfana = rt.TFile( args.output.replace(".root","_ana.root"), "recreate" )
tfana.cd()
anatree = rt.TTree("unrecoq","Vertex Activity Ana")
mcdata.bindAnaVariables( anatree )
algo.bindVarsToTree( anatree )

if args.nentries is not None and args.nentries<nentries:
    nentries = args.nentries

start_entry = 0
for ientry in range(start_entry,nentries):

    tree.GetEntry(ientry)
    iolcv.read_entry(ientry)
    ioll.go_to(ientry)
    
    rse = ( tree.run, tree.subrun, tree.event )
    print("[ENTRY ",ientry,"]: ",rse)

    vertex_v = tree.nufitted_v
    print(" Number of vertices: ",vertex_v.size())

    maxlen = 0

    #mcdata.clear()
    mcdata.process(ioll)
    mcdata.printInteractionInfo()

    for ivtx in range( tree.nufitted_v.size() ):

        nuvtx = vertex_v.at(ivtx)        
        ntracks  = nuvtx.track_v.size()
        nshowers = nuvtx.shower_v.size()
        nusel = tree.nu_sel_v[ivtx]

        print(" VTX %d (%.2f) dist2true=%.2f ntracks=%d nshowers=%d"%(ivtx,vertex_v.at(ivtx).score,nusel.dist2truevtx,ntracks,nshowers))

        outtemp = larflow.reco.NuSelectionVariables()

        if abs(mcdata._nu_pdg)==12:
            algo.setSaveMask( True )
        else:
            algo.setSaveMask( False )
            
        algo.analyze( iolcv, ioll, nuvtx, outtemp )
        anatree.Fill()        

    iolcv.save_entry()

    
    if False:
        raw_input()        
        break

iolcv.finalize()
tfana.cd()
anatree.Write()
print("[END]")
