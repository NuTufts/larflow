import os,sys,argparse,time

parser = argparse.ArgumentParser("prepare 3d spatial embed data")
parser.add_argument("-ilcv","--input-larcv",required=False,type=str,help="Input LArCV file [required]")
parser.add_argument("-ill","--input-larlitetruth",required=False,default=None,type=str,help="Input larlite truth file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output (ROOT) file name [required]")
parser.add_argument("-n", "--nentries",type=int,default=None,help="Number of entries to run [default: None (all)]")
parser.add_argument("-s", "--start-entry",type=int,default=0,help="starting entry")
parser.add_argument("-adc","--adc",type=str,default="wire",help="ADC image name")
args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

detrmaker = larflow.detr.PrepDETR2D()
detrmaker.set_verbosity(larcv.msg.kDEBUG)

ioll  = larlite.storage_manager( larlite.storage_manager.kREAD )
iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

print "[INPUT: LArCV   - DL MERGED] ",args.input_larcv
print "[INPUT: larlite - MCINFO]  ",args.input_larlitetruth
print "[OUTPUT]    ",args.output

ioll.add_in_filename( args.input_larlitetruth )
ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.set_data_to_read( larlite.data.kMCShower, "mcreco" )
ioll.set_data_to_read( larlite.data.kMCTruth,  "generator" )

iolcv.add_in_file(   args.input_larcv )
iolcv.specify_data_read( larcv.kProductImage2D, args.adc );
iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
iolcv.specify_data_read( larcv.kProductChStatus, "wiremc" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.reverse_all_products()


ioll.open()
iolcv.initialize()

outfile = rt.TFile( args.output.replace(".root","_detr2d.root"), "recreate" )
detrmaker.setupForOutput()

lcv_nentries = iolcv.get_n_entries()
ll_nentries  = ioll.get_entries()
if lcv_nentries<ll_nentries:
    nentries = lcv_nentries
else:
    nentries = ll_nentries
print "[Number of entries: ",nentries,"]"
    
if args.nentries is not None:
    print "Num entries specified by argument: ",args.nentries
    end_entry = args.start_entry + args.nentries
    if end_entry>nentries:
        end_entry = nentries
else:
    end_entry = nentries

print "[Run between event (",args.start_entry,",",end_entry,")"

if args.start_entry>0:
    ioll.go_to( args.start_entry )
else:
    ioll.go_to(0);
    
for ientry in xrange( args.start_entry, end_entry ):
    print "[ENTRY ",ientry,"]"
    iolcv.read_entry(ientry)
    ioll.go_to( ientry )
    print " ... process hits ..."
    data = detrmaker.process( iolcv, ioll )
    print " ... done ..."

ioll.close()
iolcv.finalize()

outfile.Write()
print "[FIN] clean-up"


