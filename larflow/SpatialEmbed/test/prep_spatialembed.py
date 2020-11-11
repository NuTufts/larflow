import os,sys,argparse,time

parser = argparse.ArgumentParser("prepare 3d spatial embed data")
parser.add_argument("-ilm", "--input-larmatch",required=True,type=str,help="Input larmatch file (larlite type) [required]")
parser.add_argument("-ilcv","--input-larcv",required=False,type=str,help="Input LArCV file [required]")
parser.add_argument("-il","--input-larlitetruth",required=False,type=str,help="Input larlite truth file [required]")
parser.add_argument("-o","--output",required=True,type=str,help="output (ROOT) file name [required]")
parser.add_argument("-n", "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
parser.add_argument("-s", "--start-entry",type=int,default=0,help="starting entry")
args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

voxelmaker = larflow.spatialembed.Prep3DSpatialEmbed()
voxelmaker.set_verbosity(larcv.msg.kDEBUG)

io = larlite.storage_manager( larlite.storage_manager.kREAD )
iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

print "[INPUT: LArCV   - DL MERGED] ",args.input_larcv
print "[INPUT: larlite - LARMATCH-KPS]  ",args.input_larmatch
print "[OUTPUT]    ",args.output

io.add_in_filename(  args.input_larcv )
io.add_in_filename(  args.input_larmatch )
io.set_data_to_read( larlite.data.kLArFlow3DHit, "larmatch" )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashBeam" )
io.set_data_to_read( larlite.data.kOpFlash,  "simpleFlashCosmic" )


iolcv.add_in_file(   args.input_larcv )
iolcv.specify_data_read( larcv.kProductImage2D, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "thrumu" );
iolcv.specify_data_read( larcv.kProductImage2D, "ancestor" );
iolcv.specify_data_read( larcv.kProductImage2D, "segment" );
iolcv.specify_data_read( larcv.kProductImage2D, "instance" );
iolcv.specify_data_read( larcv.kProductImage2D, "larflow" );
iolcv.specify_data_read( larcv.kProductChStatus, "wire" );
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane0" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane1" )
iolcv.specify_data_read( larcv.kProductImage2D, "ubspurn_plane2" )
#iolcv.addto_storeonly_list( ... )
iolcv.reverse_all_products()
#io.set_out_filename( args.output.replace(".root","_larlite.root") )
#iolcv.set_out_file( args.output.replace(".root","_larcv.root") )

io.open()
iolcv.initialize()

outfile = rt.TFile( args.output, "recreate" )
outtree = rt.TTree("s3dembed", "Spatial 3D embed test and train data")
voxelmaker.bindVariablesToTree( outtree )

lcv_nentries = iolcv.get_n_entries()
ll_nentries  = io.get_entries()
if lcv_nentries<ll_nentries:
    nentries = lcv_nentries
else:
    nentries = ll_nentries
print "[Number of entries: ",nentries,"]"
    
if args.nentries is not None:
    end_entry = args.start_entry + nentries
    if end_entry>nentries:
        end_entry = nentries
    pass
else:
    end_entry = nentries
    pass

print "[Run between event (",args.start_entry,",",end_entry,")"

if args.start_entry>0:
    io.go_to( args.start_entry )
else:
    io.go_to(0);

    
#io.next_event()
#io.go_to( args.start_entry )
for ientry in xrange( args.start_entry, end_entry ):
    print "[ENTRY ",ientry,"]"
    iolcv.read_entry(ientry)
    print " ... process hits ..."
    data = voxelmaker.process( iolcv, io, False )
    print " ... done ..."

    voxelmaker.fillTree( data )
    outtree.Fill()
    
    io.set_id( io.run_id(), io.subrun_id(), io.event_id() )
    io.next_event()
    break

#io.close()
#iolcv.finalize()

outfile.Write()
print "[FIN] clean-up"

