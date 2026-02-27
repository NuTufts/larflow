import os,sys
import argparse

# parse command line arguments
parser = argparse.ArgumentParser(description='Process larcv/larlite into HDF5 entry data files.')
parser.add_argument("-i","--input-dlmerged",required=True,type=str,help="Input dlmerged file")
parser.add_argument("-v","--verbosity",type=int,default=2,help="Verbosity level from normal=2 to debug=0")

args = parser.parse_args(sys.argv[1:])

import ROOT
from larlite import larlite
from larcv import larcv
from larflow import larflow

dlmerged_input = args.input_dlmerged

start_entry = 0
end_entry = -1

# Setup algorithm
simchmaker = larflow.prep.SimChTripletLabelMaker()
simchmaker.set_verbosity(args.verbosity)
simchmaker.save_truth_tripletinfo( True )

# setup truth point label maker
simchmaker._mcpixelmaker.set_verbosity(args.verbosity)
# configure to use wirecell driftWC simch
simchmaker._mcpixelmaker.set_driftwc_source()
# how much do we bleed out the truth labels?
simchmaker._mcpixelmaker.set_dwire(1)
simchmaker._mcpixelmaker.set_drow(0)

# set keypoint maker verbosity
simchmaker._mckpmaker.set_verbosity(args.verbosity)

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( dlmerged_input )
ioll.set_verbosity(2)
ioll.open()

iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv" )
iolcv.add_in_file( dlmerged_input )
iolcv.set_verbosity(2)
iolcv.initialize()

nentries = ioll.get_entries()
if end_entry<0 or end_entry>=nentries:
  end_entry = nentries
else:
  end_entry = end_entry+1

run_nentries = 2

# process input file name. we will use it to name the invididual event files
basefilename = os.path.basename( args.input_dlmerged )
# remove .root extension if its there
if ".root" in basefilename and basefilename[-5:]==".root":
  basefilename = basefilename[:-5]

for ientry in range(start_entry,end_entry):

  ioll.go_to(ientry)
  iolcv.read_entry(ientry)

  # we save one entry per file for use in Pointcept
  entryfile_name = f"pointceptdata_{basefilename}_entry{ientry:06d}.h5"
  print(f"[{ientry}] output filename: ",entryfile_name)
  
  simchmaker.process( ioll, iolcv )
  #simchmaker._mckpmaker.printKeypoints()

  # save entry
  hdf_entry_prefix = f"/entry_0"  
  simchmaker.open_hdf_file( entryfile_name )
  simchmaker.save_entry( hdf_entry_prefix )
  simchmaker.close_hdf_file()

  if ientry+1>=run_nentries:
    break


simchmaker.close_hdf_file()
