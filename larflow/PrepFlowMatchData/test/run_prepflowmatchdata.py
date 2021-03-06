from __future__ import print_function
import os,sys,argparse

parser = argparse.ArgumentParser(description='Run Prep LArFlow Match Data')
parser.add_argument('-olcv','--out-larcv',required=True,type=str,help="Filename for LArCV output")
parser.add_argument("-c","--config",required=True,type=str,help="Configuration file")
parser.add_argument('input_larcv',nargs='+',help="Input larcv files")

args = parser.parse_args(sys.argv[1:])

#print("inputfiles: ",args.input_larcv)
#print("type(inpufiles)=",type(args.input_larcv))

import ROOT as rt
from ROOT import std
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow

print(larflow.PrepFlowMatchData)

inputfiles = std.vector("std::string")()
if type(args.input_larcv) is str:
    inputfiles.push_back( args.input_larcv )
elif type(args.input_larcv) is list:
    for f in args.input_larcv:
        inputfiles.push_back( f )

driver = larcv.ProcessDriver("ProcessDriver")
driver.configure( args.config )
driver.override_input_file( inputfiles )
driver.override_output_file( args.out_larcv )

# get processor, add larlite files
processors = driver.process_map()
#it_process = processors.find("PrepFlowMatchData")
#prepmatchdata = driver.process_ptr(it_process.second)

driver.initialize()

nentries = driver.io().get_n_entries()

for ientry in xrange(0,nentries):
    driver.process_entry(ientry)
    break

driver.finalize()

