import os,sys
import argparse

parser = argparse.ArgumentParser(description='Run the DL tagger')
parser.add_argument('-olcv','--out-larcv',required=True,type=str,help="Filename for LArCV output")
parser.add_argument('-oll', '--out-larlite',required=True,type=str,help="Filename for LArlite output")
parser.add_argument("-c","--config",required=True,type=str,help="Configuration file")
parser.add_argument('-op','--opreco',required=True,type=str,help='Input larlite opreco file')
parser.add_argument("-mc",'--mcinfo',required=False,type=str,help='Input larcvtruth file')
parser.add_argument('input_larcv',type=str,nargs='+',help="Input larcv files")

args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from ROOT import std
from larcv import larcv
from ublarcvapp import ublarcvapp
factory = ublarcvapp.dltagger.DLTaggerProcessFactory() # load the library, which registers the factory
print factory

inputfiles = std.vector("std::string")()
if type(args.input_larcv) is str:
    #inputfiles.push_back( "testset1/out_larcv_test.root" )
    inputfiles.push_back( args.input_larcv )
elif type(args.input_larcv) is list:
    for f in args.input_larcv:
        inputfiles.push_back( f )

driver = larcv.ProcessDriver("DLTagger")
driver.configure( args.config )
driver.override_input_file( inputfiles )
driver.override_output_file( args.out_larcv )

# get processor, add larlite files
processors = driver.process_map()
it_process = processors.find("DLTaggerProcess")
dltagger = driver.process_ptr(it_process.second)
dltagger.add_larlite_infile( args.opreco )
dltagger.set_larlite_outfile( args.out_larlite )

driver.initialize()

nentries = driver.io().get_n_entries()

for ientry in xrange(0,nentries):
    driver.process_entry(ientry)
    if ientry==1:
        break
#driver.process_entry(11)

driver.finalize()
