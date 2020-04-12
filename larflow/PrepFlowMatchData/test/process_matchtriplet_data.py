from __future__ import print_function
import os,sys,argparse,time
from ctypes import c_int

parser = argparse.ArgumentParser(description='Run Prep LArFlow MatchTriplet Data')
parser.add_argument('-o','--out',required=True,type=str,help="Filename for LArCV output")
parser.add_argument("-mc","--has-mc",default=False,action="store_true",help="Has MC information")
parser.add_argument('input_larcv',nargs='+',help="Input larcv files")
parser.add_argument("-n","--num-events",default=None,type=int,help="Number of events to run")
parser.add_argument("-c","--config",type=str,default="prepmatchtriplet.cfg",help="Set configuration file.")


args = parser.parse_args(sys.argv[1:])

import ROOT as rt
from ROOT import std
from larcv import larcv
larcv.load_pyutil()
from ublarcvapp import ublarcvapp
from larflow import larflow

rt.gStyle.SetOptStat(0)

inputfiles = std.vector("string")()
for inputfile in args.input_larcv:
    inputfiles.push_back( inputfile )

driver = larcv.ProcessDriver("ProcessDriver")
driver.configure( args.config  )
driver.override_input_file( inputfiles )
driver.override_ana_file( args.out )

processors = driver.process_map()
tripletmaker = driver.process_ptr( processors.find("MatchTripletProcessor").second )

driver.initialize()

nentries = driver.io().get_n_entries()
if args.num_events is not None and nentries>args.num_events:
    nentries = args.num_events

for ientry in xrange(nentries):
    driver.process_entry(ientry)

    # for debug
    #break

driver.finalize()
