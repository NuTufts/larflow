# run DataLoader code on root file

from __future__ import print_function
import os,sys,argparse
from array import *

parser = argparse.ArgumentParser("Test DataLoader")
parser.add_argument("input_file",type=str,help="input root file [required]")
parser.add_argument("entry",type=int,help="entry # [required]")
#parser.add_argument("-in", "--input-file",required=True,type=str,help="input root file [required]")
#parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
# maybe add a line for num entries
#parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from larflow import larflow
#larcv.SetPyUtil()

rt.gStyle.SetOptStat(0)

#ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll.add_in_filename(  args.input_larlite )
#ioll.open()

# load tree
tfile = rt.TFile(args.input_file,'open')
preppedTree  = tfile.Get('preppedTree')
print("Got tree")

nentries = preppedTree.GetEntries()
print("NENTRIES: ",nentries)

ientry = args.entry
print("Entry requested is: ",ientry)

input_files = rt.std.vector("std::string")()
input_files.push_back(args.input_file)

dataloader = larflow.lightmodel.DataLoader(input_files)
dataloader.load_entry(ientry)

#for ientry in range(nentries):
#    dataloader.load_entry( ientry )
#    dataloader.make_arrays()

#arr = []
data_dict = dataloader.make_arrays()
print("data_dict['flash_info']: ", data_dict["flash_info"])
print("shape:", data_dict["flash_info"].shape)
#print(data_dict.items())
print("data_dict['charge_array']: ", data_dict["charge_array"])
print("shape:", data_dict["charge_array"].shape)

print("data_dict['coord_array']: ", data_dict["coord_array"])
print("shape:", data_dict["coord_array"].shape)

#arr = data_dict["coord_array"].reshape((10,4))

#print("arr: ",arr)
#print(arr.shape)

#arr = dataloader.make_arrays()

#print(arr.shape)

print("== FIN ==")
