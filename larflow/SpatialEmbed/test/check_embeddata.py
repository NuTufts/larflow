from __future__ import print_function
import os,sys,argparse,json
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Visuzalize Voxel Data")
parser.add_argument("input_file",type=str,help="file produced by 'prep_spatialembed.py'")
args = parser.parse_args()

import numpy as np
import ROOT as rt
from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
larcv.SetPyUtil()

# LOAD TREES
infile = rt.TFile(args.input_file)
io = infile.Get("s3dembed")
nentries = io.GetEntries()
print("NENTRIES: ",nentries)

voxelloader = larflow.spatialembed.Prep3DSpatialEmbed()
voxelloader.loadTreeBranches( io )
    
for ientry in range(nentries):
    data = voxelloader.getTreeEntry(ientry)
    print("number of voxels: ",data.size())
    data_dict = voxelloader.getTreeEntryDataAsArray(ientry)
    print("voxel entries: ",data_dict["coord_t"].shape)
print("[FIN]")
