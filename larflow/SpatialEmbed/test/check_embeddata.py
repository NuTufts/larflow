from __future__ import print_function
import os,sys,argparse,json
from ctypes import c_int
from math import log

parser = argparse.ArgumentParser("Check embedded data")
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
input_files = rt.std.vector("std::string")()
input_files.push_back(args.input_file)

nentries = 20
batchsize = 5

voxelloader = larflow.spatialembed.Prep3DSpatialEmbed(input_files)
    
for ientry in range(nentries):
    #data = voxelloader.getTreeEntry(ientry)
    #print("number of voxels: ",data.size())
    #data_dict = voxelloader.getNextTreeEntryDataAsArray()
    data_dict = voxelloader.getTrainingDataBatch(batchsize)
    if data_dict:
        print("entry[",ientry,"] voxel entries: ",data_dict["coord_t"].shape)
        for tname,tarr in data_dict.items():
            print("   ",tname," nan=",np.isnan(tarr).sum()," inf=",np.isinf(tarr).sum()," min=",np.min(tarr)," max=",np.max(tarr))
    else:
        print("returned None")
        break

print("[FIN]")
