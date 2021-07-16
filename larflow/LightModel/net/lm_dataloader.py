# runs DataLoader code on root file

from __future__ import print_function
import os,sys,argparse
from array import *

'''
parser = argparse.ArgumentParser("Test DataLoader")
parser.add_argument("input_file",type=str,help="input root file [required]")
parser.add_argument("entry",type=int,help="entry # [required]")
#parser.add_argument("-in", "--input-file",required=True,type=str,help="input root file [required]")
#parser.add_argument("-o","--output",required=True,type=str,help="output file name [required]")
# maybe add a line for num entries
#parser.add_argument("-n",   "--nentries",type=int,default=-1,help="Number of entries to run [default: -1 (all)]")
args = parser.parse_args()
'''

import numpy as np
np.set_printoptions(threshold=np.inf) # to print out full array
import ROOT as rt
from larlite import larlite
from larcv import larcv
from larflow import larflow
#larcv.SetPyUtil()

import torch
#from torch.utils import data as torchdata

rt.gStyle.SetOptStat(0)

#ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll.add_in_filename(  args.input_larlite )
#ioll.open()

def load_lm_data(input_file, entry):


    # load tree
    tfile = rt.TFile(input_file,'open')
    preppedTree  = tfile.Get('preppedTree')
    print("Got tree")

    ientry = entry
    print("Entry requested is: ",ientry)

    input_files = rt.std.vector("std::string")()
    input_files.push_back(input_file)

    # will loop through to create a batch this many times 
    nentries = 1 # how many batches
    batchsize = 1 # how many inside a batch    

    dataloader = larflow.lightmodel.DataLoader(input_files)
    dataloader.load_entry(ientry)
    
    for ientry in range(nentries):
    #dataloader.load_entry( ientry )
    #dataloader.make_arrays()

        print("Starting batch number",ientry)
        data_dict = dataloader.getTrainingDataBatch(batchsize)
        print("hi2")
        if data_dict:
            print("entry[",ientry,"] voxel entries: ",data_dict["coord_t"].shape)
            print("data_dict['coord_t']: ", data_dict["coord_t"])
            print("data_dict['feat_t']: ", data_dict["feat_t"])
            print("shape:", data_dict["feat_t"].shape)
            print("data_dict['flash_t']: ", data_dict["flash_t"])
            print("shape:", data_dict["flash_t"].shape)

    # make into torch tensors
    coord_t = torch.from_numpy(np.array(data_dict["coord_t"]))
    feat_t = torch.from_numpy(np.array(data_dict["feat_t"]))
    flash_t = torch.from_numpy(np.array(data_dict["flash_t"]))
            
    data = {"coord_t":coord_t, "feat_t":feat_t, "flash_t":flash_t}
    
    return data



#nentries = preppedTree.GetEntries()
#print("NENTRIES: ",nentries)
       
# NOTE: The below was all for 1 entry
#data_dict = dataloader.make_arrays()
#print("data_dict['flash_info']: ", data_dict["flash_info"])
#print("shape:", data_dict["flash_info"].shape)
#print(data_dict.items())
#print("data_dict['charge_array']: ", data_dict["charge_array"])
#print("shape:", data_dict["charge_array"].shape)

#print("data_dict['coord_array']: ", data_dict["coord_array"])
#print("shape:", data_dict["coord_array"].shape)

#print("Now that we have our tensors, feed into network!")

#print("== FIN ==")

if __name__=="__main__":

    # test
    print("hi")
    input_file = "../../Ana/CRTPreppedTree_crttrack_b40ad76a-1eb4-4ab0-8bf5-afbf194f216f-jobid0035.root"
    entry = 2
    
    load_lm_data(input_file, entry)
