from __future__ import print_function
import os,sys

import ROOT as rt
from larcv import larcv

from ROOT import std

flowmatch_ana   = "ana_flowmatch_data.root"
flowmatch_larcv = "test.root"

io = larcv.IOManager(larcv.IOManager.kREAD,"larcv")
io.add_in_file( flowmatch_larcv )
io.initialize()

flowmatch_v = {}
flowchain_v = {}
input_ana_files = [ flowmatch_ana ]
for plane in [0,1,2]:
    flowmatch_v[plane] = std.vector("larflow::FlowMatchMap")()
    flowchain_v[plane] = rt.TChain("flowmatchdata_plane%d"%(plane))
    for fin in input_ana_files:
        print("adding ana file: ",fin)
        flowchain_v[plane].Add( fin )
    flowchain_v[plane].SetBranchAddress( "matchmap", rt.AddressOf( flowmatch_v[plane] ) )
    print("chain plane[",plane,"] has ",flowchain_v[plane].GetEntries()," entries")

nentries = io.get_n_entries()

for ientry in range(nentries):

    spimg_v = {}    
    for p in range(3):
        flowchain_v[p].GetEntry(ientry)
        spimg_v[p] = io.get_data( larcv.kProductSparseImage, "larflow_plane%d"%(p) ).SparseImageArray()
