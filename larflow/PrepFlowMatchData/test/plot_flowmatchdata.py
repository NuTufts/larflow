from __future__ import print_function
import os,sys

import ROOT as rt
from larcv import larcv

from ROOT import std

rt.gStyle.SetOptStat(0)

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

    c = rt.TCanvas("c","c",1200,1200)
    c.Divide(3,3)
    
    spimg_v = {}
    hsrc_v = {0:[],1:[],2:[]}
    htar1_v = {0:[],1:[],2:[]}
    htar2_v = {0:[],1:[],2:[]}        
    for p in range(3):
        # for each plane, we plot the source and target pixels
        flowchain_v[p].GetEntry(ientry)
        spimg_v[p] = io.get_data( larcv.kProductSparseImage, "larflow_plane%d"%(p) ).SparseImageArray()
        #adc = io.get_data( larcv.kProductImage2D, "wire" ).Image2DArray().at(p)

        srcimg =spimg_v[p].at(0)
        matchdata1 = flowmatch_v[p].at(0)
        matchdata2 = flowmatch_v[p].at(1)
        
        # make a histogram
        nwires = 3456
        #if p<2:
        #    nwires = 2400
        hsrc  = rt.TH2D("hsrc_plane%d_entry%d"%(p,ientry), "",nwires,0,nwires,1008,2400,8448)
        htar1 = rt.TH2D("htar1_plane%d_entry%d"%(p,ientry),"",nwires,0,nwires,1008,2400,8448)
        htar2 = rt.TH2D("htar2_plane%d_entry%d"%(p,ientry),"",nwires,0,nwires,1008,2400,8448) 
        for idx in range(srcimg.len()):

            row = srcimg.getfeature(idx,0)
            col = srcimg.getfeature(idx,1)
            hsrc.SetBinContent( int(col)+1, int(row)+1, float(srcimg.getfeature(idx,2)) )

            tarindices1 = matchdata1.getTargetIndices( idx )
            tarindices2 = matchdata2.getTargetIndices( idx )

            truthindices1 = matchdata1.getTruthVector( idx )
            truthindices2 = matchdata2.getTruthVector( idx )

            # loop through target indices and label image
            for tidx in range(tarindices1.size()):
                tarcol = spimg_v[p][1].getfeature( tarindices1[tidx], 1 )
                tarval = 1.0+truthindices1[tidx]
                if tarval>htar1.GetBinContent( int(tarcol)+1, int(row)+1 ):
                    htar1.SetBinContent( int(tarcol)+1, int(row)+1, tarval )

            for tidx in range(tarindices2.size()):
                tarcol = spimg_v[p][2].getfeature( tarindices2[tidx], 1 )
                tarval = 1.0+truthindices2[tidx]
                if tarval>htar2.GetBinContent( int(tarcol)+1, int(row)+1 ):
                    htar2.SetBinContent( int(tarcol)+1, int(row)+1, tarval )
                    

        c.cd( 3*p+1 )
        hsrc.Draw("colz")
        hsrc_v[p].append(hsrc)
        c.cd( 3*p+2 )
        htar1.Draw("colz")
        htar1_v[p].append( htar1 )
        c.cd( 3*p+3 )
        htar2.Draw("colz")
        htar2_v[p].append( htar2 )

    c.Draw()
    c.Update()
    raw_input()

            
