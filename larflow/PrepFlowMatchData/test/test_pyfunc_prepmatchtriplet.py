import os,sys,time

import ROOT as rt
from ROOT import std
from larcv import larcv
larcv.load_rootutil()
from larflow import larflow
from ctypes import c_int

# test file
infile = "out.root"
PLOT = True
if PLOT:
    rt.gStyle.SetOptStat(0)

tree = rt.TChain("larmatchtriplet")
tree.Add(infile)

nentries = tree.GetEntries()

print "num of entries in the tree: ",nentries

nfilled = c_int()
nmax_samples = 50000

dt_load = 0.0

for ientry in xrange(0,nentries):

    io_start = time.time()    
    tree.GetEntry(ientry)
    sparse_v = [ tree.triplet_v[0].make_sparse_image( p ) for p in xrange(3) ]
    index_array = tree.triplet_v[0].sample_triplet_matches( nmax_samples, nfilled, True )
    print "[entry {}] nfilled={} index array returned {} for sparse image {}".format( ientry, nfilled.value,
                                                                                      index_array.shape,
                                                                                      [sparse_v[x].shape for x in xrange(3)] )
    dt_load += (time.time()-io_start)

    if PLOT:

        hist_v = [ rt.TH2D( "htest_entry%d_p%d"%(ientry,p), "", 3456, 0, 3456, 1008, 2400, 2400+6*1008) for p in xrange(3) ]
        for ipt in xrange(index_array.shape[0]):
            for p in xrange(3):
                idx = index_array[ipt,p]
                row = int(sparse_v[p][idx,0])
                col = int(sparse_v[p][idx,1])
                adc = float(sparse_v[p][idx,2])
                hist_v[p].SetBinContent( col+1, row+1, adc ) 
        
        c = rt.TCanvas("c","c",800,1500)
        c.Divide(1,3)
        for p in xrange(3):
            c.cd(p+1)
            hist_v[p].Draw("colz")
            if p in [0,1]:
                hist_v[p].GetXaxis().SetRangeUser(0,2400)
        c.Update()
        print "[plotted output] ENTER to continue."
        raw_input()

io_end = time.time()
print "Time to sample each event: ",dt_load," secs total, ",(dt_load)/float(nentries)," sec/event"

