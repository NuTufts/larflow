import os,sys,time

import ROOT as rt
from ROOT import std
from larflow import larflow
from ctypes import c_int

# test file
infile = "out.root"


tree = rt.TChain("larmatchtriplet")
tree.Add(infile)

nentries = tree.GetEntries()

print "num of entries in the tree: ",nentries

io_start = time.time()

nfilled = c_int()
nmax_samples = 50000

for ientry in xrange(0,nentries):
    tree.GetEntry(ientry)
    sparse_v = [ tree.triplet_v[0].make_sparse_image( p ) for p in xrange(3) ]
    index_array = tree.triplet_v[0].sample_triplet_matches( nmax_samples, nfilled, True )
    print "[entry {}] nfilled={} index array returned {} for sparse image {}".format( ientry, nfilled.value,
                                                                                      index_array.shape,
                                                                                      [sparse_v[x].shape for x in xrange(3)] )

io_end = time.time()
print "Time to sample each event: ",(io_end-io_start)," secs total, ",(io_end-io_start)/float(nentries)," sec/event"

