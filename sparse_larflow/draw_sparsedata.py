import os,sys
import ROOT as rt
from larcv import larcv

rt.gStyle.SetOptStat(0)

io = larcv.IOManager(larcv.IOManager.kREAD)
io.add_in_file(sys.argv[1])
io.initialize()

for ientry in xrange(io.get_n_entries()):
    io.read_entry(ientry)
    ev_sparse = io.get_data(larcv.kProductSparseImage,sys.argv[2])
    sparseimg = ev_sparse.at(0)

    nfeatures = sparseimg.nfeatures()
    c = {}
    h = {}
    for ifeat in xrange(nfeatures):
        c[ifeat] = rt.TCanvas("c%d"%(ifeat),"Image %d"%(ifeat),1200,600)
        h[ifeat] = rt.TH2D("h%d"%(ifeat),"",3456,0,3456,1008,2400,2400+1008*6)

    nvalues = sparseimg.pixellist().size()
    npts    = nvalues/(2+nfeatures)
    for ipt in xrange(npts):
        start = ipt*(2+nfeatures)
        row = int(sparseimg.pixellist()[start])
        col = int(sparseimg.pixellist()[start+1])
        for ifeat in xrange(nfeatures):
            h[ifeat].SetBinContent( col+1, row+1,
                                    sparseimg.pixellist()[start+2+ifeat] )

    for ifeat in xrange(nfeatures):
        c[ifeat].cd()
        c[ifeat].Draw()
        h[ifeat].Draw("COLZ")
        c[ifeat].Update()
        #c[ifeat].SaveAs("test_sparseimg_img%d.pdf"%(ifeat))
    print "[ENTER] for next entry"
    raw_input()
