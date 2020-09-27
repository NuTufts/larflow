import os,sys
import ROOT as rt

inputfile = sys.argv[1]
rfile = rt.TFile(inputfile)

tree = rfile.Get("KPSRecoManagerTree")

nentries = tree.GetEntries()
for ientry in xrange(nentries):
    tree.GetEntry(ientry)
    print "==== ENTRY[",ientry,"] ===="
    
    nvtxs = tree.track_truthreco_vtxinfo_v.size()
    for ivtx in xrange(nvtxs):
        vtxinfo = tree.track_truthreco_vtxinfo_v.at(ivtx)
        ntrks = vtxinfo.trackinfo_v.size()
        for itrack in xrange(0,ntrks):
            trackinfo = vtxinfo.trackinfo_v.at(itrack)
            print "vtx[",ivtx,"] track[",itrack,"]"
            print "  matched_true_trackid: ",trackinfo.matched_true_trackid
            print "  matched_true_pid: ",trackinfo.matched_true_pid
            print "  matched_mse: ",trackinfo.matched_mse
        print "------------------------------------------"
