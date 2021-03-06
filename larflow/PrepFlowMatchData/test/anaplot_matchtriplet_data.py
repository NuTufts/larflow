import os,sys
import ROOT as rt
from larflow import larflow

rt.gStyle.SetOptStat(0)

infile = "out.root"

tfile = rt.TFile(infile,"read")
tfile.ls()

tree = tfile.Get("larmatchtriplet")
nentries = tree.GetEntries()

print "Number of entries: ",nentries
PLOT_TRIAREA = False

for ientry in xrange(0,nentries):
    tree.GetEntry(ientry)
    print "[ENTRY ",ientry,"]"
    
    # plot: ADC vs. possible
    c = rt.TCanvas("c","c",1800,1200)
    c.Divide(3,2)

    
    th2d_adc_v   = tree.triplet_v.front().plot_sparse_images("entry%d"%(ientry))
    th2d_truth_v = tree.triplet_v.front().plot_truth_images("entry%d"%(ientry))
    for p in xrange(3):
        c.cd(p+1)
        th2d_adc_v[p].Draw("colz")
        if p in [0,1]:
            th2d_adc_v[p].GetXaxis().SetRangeUser(0,2400)
        c.cd(p+4)
        th2d_truth_v[p].Draw("colz")
        if p in [0,1]:
            th2d_truth_v[p].GetXaxis().SetRangeUser(0,2400)

    c.Update()

    if PLOT_TRIAREA:
        ctri = rt.TCanvas("ctri","ctri",600,400)
        htri = rt.TH1D("htri_entry%d"%(ientry),"",100,0,1)
        for i in xrange(tree.triplet_v.front()._triarea_v.size()):
            htri.Fill( tree.triplet_v.front()._triarea_v[i] )
        htri.Draw()
        ctri.Update()

    print "[ENTER]"
    raw_input()
