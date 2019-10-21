import os,sys

import ROOT as rt

rt.gStyle.SetOptStat(0)

tfile = rt.TFile("out_truthana.root")

herrflow_y2u = tfile.Get("herrflow_y2u")
herrflow_y2v = tfile.Get("herrflow_y2v")

hcdf_y2u = rt.TH1D("hcdf_y2u",";cm from truth;CDF",1000,0,1000*0.3)
hcdf_y2v = rt.TH1D("hcdf_y2v",";cm from truth;CDF",1000,0,1000*0.3)

tot_y2u = herrflow_y2u.Integral()
tot_y2v = herrflow_y2v.Integral()
for i in range(1000):
    hcdf_y2u.SetBinContent( i+1, herrflow_y2u.Integral(1,i+1)/tot_y2u )
    hcdf_y2v.SetBinContent( i+1, herrflow_y2v.Integral(1,i+1)/tot_y2v )

c = rt.TCanvas("c","c",1200,600)
c.Divide(2,1)
c.cd(1)
hcdf_y2u.Draw()
c.cd(2)
hcdf_y2v.Draw()

raw_input()



