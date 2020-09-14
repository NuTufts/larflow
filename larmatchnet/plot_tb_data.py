import os,sys
import ROOT as rt

tfile = rt.TFile("extracted_tensorboard_train_kpspaf.root")
tfile.ls()

"""
  KEY: TGraphErrors	gtrain_loss;1	Graph
  KEY: TGraphErrors	gvalid_loss;1	Graph

  KEY: TGraphErrors	gtrain_acc_lm_all;1	Graph
  KEY: TGraphErrors	gvalid_acc_lm_all;1	Graph
  KEY: TGraphErrors	gtrain_acc_kp_nu;1	Graph
  KEY: TGraphErrors	gvalid_acc_kp_nu;1	Graph
  KEY: TGraphErrors	gtrain_acc_kp_trk;1	Graph
  KEY: TGraphErrors	gvalid_acc_kp_trk;1	Graph
  KEY: TGraphErrors	gtrain_acc_kp_shr;1	Graph
  KEY: TGraphErrors	gvalid_acc_kp_shr;1	Graph
  KEY: TGraphErrors	gtrain_acc_paf;1	Graph
  KEY: TGraphErrors	gvalid_acc_paf;1	Graph

"""

c = rt.TCanvas("closs","closs",600,400)
c.Draw()

gtrain_loss = tfile.Get("gtrain_loss")
gvalid_loss = tfile.Get("gvalid_loss")

gtrain_loss.SetLineColor(rt.kBlue)
gtrain_loss.SetFillColor(rt.kBlue+2)
gtrain_loss.SetFillStyle(3003)
gtrain_loss.Draw("A3")
gtrain_loss.Draw("LX")
gtrain_loss.SetTitle(";iterations;loss")

gvalid_loss.SetLineColor(rt.kRed)
gvalid_loss.SetLineWidth(2)
#gvalid_loss.SetFillStyle(3003)
#gvalid_loss.Draw("3")
gvalid_loss.Draw("LX")

tloss = rt.TLegend(0.5,0.5,0.8,0.8)
tloss.AddEntry(gtrain_loss,"Training","L")
tloss.AddEntry(gvalid_loss,"Validation","L")
c.SetGridy(1)
c.SetGridx(1)
tloss.Draw()
c.Update()

cacc = rt.TCanvas("cacc","cacc",600,400)
gacc = {}
acc_color = {"lm_all":rt.kRed,
             "kp_nu":rt.kBlue,
             "kp_trk":rt.kGreen-8,
             "kp_shr":rt.kMagenta,
             "paf":rt.kBlack}
acc_name = {"lm_all":"LArMatch",
            "kp_nu":"Keypoint-#nu_{e}",
            "kp_trk":"Keypoint-Track",
            "kp_shr":"Keypoint-Shower",
            "paf":"Flow Field, cos>0.94"}

for s in ["train","valid"]:
    for n in ["lm_all","kp_nu","kp_trk","kp_shr","paf"]:
        gacc[(s,n)] = tfile.Get("g%s_acc_%s"%(s,n))
        gacc[(s,n)].SetLineWidth(2)
        if "train"==s:
            gacc[(s,n)].SetLineStyle(2)
        gacc[(s,n)].SetLineColor( acc_color[n] )

tacc = rt.TLegend(0.5,0.1,0.8,0.4)        
gacc[("train","paf")].Draw("ALX")
gacc[("train","paf")].SetTitle(";iterations;accuracy")
gacc[("train","paf")].GetYaxis().SetRangeUser(0,1.0)
for s in ["valid"]:    
    for n in ["lm_all","kp_nu","kp_trk","kp_shr","paf"]:
        gacc[(s,n)].Draw("LX")
        tacc.AddEntry( gacc[(s,n)], acc_name[n], "L" )

tacc.Draw()

cacc.SetGridx(1)
cacc.SetGridy(1)
cacc.Update()

raw_input()
