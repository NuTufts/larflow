import os,sys
import ROOT as rt

tfile = rt.TFile("extracted_tensorboard.root")
tfile.ls()

"""
  KEY: TGraphErrors	gtrain_loss;1	Graph
  KEY: TGraphErrors	gvalid_loss;1	Graph
  KEY: TGraphErrors	gtrain_acc_pos_examples;1	Graph
  KEY: TGraphErrors	gvalid_acc_pos_examples;1	Graph
  KEY: TGraphErrors	gtrain_acc_neg_examples;1	Graph
  KEY: TGraphErrors	gvalid_acc_neg_examples;1	Graph
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
gtrain_acc_pos_examples = tfile.Get("gtrain_acc_pos_examples")
gtrain_acc_neg_examples = tfile.Get("gtrain_acc_neg_examples")
gvalid_acc_pos_examples = tfile.Get("gvalid_acc_pos_examples")
gvalid_acc_neg_examples = tfile.Get("gvalid_acc_neg_examples")

for g in [gtrain_acc_pos_examples,gtrain_acc_neg_examples,gvalid_acc_pos_examples,gvalid_acc_neg_examples]:
    g.SetLineWidth(2)
    
gtrain_acc_pos_examples.SetLineColor(rt.kBlue+1)
gtrain_acc_neg_examples.SetLineColor(rt.kBlue-9)
#gtrain_acc_neg_examples.SetLineStyle(2)
gvalid_acc_pos_examples.SetLineColor(rt.kRed+1)
gvalid_acc_neg_examples.SetLineColor(rt.kRed-9)
#gvalid_acc_neg_examples.SetLineStyle(2)

gtrain_acc_pos_examples.Draw("ALX")
gtrain_acc_neg_examples.Draw("LX")
gvalid_acc_pos_examples.Draw("LX")
gvalid_acc_neg_examples.Draw("LX")

tacc = rt.TLegend(0.5,0.1,0.8,0.4)
tacc.AddEntry(gtrain_acc_pos_examples,"Pos. Examples (train)","L")
tacc.AddEntry(gtrain_acc_neg_examples,"Neg. Examples (train)","L")
tacc.AddEntry(gvalid_acc_pos_examples,"Pos. Examples (valid)","L")
tacc.AddEntry(gvalid_acc_neg_examples,"Neg. Examples (valid)","L")
tacc.Draw()

cacc.SetGridx(1)
cacc.SetGridy(1)
cacc.Update()

raw_input()
