import os,sys
import ROOT as rt

tfile = rt.TFile("extracted_tensorboard.root")
tfile.ls()

"""
  KEY: TGraphErrors	gtrain_loss_kp;1	Graph
  KEY: TGraphErrors	gvalid_loss_kp;1	Graph
  KEY: TGraphErrors	gtrain_loss_lm;1	Graph
  KEY: TGraphErrors	gvalid_loss_lm;1	Graph

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

# =========================
# LOSSES
# ------
c = rt.TCanvas("closs","closs",1600,800)
c.Draw()
tloss = rt.TLegend(0.5,0.5,0.8,0.8)
isfirst=True
for lossname,losscolor,losstitle in [("kp",rt.kRed,"Keypoint"),("lm",rt.kBlue+3,"Larmatch")]:    
    gtrain_loss = tfile.Get("gtrain_loss_"+lossname)
    gvalid_loss = tfile.Get("gvalid_loss_"+lossname)

    gtrain_loss.SetLineColor(losscolor-2)
    gtrain_loss.SetLineWidth(2)
    gtrain_loss.SetFillColor(losscolor+2)
    gtrain_loss.SetFillStyle(3003)
    if isfirst:
        gtrain_loss.Draw("AL3")
        isfirst=False
    
    gtrain_loss.Draw("LX")
    gtrain_loss.SetTitle(";iterations;loss")

    gvalid_loss.SetLineColor(losscolor)
    gvalid_loss.SetLineWidth(4)
    gvalid_loss.SetLineStyle(9)
    gvalid_loss.SetFillStyle(3003)
    #gvalid_loss.Draw("3")
    gvalid_loss.Draw("LX3")

    tloss.AddEntry(gtrain_loss,losstitle+" Training","L")
    tloss.AddEntry(gvalid_loss,losstitle+" Validation","L")

    
c.SetGridy(1)
c.SetGridx(1)
tloss.Draw()
c.Update()
print("Loss plot done.")

#======================================
# KEYPOINT ACC
cacc = rt.TCanvas("cacc","cacc",1600,800)
gacc = {}
kp_classes = ["kp_nu","kp_trackstart","kp_trackend","kp_shower","kp_michel","kp_delta"]
kp_acc_color = {"kp_nu":rt.kRed,
                "kp_trackstart":rt.kBlue,
                "kp_trackend":rt.kOrange,
                "kp_shower":rt.kMagenta,
                "kp_michel":rt.kCyan,
                "kp_delta":rt.kGreen-8}
kp_acc_name = {"kp_nu":"Keypoint-#nu",
               "kp_trackstart":"Keypoint-Track Start",
               "kp_trackend":"Keypoint-Track Start",               
               "kp_shower":"Keypoint-Primary-Shower",
               "kp_michel":"Keypoint-Michel",
               "kp_delta":"Keyopint-Delta"}
               
for s in ["train","valid"]:
    for n in kp_classes:
        gacc[(s,n)] = tfile.Get("g%s_kp_acc_%s"%(s,n))
        if "train"==s:
            gacc[(s,n)].SetLineWidth(1)
            gacc[(s,n)].SetLineColor( kp_acc_color[n]-2 )
        else:
            gacc[(s,n)].SetLineWidth(4)
            gacc[(s,n)].SetLineStyle(9)            
            gacc[(s,n)].SetLineColor( kp_acc_color[n] )

tacc = rt.TLegend(0.5,0.1,0.8,0.4)        
gacc[("train","kp_nu")].Draw("ALX")
gacc[("train","kp_nu")].SetTitle(";iterations;accuracy")
gacc[("train","kp_nu")].GetYaxis().SetRangeUser(0,1.0)
for n in kp_classes:
    for s in ["train","valid"]:        
        gacc[(s,n)].Draw("LX")
        tacc.AddEntry( gacc[(s,n)], kp_acc_name[n]+", "+s, "L" )

tacc.Draw()

cacc.SetGridx(1)
cacc.SetGridy(1)
cacc.Update()

#======================================
# LARMATCH ACC
clmacc = rt.TCanvas("clmacc","clmacc",1600,800)
glmacc = {}
lm_classes = ["lm_all","lm_neg","lm_pos"]
lm_acc_color = {"lm_all":rt.kBlack,
                "lm_neg":rt.kBlue,
                "lm_pos":rt.kRed}
lm_acc_name = {"lm_all":"Larmatch: all proposals",
               "lm_pos":"Larmatch: true proposals",
               "lm_neg":"Larmatch: false (ghost) proposals"}
               
for s in ["train","valid"]:
    for n in lm_classes:
        glmacc[(s,n)] = tfile.Get("g%s_lm_acc_%s"%(s,n))
        if "train"==s:
            glmacc[(s,n)].SetLineWidth(1)
            glmacc[(s,n)].SetLineColor( lm_acc_color[n]-2 )
        else:
            glmacc[(s,n)].SetLineWidth(4)
            glmacc[(s,n)].SetLineStyle(9)            
            glmacc[(s,n)].SetLineColor( lm_acc_color[n] )

tlmacc = rt.TLegend(0.5,0.1,0.8,0.4)        
glmacc[("train","lm_all")].Draw("ALX")
glmacc[("train","lm_all")].SetTitle(";iterations;accuracy")
glmacc[("train","lm_all")].GetYaxis().SetRangeUser(0,1.0)
for n in lm_classes:
    for s in ["train","valid"]:        
        glmacc[(s,n)].Draw("LX")
        tlmacc.AddEntry( glmacc[(s,n)], lm_acc_name[n]+", "+s, "L" )

tlmacc.Draw()

clmacc.SetGridx(1)
clmacc.SetGridy(1)
clmacc.Update()

input()
