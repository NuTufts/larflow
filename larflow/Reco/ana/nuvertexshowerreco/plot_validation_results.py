import ROOT as rt

rt.gStyle.SetOptStat(0)

rfile = rt.TFile("xgb_validout_v1.4.0_v2.root")
modelout = rfile.Get("modelout")


# cut sets
cutdef = {"energybin10":"recofragment_pixsum<10.0",
          "energybin20":"recofragment_pixsum>=10.0 && recofragment_pixsum<20.0",
          "energybin30":"recofragment_pixsum>=20.0 && recofragment_pixsum<30.0",
          "energybin40":"recofragment_pixsum>=30.0 && recofragment_pixsum<40.0",
          "energybin50":"recofragment_pixsum>=40.0 && recofragment_pixsum<50.0",
          "energybin100":"recofragment_pixsum>=50.0 && recofragment_pixsum<100.0",          
          "energybinhigh":"recofragment_pixsum>=100.0",
          "distbin20":"recofragment_dist2vtx<20.0",
          "distbin50":"recofragment_dist2vtx>=20.0 && recofragment_dist2vtx<50.0",
          "distbin100":"recofragment_dist2vtx>=50.0 && recofragment_dist2vtx<100.0",
          "distbin100":"recofragment_dist2vtx>=100.0 && recofragment_dist2vtx<150.0",
          "distbin150":"recofragment_dist2vtx>=100.0 && recofragment_dist2vtx<150.0",
          "distbin200":"recofragment_dist2vtx>=150.0 && recofragment_dist2vtx<200.0",
          "distbinhigh":"recofragment_dist2vtx>=200.0"}


# for each cut, make score hist for groundtruth=0 and 1
nbins=100
c_v = {}
hist_v = {}
for cdef in cutdef:
    c = rt.TCanvas("c%s"%(cdef),"Validation Results: %s"%(cdef),800,600)
    hname_true="hscore_%s_true"%(cdef)
    hname_false="hscore_%s_false"%(cdef)

    htrue  = rt.TH1D( hname_true,"",nbins,-20.0,10)
    hfalse = rt.TH1D( hname_false,"",nbins,-20.0,10)    

    modelout.Draw("-TMath::Log(1.0/score-1.0)>>%s"%(hname_false),"label==0 && "+cutdef[cdef])    
    modelout.Draw("-TMath::Log(1.0/score-1.0)>>%s"%(hname_true),"label==1 && "+cutdef[cdef],"same")

    htrue.SetLineColor(rt.kRed)
    hist_v[hname_true] = htrue
    hist_v[hname_false] = hfalse
    c_v[cdef] = c
    c.Update()

input()



