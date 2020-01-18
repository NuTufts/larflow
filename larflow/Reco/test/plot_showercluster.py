import os,json
import ROOT as rt

f = open("showercluster.json",'r')
j = json.load(f)

graphs_v = []

c = rt.TCanvas("c","c",1400,600)
c.Divide(2,1)
c.Draw()

for jshower in j["shower"]:
    g = rt.TGraph( len(jshower["radius"]) )
    for i,r in enumerate(jshower["radius"]):
        g.SetPoint(i,float(jshower["s"][i]),float(jshower["radius"][i]))

    c.cd(2)
    g.Draw("ALP")
    c.Update()

    c.cd(1)
    g3d = rt.TGraph2D( len(jshower["points"]) )
    g3d.SetMarkerStyle(20)
    for i in xrange( len(jshower["points"]) ):
        g3d.SetPoint( i, jshower["points"][i][0], jshower["points"][i][1], jshower["points"][i][2] )
    g3d.Draw("P")

    pca = rt.TGraph2D(3)
    pca.SetLineColor(rt.kRed)
    pca.SetMarkerStyle(20)
    for i in xrange(3):
        pca.SetPoint(i,jshower["pca"][i][0], jshower["pca"][i][1], jshower["pca"][i][2] )
    pca.Draw("LINE SAME")
    
    c.Update()

    print "[enter]"
    raw_input()

    


