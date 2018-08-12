import os,sys
import ROOT as rt
from larlite import larlite

#tmp = open("trueflow2",'r')
tmp = open("temp2",'r')
ll = tmp.readlines()

pts = []
border_pts = []

for l in ll:
    l = l.strip()
    info = l.split("pos=(")[1][:-1].strip()
    info = info.split(",")
    x = float(info[0])
    y = float(info[1])
    z = float(info[2])
    pts.append( (z,x,y) )
    if y>100.0 or y<-100.0 or z<20.0 or z>1000.0:
        border_pts.append( (z,x,y) )

c = rt.TCanvas("c","c",1200,600)

g = rt.TGraph2D( len(pts) )
b = rt.TGraph2D( len(border_pts) )

for i,pt in enumerate(pts):
    g.SetPoint(i,pt[0],pt[1],pt[2])
for i,pt in enumerate(border_pts):
    b.SetPoint(i,pt[0],pt[1],pt[2])
b.SetMarkerColor(rt.kRed)

g.Draw("colp")
b.Draw("colpsame")

c.Update()
c.Draw()

raw_input()
