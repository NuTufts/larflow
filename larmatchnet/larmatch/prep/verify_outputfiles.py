import os,sys
import ROOT as rt
from larflow import larflow

outdir="outdir_mcc9_v13_bnbnue_corsika_kpreweight_wmatchlabel/"
#outdir="outdir_mcc9_v13_bnb_nu_corsika_kpreweight_wmatchlabel/"

badlist = []
outfiles = os.listdir(outdir)
for f in outfiles:
    #print(f)
    c = rt.TChain("larmatchtrainingdata")
    c.AddFile(outdir+'/'+f)
    nentries = c.GetEntries()
    if nentries!=100:
        print("bad file: ",f," nentries=",nentries)
        badlist.append(outdir+"/"+f)

print("BADLIST")
for f in badlist:
    print(f)
print("")

badstr=""
for b in badlist:
    bnum = int(b.split("_")[-1].split(".")[0])
    badstr += "%d,"%(bnum)
print(badstr)
