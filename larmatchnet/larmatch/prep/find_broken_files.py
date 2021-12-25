import os,sys
import ROOT as rt

# NUE
# files look like: larmatchdata_bnbnue_bnbnue_0490.root
#outdir="outdir_mcc9_v13_bnbnue_corsika"
#N = 493


# BNB NU 
# files look like: larmatchdata_bnb_nu_bnbnue_0490.root
outdir="outdir_mcc9_v13_bnb_nu_corsika"
N = 573

completed = os.listdir(outdir)

fileids = []
filedict = {}

for f in completed:
    fname = outdir+"/"+f.strip()
    fid = int(f.strip().split("_")[-1].split(".")[0])
    fileids.append(fid)
    filedict[fid] = fname
    
print("Number of files processed: %d/%d"%(len(fileids),N))

fileids.sort()
broken_ids = []
rerun_jobs = []
for i in fileids:
    jobid = i//5
    rf = rt.TFile(filedict[i],'open')
    fileok = True
    t = rf.Get("larmatchtrainingdata")
    n = t.GetEntries()
    print("[%d] num events =  %d"%(jobid,n))
    if n<100 and jobid not in rerun_jobs:        
        rerun_jobs.append(jobid)
        broken_ids.append(i)

broken_ids.sort()
rerun_jobs.sort()

zbroken = ""
zjobid = ""
for i in broken_ids:
    # need to get the line number
    gpipe = os.popen( "grep -n %03d inputlists/mcc9_v13_bnb_nu_corsika.triplettruth.list"%(i))
    glines = gpipe.readlines()
    x = glines[0]
    lineno = int(x.split(":")[0])-1
    zbroken += "%d,"%(i)
    zjobid += "%d,"%(lineno)
print("broken file ids: ",zbroken)
print("single job ids: ",zjobid)


print("rerun jobs: ",)
zjobs = ""
for i in rerun_jobs:
    zjobs += "%d,"%(i)
print(zjobs)


