import os,sys

# NUE
# files look like: larmatchdata_bnbnue_bnbnue_0490.root
outdir="outdir_mcc9_v13_bnbnue_corsika_kpreweight_wmatchlabel/"
N = 493


# BNB NU 
# files look like: larmatchdata_bnb_nu_bnbnue_0490.root
#outdir="outdir_mcc9_v13_bnb_nu_corsika"
#N = 573

completed = os.listdir(outdir)

fileids = []

for f in completed:
    fname = outdir+"/"+f.strip()
    fid = int(f.strip().split("_")[-1].split(".")[0])
    fileids.append(fid)
    print(fname)

fileids.sort()
missing = []
rerun_jobs = []
for i in range(N):
    if i not in fileids:
        missing.append(i)
        jobid = i//5
        if jobid not in rerun_jobs:
            rerun_jobs.append(jobid)
            
missing.sort()
rerun_jobs.sort()

print("missing fileids: ",)
for i in missing:
    print(i," ",)
print()

print("rerun jobs: ",)
zjobs = ""
for i in rerun_jobs:
    zjobs += "%d,"%(i)
print(zjobs)


