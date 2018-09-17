import os,sys

# FILE LISTS COME FROM PUBS DB TOOLS
# specifically, pubs/dlleepubs/utils/dump_
# ------------------------------------------------------------------------

# using directory as shortcut for now
#inputdirs=["/cluster/kappa/90-days-archive/wongjiradlab/twongj01/data/larflow"]
inputdirs=["/cluster/kappa/90-days-archive/wongjiradlab/twongj01/llf/larflow/datasets/mcc8.4_bnbcosmics_p00/larcv/"]

os.system("mkdir inputlists")

joblist = open("joblist.txt",'w')

njobs = 0
for d in inputdirs:
    dirfiles = os.listdir(d)

    for f in dirfiles:
        path = d+"/"+f
        print path
        l = f.strip()
        i = l.split("_")
        run = int(i[-2])
        subrun = int(i[-1].split(".root")[0])
        jobid = 10000*run + subrun
        print >> joblist,jobid
        infile = open('inputlists/input_%d.txt'%(jobid),'w')
        # temp
        path = path.replace("/cluster/kappa/90-days-archive","/cluster/kappa")
        print >> infile,path.strip()
        infile.close()
        njobs+=1
print "number of jobs: ",njobs
joblist.close()

