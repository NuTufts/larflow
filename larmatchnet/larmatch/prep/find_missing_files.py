import os,sys

# NUE
# files look like: larmatchdata_bnbnue_bnbnue_0490.root
#samplename="mcc9_v13_bnbnue_corsika_training"
#outdirs=["/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnbnue_corsika_training"]
#inputlist="../../dataprep/inputlists/mcc9_v13_bnbnue_corsika_training.paired.list"

#samplename="mcc9_v13_bnbnue_corsika_validation"
#outdirs=["/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnbnue_corsika_validation"]
#inputlist="../../dataprep/inputlists/mcc9_v13_bnbnue_corsika_validation.paired.list"


# BNB NU 
# files look like: larmatchdata_bnb_nu_bnbnue_0490.root
#samplename = "mcc9_v13_bnb_nu_corsika_training"
#outdirs=["/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnb_nu_corsika_training"]
#inputlist="../../dataprep/inputlists/mcc9_v13_bnb_nu_corsika_training.paired.list"

samplename = "mcc9_v13_bnb_nu_corsika_validation"
outdirs=["/cluster/tufts/wongjiradlabnu/twongj01/gen2/photon_analysis/ubdl/larflow/larmatchnet/larmatch/prep/outdir_mcc9_v13_bnb_nu_corsika_validation"]
inputlist="../../dataprep/inputlists/mcc9_v13_bnb_nu_corsika_validation.paired.list"

# get the source list
file_hashes = {}
file_pairs = {}
with open(inputlist,'r') as f:
    l = f.readlines()
    for ll in l:
        ll = ll.strip()
        if "bnb_nu" in samplename:
            # files look like: [dir]/larcv_mctruth_25f3f139-3af4-4822-8308-e95a10724466.root
            x = os.path.basename(ll.split()[0]) # larcv_mctruth_25f3f139-3af4-4822-8308-e95a10724466.root
            x = x.split(".")[0] # larcv_mctruth_25f3f139-3af4-4822-8308-e95a10724466
            x = x.split("_")[-1] # 25f3f139-3af4-4822-8308-e95a10724466
        else:
            # files looks like: [dir]/larcvtruth-Run000001-SubRun001869.root
            x = os.path.basename(ll.split()[0]) # larcvtruth-Run000001-SubRun001869.root
            x = x[x.find("-")+1:].split(".")[0]
        
        file_hashes[x] = False
        file_pairs[x] = [ll.split()[0],ll.split()[1]] # [ larcv_mctruth, mcinfo ]
            
    print("input list has ",len(file_hashes)," files.")



ncompleted = 0
for outdir in outdirs:        
    completed = os.listdir(outdir)
    for f in completed:
        #print(f)
        if "bnb_nu" in samplename:
            # file looks like: [dir]/larmatch_trainingdata_0001b7e4-e09a-4a2e-9eb4-0c47de49dfb1.h5
            f = os.path.basename(f) # larmatch_trainingdata_0001b7e4-e09a-4a2e-9eb4-0c47de49dfb1.h5            
            fhash = f.strip().split("_")[-1].split(".")[0] # 0001b7e4-e09a-4a2e-9eb4-0c47de49dfb1
        else:
            # files look like: [dir]/
            f = os.path.basename(f) # larmatch_trainingdata-Run000001-SubRun000003.h5
            fhash = f[f.find("-")+1:].split(".")[0]
        
        if fhash in file_hashes:
            file_hashes[fhash] = True
            ncompleted += 1
        else:
            # hack
            print("unrecognized file hash in the input list: ",fhash,f)
            # hack
            if samplename=="mcc9_v13_bnbnue_corsika_training":
                cmd = "mv %s/%s %s/"%("outdir_mcc9_v13_bnbnue_corsika_training",f,"outdir_mcc9_v13_bnbnue_corsika_validation")
                print("  ",cmd)
                os.system(cmd)

print("Number of completed files: ",ncompleted)
hashlist = [ x for x in file_hashes.keys() ]
hashlist.sort()

makeupfile = 'makeuplist.%s.paired.txt'%(samplename)
nmakeup = 0
with open(makeupfile,'w') as fout:
    for xhash in hashlist:
        if not file_hashes[xhash]:
            print(file_pairs[xhash][0]," ",file_pairs[xhash][1],file=fout)
            nmakeup += 1
print("number of jobs to makeup: ",nmakeup)
print("made makeupfile: ",makeupfile)

