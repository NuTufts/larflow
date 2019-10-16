import os,sys

inputlist = "inputlists/mcc9_v13_bnbnue_corsika.txt"

panafiles = os.popen("ls outdir/*ana*root | sort")
lana_input = panafiles.readlines()
plcvfiles = os.popen("ls outdir/*larcv*.root | sort")
llcv_input = plcvfiles.readlines()

for i in range(2):
    ftrain = open("trainlist_p%02d.txt"%(i),'w')
    for l in lana_input[i*100:(i+1)*100]:
        print>>ftrain,l.strip()
    for l in llcv_input[i*100:(i+1)*100]:
        print>>ftrain,l.strip()
    ftrain.close()

fvalid = open("validlist.txt",'w')
for l in lana_input[200:]:
    print>>fvalid,l.strip()
for l in llcv_input[200:]:
    print>>fvalid,l.strip()
fvalid.close()
