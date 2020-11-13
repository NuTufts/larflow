import os,sys

inputlist = "inputlists/mcc9_v13_bnbnue_corsika.txt"

panafiles = os.popen("ls outdir/larmatchtriplet_ana*root | sort")
lana_input = panafiles.readlines()

for i in range(5):
    ftrain = open("trainlist_p%02d.txt"%(i),'w')
    for l in lana_input[i*40:(i+1)*40]:
        print>>ftrain,l.strip()
    ftrain.close()

for i in xrange(2):
    fvalid = open("validlist_p%02d.txt"%(i),'w')
    start = 200+i*25
    end   = 200+(i+1)*25
    if i+1==2:
        end = len(lana_input)
    for l in lana_input[start:end]:
        print>>fvalid,l.strip()
    fvalid.close()

