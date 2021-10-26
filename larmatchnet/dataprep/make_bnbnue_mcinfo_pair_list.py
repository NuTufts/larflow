import os,sys,pickle

larcvlist=open('inputlists/mcc9_v13_bnbnue_corsika.list','r')
ll = larcvlist.readlines()

fileoutname="inputlists/mcc9_v13_bnbnue_corsika.paired.list"
fout=open(fileoutname,'w')

# load sample info

for l in ll:
    l = l.strip()
    lbase = os.path.basename(l)
    mcinfopath=l.replace("larcvtruth","mcinfo")
    print(mcinfopath)
    print(l," ",mcinfopath,file=fout)
fout.close()

