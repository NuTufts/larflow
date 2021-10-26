import os,sys,pickle

# Used to make inputlists/inputlists/mcc9_v13_bnb_nu_corsika.paired.list
# not needed to rerun unless sample changes

larcvlist=open('inputlists/mcc9_v13_bnb_nu_corsika.list','r')
ll = larcvlist.readlines()

fileoutname="inputlists/mcc9_v13_bnb_nu_corsika.paired.list"
fout=open(fileoutname,'w')

# load sample info
mergedlist = open("/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v13_bnb_nu_corsika/mergedlist_mcc9_v13_bnb_nu_corsika.pkl",'rb')
data = pickle.load( mergedlist )
mctruth_mcinfo_map = {}
for x in data:
    lcv_mct = os.path.basename(data[x]["larcv_mctruth"])
    ll_mcinfo = os.path.basename(data[x]["larlite_mcinfo"])
    mctruth_mcinfo_map[lcv_mct] = ll_mcinfo

for l in ll:
    l = l.strip()
    lbase = os.path.basename(l)
    if lbase not in mctruth_mcinfo_map:
        continue
    mcinfofile = mctruth_mcinfo_map[lbase]
    folder = os.path.dirname(l).replace("larcv_mctruth","larlite_mcinfo")
    mcinfopath=folder+"/"+mcinfofile
    print(mcinfopath)
    print(l," ",mcinfopath,file=fout)

fout.close()

