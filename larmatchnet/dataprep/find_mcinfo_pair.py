import os,sys,pickle

larcv_mctruth_filepath = sys.argv[1]

mergedlist = open("/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v13_bnb_nu_corsika/mergedlist_mcc9_v13_bnb_nu_corsika.pkl",'r')
data = pickle.load( mergedlist )

mctruth_mcinfo_map = {}

for x in data:
    lcv_mct = os.path.basename(data[x]["larcv_mctruth"])
    ll_mcinfo = os.path.basename(data[x]["larlite_mcinfo"])
    mctruth_mcinfo_map[lcv_mct] = ll_mcinfo

larcv_mctruth_filename = os.path.basename(larcv_mctruth_filepath)

if larcv_mctruth_filename in mctruth_mcinfo_map:
    print mctruth_mcinfo_map[larcv_mctruth_filename]
else:
    print "0"
