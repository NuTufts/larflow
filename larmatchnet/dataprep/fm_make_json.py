import os,sys,json

# TEST-BNB-NUE
#paired_input_list = "inputlists/mcc9_v13_bnbnue_corsika.paired.list"
#triplet_list="inputlists/triplets.list"
#outfilename="filelist_test.json"

# FM TEST-BNB-NUE
paired_input_list = "/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.paired.list"
triplet_list="/cluster/tufts/wongjiradlabnu/pabrat01/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.triplettruth.list"
outfilename="filelist_fm_bnb_nue.json"

# NUE DATA
#paired_input_list = "/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.paired.list"
#triplet_list = "/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnbnue_corsika.triplettruth.list"
#outfilename="filelist_bnb_nue.json"

# BNB NU DATA
#paired_input_list="/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika.paired.list"
#triplet_list="/cluster/tufts/wongjiradlabnu/twongj01/gen2/ubdl/larflow/larmatchnet/dataprep/inputlists/mcc9_v13_bnb_nu_corsika.triplettruth.list"
#outfilename="filelist_bnb_nu.json"

f = open(paired_input_list,'r')
ll = f.readlines()

input_groups = {}

i = 0
for l in ll:
    l = l.strip()
    mcinfo = l.split()[1]
    larcvtruth = l.split()[0]
    g = i//5
    if g not in input_groups:
        input_groups[g] = []
    mcinfo = os.environ["PWD"]+"/testdata/"+os.path.basename(mcinfo)
    input_groups[g].append(mcinfo)
    i += 1

# read triplet list
out_json = {}
ft = open(triplet_list,'r')
ll = ft.readlines()
for l in ll:
    fname = os.path.basename(l.strip())
    fileid = int(fname.split("_")[-1].split(".")[0])
    if fileid in input_groups:
        out_json[fileid] = {"mcinfo":input_groups[fileid],
                            "triplet":l.strip()}

with open(outfilename, 'w') as outfile:
    json.dump(out_json, outfile)
