import os,sys,json

f = open("inputlists/mcc9_v13_bnbnue_corsika.paired.list",'r')
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
ft = open("inputlists/triplets.list",'r')
ll = ft.readlines()
for l in ll:
    fname = os.path.basename(l.strip())
    fileid = int(fname.split("_")[-1].split(".")[0])
    if fileid in input_groups:
        out_json[fileid] = {"mcinfo":input_groups[fileid],
                            "triplet":l.strip()}

with open('filelist.json', 'w') as outfile:
    json.dump(out_json, outfile)
    

        
