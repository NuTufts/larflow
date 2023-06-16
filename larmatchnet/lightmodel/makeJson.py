import argparse,json

# in bash script, write list of mcinfo paths to a temporary txt file
# there should be 5 lines
# here I edit the filename for the 5 opreco paths (replace mcinfo -> opreco)
# and take triplet file path as an input arg

parser = argparse.ArgumentParser("Make json file")
parser.add_argument('-t','--triplet',required=True,type=str,help="Path with triplet file required. [required]")
args = parser.parse_args()

mcinfoPathList = open("paths.txt",'r').read().split('\n')
print("mcinfoPathList: ", len(mcinfoPathList))
mcinfoPathList.pop() # removes last blank "" entry
oprecoPathList = [w.replace('mcinfo', 'opreco') for w in mcinfoPathList] # list comprehension to replace
print("mcinfoPathList: ", len(mcinfoPathList))

mcinfo = mcinfoPathList
opreco = oprecoPathList
triplet = args.triplet

topDict = {}
pathDict = {}

pathDict["mcinfo"] = mcinfo
pathDict["opreco"] = opreco
pathDict["triplet"] = triplet
topDict["0"] = pathDict

json1 = json.dumps(topDict)

fh = open("tempJson.json", "w")
fh.write( json1 )
fh.close()

