#import ROOT
from ROOT import TFile, TTree
from array import array

f = TFile("crt_0-1318.root","READ")
hist3D = f.Get("hitcount_xyz_th3d")
#f.Print()

# new ROOT file that will contain tree
newF = TFile("hitsPerVoxel.root","recreate")
newT = TTree("tree","tree")

hitsPerVoxel = array('d',[0.])
newT.Branch('hitsPerVoxel', hitsPerVoxel, 'hitsPerVoxel/D')

newF.cd()

binx = hist3D.GetNbinsX()
biny = hist3D.GetNbinsY()
binz = hist3D.GetNbinsZ()
minx = hist3D.GetXaxis().GetXmin()
miny = hist3D.GetYaxis().GetXmin()
minz = hist3D.GetZaxis().GetXmin()
maxx = hist3D.GetXaxis().GetXmax()
maxy = hist3D.GetYaxis().GetXmax()
maxz = hist3D.GetZaxis().GetXmax()

print binx, biny, binz
print minx, miny, minz
print maxx, maxy, maxz

print hist3D.GetBinContent(75,75,75)
"""
for i in xrange( 0, binx):
    for j in xrange(0, biny):
        for k in xrange(0, binz):
            hitsPerVoxel[0] = hist3D.GetBinContent(i, j, k)

newT.Fill()

initialBin = hist3D.GetBin( int( minx ), int( miny ), int( minz ) )

for i in xrange( 0, binx ):
    print hist3D.GetBinContent( initialBin + i )
"""

newF.Write()
newF.Close()

print "Done!"
