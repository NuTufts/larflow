import os,sys
import ROOT as rt
from ROOT import std
import numpy

from larlite import larlite
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
import os,sys,argparse,time

parser = argparse.ArgumentParser("Read TTree")
parser.add_argument("-f", "--input-tree",required=True,type=str,help="Input TTreee file [required]")
args = parser.parse_args()

f = rt.TFile.Open(args.input_tree)
tree = f.Get('trainingdata')
print tree

i = 0
for entry in tree:
    # temp = entry.DataBranch.getIntVector()
    # temp = entry.DataBranch.coord_t_pyarray(0)
    # print entry.DataBranch.intvector3.size()
    # temp = entry.DataBranch.intvector3[0][0].row
    # print temp
    # print entry.DataBranch.intvector3[0][0].col

    print entry.DataBranch.coord_t_pyarray(0)
    # f.write(str(entry.DataBranch.coord_t_pyarray(0)))

    # print entry.DataBranch.num_instances_0

    # entry.DataBranch.printIntVectorSize()


    i += 1

print i


print 'End'