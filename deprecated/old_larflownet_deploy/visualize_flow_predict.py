from __future__ import print_function
import os,sys
import numpy as np
import ROOT as rt
from larcv import larcv
larcv.load_rootutil()

rt.gStyle.SetOptStat(0)

# Open file, get entry, loop throw flow images and indicate prediction

flowout = sys.argv[1]
entry   = int(sys.argv[2])

io = larcv.IOManager()
io.add_in_file(flowout)
io.initialize()

flowdir='y2u'
if 'y2u' in flowout:
    flowdir='y2u'
else:
    flowdir='y2v'

io.read_entry(entry)
try:
    ev_flowimgs = io.get_data(larcv.kProductImage2D, "larflow_{}".format(flowdir))
    ev_cropped  = io.get_data(larcv.kProductImage2D, "adc")
except:
    raise RuntimeError("Missing {} output".format(flowdir) )


flow_v = ev_flowimgs.Image2DArray()
cropped_v = ev_cropped.Image2DArray()
nimgs  = flow_v.size()

print("number of flow crops: ",nimgs)
print("number of adc crops:  ",cropped_v.size())

c = rt.TCanvas("c","LArFlow",1500,400)
c.Divide(3,1)

for iset in xrange(nimgs):
    srcimg = cropped_v.at(iset*3+2)
    if flowdir=='y2u':
        tarimg = cropped_v.at(iset*3+0)
    else:
        tarimg = cropped_v.at(iset*3+1)

    hsrc = larcv.as_th2d( srcimg, "hsrc_set{}".format(iset) )
    htar = larcv.as_th2d( tarimg, "htar_set{}".format(iset) )
    hsrc.GetZaxis().SetRangeUser(0,100)
    htar.GetZaxis().SetRangeUser(0,100)
    hsrc.SetTitle("Source Plane Y: Set {};wire;tick".format(iset))
    if flowdir=='y2u':
        htar.SetTitle("Target Plane U: Set {};wire;tick".format(iset))
    else:
        htar.SetTitle("Target Plane V: Set {};wire;tick".format(iset))

    # prepare flow image
    flow = flow_v.at(iset)
    src_np = larcv.as_ndarray(srcimg)
    flow_np = larcv.as_ndarray(flow).astype(np.int)
    colidx = np.arange(src_np.shape[0],dtype=np.int)
    for row in xrange(src_np.shape[1]):
        flow_np[:,row] += colidx[:]
    flow_np[ src_np<10 ] = 0
    flow_np[ flow_np<0 ] = 0
    flow_np[ flow_np>=src_np.shape[0] ] = 0

    flowed_np = np.zeros( src_np.shape, dtype=np.float32 )
    pixlist   = np.argwhere( flow_np!=0 )
    print("pixlist: ",pixlist.shape)
    for idx in xrange(pixlist.shape[0]):
        row     = pixlist[idx,1]
        src_col = pixlist[idx,0]
        tar_col = flow_np[src_col,row]
        #print(row,src_col,"to",tar_col)
        flowed_np[ tar_col, row ] = src_np[src_col,row]
    flowed_lcv = larcv.as_image2d_meta(flowed_np,tarimg.meta())
    hflowed = larcv.as_th2d(flowed_lcv,"hflowed_{}".format(iset))
    hflowed.GetZaxis().SetRangeUser(0,100)
    hflowed.SetTitle("Flowed Pixels from {};wire;tick".format(flowdir))
    
    c.cd(1)
    hsrc.Draw("COLZ")

    c.cd(2)
    hflowed.Draw("COLZ")
    
    c.cd(3)
    htar.Draw("COLZ")

    c.Update()
    c.Draw()
    print("Viewing Set {} of {}".format(iset,nimgs))
    print("[Enter] to continue")
    raw_input()

