import os,sys
import numpy as np
import torch
sys.path.append("../")
from larflow_consistency3d_loss import LArFlow3DConsistencyLoss
from func_intersect_ub import IntersectUB
from larlite import larlite
from larlite import larutil
from ROOT import std,Double

## create underlying truth (random choice of Y,U,V values)

loss3d = LArFlow3DConsistencyLoss( 3, 1, 1, intersectiondata="../../gen3dconsistdata/consistency3d_data_larcv2.root", larcv_version=2 )

print IntersectUB.src_index_t.shape
print IntersectUB.src_index_t

# picking y and u values at random
print "==============================="
print "Inputs"
input_y_wires = [1300.,1301.,1302.]
input_u_wires = [ 800., 900.,1000.]
input_v_wires = [1100.,1200.,1300.]
print "input-y-wires: ",input_y_wires
print "input-u-wires: ",input_u_wires
print "input-v-wires: ",input_v_wires

yorigin = [1300.,1300.,1300.]
uorigin = [ 700., 700., 700.]
vorigin = [1050.,1050.,1050.]
print "y-origin: ",yorigin
print "u-origin: ",uorigin
print "v-origin: ",vorigin


y2u_flowvalues= [ (input_u_wires[i]-uorigin[i]) - (input_y_wires[i]-yorigin[i]) for i in xrange(len(input_y_wires)) ]
y2v_flowvalues= [ (input_v_wires[i]-vorigin[i]) - (input_y_wires[i]-yorigin[i]) for i in xrange(len(input_y_wires)) ]
flow_y2u_t = torch.Tensor( y2u_flowvalues ).reshape( (1,1,3,1) )
flow_y2v_t = torch.Tensor( y2v_flowvalues ).reshape( (1,1,3,1) )
yorigin_t  = torch.Tensor( yorigin ).reshape( (3,1) )
uorigin_t  = torch.Tensor( uorigin ).reshape( (3,1) )
vorigin_t  = torch.Tensor( vorigin ).reshape( (3,1) )
print "flow_y2u_t: ",flow_y2u_t
print "flow_y2v_t: ",flow_y2v_t

fmask_y2u_t = torch.ones( (1,1,3,1) )
fmask_y2v_t = torch.ones( (1,1,3,1) )

# require grad for fake flow inputs
flow_y2u_t.requires_grad_(True)
flow_y2v_t.requires_grad_(True)

# calculate loss
loss = loss3d( flow_y2u_t, flow_y2v_t, fmask_y2u_t, fmask_y2v_t, yorigin, uorigin, vorigin )

# use IntersectUB let (y,z) detector positions
print loss

# run backward
loss.backward()


# gradients
print "loss: ",loss.grad
print "flow_y2u.grad: ",flow_y2u_t.grad
print "flow_y2v.grad: ",flow_y2v_t.grad


# use consistency matrix to get correct (y,z) detector positions
pos3dyu_ypo = [ IntersectUB.intersections_t[0,0,int(input_y_wires[i]),int(input_u_wires[i])] for i in xrange(3) ]
pos3dyu_zpo = [ IntersectUB.intersections_t[0,1,int(input_y_wires[i]),int(input_u_wires[i])] for i in xrange(3) ]
pos3dyv_ypo = [ IntersectUB.intersections_t[1,0,int(input_y_wires[i]),int(input_v_wires[i])] for i in xrange(3) ]
pos3dyv_zpo = [ IntersectUB.intersections_t[1,1,int(input_y_wires[i]),int(input_v_wires[i])] for i in xrange(3) ]

print "======================================"
print "Positions from flow loss function "
print "pos3d_y2u: ",
for i in xrange(len(pos3dyu_ypo)):
    print "(%.2f,%.2f) "%(pos3dyu_ypo[i],pos3dyu_zpo[i]),
print
print "pos3d_y2v: ",
for i in xrange(len(pos3dyv_ypo)):
    print "(%.2f,%.2f) "%(pos3dyv_ypo[i],pos3dyv_zpo[i]),
print

print "======================================"
print "Intersection check "
geo = larutil.Geometry.GetME()
intersect_y2u = []
intersect_y2v = []
for i in xrange(3):
    fy = Double()
    fz = Double() 
    geo.IntersectionPoint( int(input_y_wires[i]), int(input_u_wires[i]), 2, 0, fy, fz )
    intersect_y2u.append( (fy,fz) )
    fy = Double()
    fz = Double()    
    geo.IntersectionPoint( int(input_y_wires[i]), int(input_v_wires[i]), 2, 1, fy, fz )    
    intersect_y2v.append( (fy,fz) )    
print "Y-U intersection: {}".format( intersect_y2u )
print "Y-V intersection: {}".format( intersect_y2v )


print "======================================"
print "Loss Check"
losscheck = []
ave = 0
for i in xrange(3):
    dy = intersect_y2u[i][0]-intersect_y2v[i][0]
    dz = intersect_y2u[i][1]-intersect_y2v[i][1]
    dd = dy*dy + dz*dz
    losscheck.append( dd )
    ave += dd
ave /= len(input_y_wires)
print "Losses: ",losscheck," ave",ave
    

print "======================================"
print "Newton test"
learning_rate = 1.0

niters = 100
last_flowy2u_t = flow_y2u_t.detach().requires_grad_(True)
last_flowy2v_t = flow_y2v_t.detach().requires_grad_(True)
for iiter in xrange(niters):
    print "---------------------------------------------------------"
    #print "iter[",iiter,"]: ",last_flowy2u_t[0,0,:,0]," ",last_flowy2v_t[0,0,:,0]
    print "iter[",iiter,"]: ",last_flowy2u_t," ",last_flowy2v_t
    loss = loss3d( last_flowy2u_t, last_flowy2v_t, fmask_y2u_t, fmask_y2v_t, yorigin, uorigin, vorigin )
    loss.backward()
    print "LOSS: ",loss
    
    print "y2u.grad: ",last_flowy2u_t.grad
    print "y2v.grad: ",last_flowy2v_t.grad

    last_flowy2u_t = last_flowy2u_t.detach() - last_flowy2u_t.grad*learning_rate
    last_flowy2v_t = last_flowy2v_t.detach() - last_flowy2v_t.grad*learning_rate

    last_flowy2u_t.requires_grad_(True)
    last_flowy2v_t.requires_grad_(True)
    

print "========================================================"
print "FINAL LOSS: ",loss
print "Final flow [Y2U]: ",last_flowy2u_t
print "Final flow [Y2V]: ",last_flowy2v_t
print "========================================================"
