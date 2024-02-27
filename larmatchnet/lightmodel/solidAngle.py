# This script makes a csv file of solid angle values
# per entry (interaction), given a set of coordinates and flashTick

#import torch
import ROOT as rt
import numpy as np
import torch
import sys, argparse
#np.set_printoptions(threshold=sys.maxsize)

import time
import math
import mpmath
import scipy
from scipy import special

from larlite import larlite
from larlite import larutil
import lardly
from lardly.ubdl.pmtpos import pmtposmap


dv = larutil.LArProperties.GetME().DriftVelocity()
voxelsize = 5

solidAngle = 0.0
radius = 10.16 # PMT disk radius in cm

# takes 3 inputs: r0, rm, and L from the paper
def solidAngleCalc(r0, rm, L):

    Rmax = math.sqrt( L**2 + (r0+rm)**2 )
    R1 = math.sqrt( L**2 + (r0-rm)**2 )
    k = 2*math.sqrt( r0 * rm / (L * L + ( r0 + rm )**2) )
    alpha = (4*r0*rm) / (r0+rm)**2 #alpha^2 in the paper

    if (r0 < rm): 
        solidAngle = 2*math.pi - 2*(L/Rmax)*(scipy.special.ellipk (k**2) + math.sqrt(1 - alpha) * mpmath.ellippi(alpha, k**2))

    if (r0 == rm):
        solidAngle = math.pi - (2*L / Rmax) * scipy.special.ellipk (k**2)

    if (r0 > rm): 
        solidAngle = (2*L / Rmax) * ( ((r0-rm) / (r0+rm))*mpmath.ellippi(alpha, k**2) - scipy.special.ellipk (k**2))

    return solidAngle

def coordFlashFromFile(input_file, entry):

    tfile = rt.TFile(input_file,'open')
    larvoxeltrainingdata  = tfile.Get('larvoxeltrainingdata')
    print("Got tree")  

    larvoxeltrainingdata.GetEntry(entry)

    coord_v = larvoxeltrainingdata.coord_v
    flashTick = larvoxeltrainingdata.flashTick

    coord_np = []
    for i in range( coord_v.size() ):
        print("i: ", i)
        coord = coord_v.at(i)
        coord_np = np.copy( coord.tonumpy() )
        print("This is coord_np: ", coord_np)

    return coord_np, flashTick

# this method calls the solidAngleCalc above for an interaction
# for the 32 PMTs
# given an interactions coords and flashtick
def makeSA(coord_v, flashTick, ientry): 

    # Grab X component and convert to cm
    # (note anode plane x-coord is 0.0 cm)
    coord_x = coord_v[:,0]*voxelsize+(2399-3200-(flashTick-3200))*0.5*dv

    SA_allPMTs = []

    for ipmt in range(0,32): #32

        print("This is PMT #: ", ipmt)

        center = [pmtposmap[ipmt][0]-15.0, pmtposmap[ipmt][1], pmtposmap[ipmt][2] ]
        print("center of pmt is: ", center)
        x_center = center[0]
        y_center = center[1]
        z_center = center[2]
        print("y_center",y_center)
        print("z_center",z_center)

        print("x-coords are (not in cm): ", coord_v[:,0] )
        print("x-coords are (in cm): ", coord_v[:,0]*voxelsize+(2399-3200-(flashTick-3200))*0.5*dv )

        print("y-coords are (not in cm): ", coord_v[:,1] )
        print("y-coords are (in cm): ", coord_v[:,1]*voxelsize-120.0 )

        print("z-coords are (not in cm): ", coord_v[:,2] )
        print("z-coords are (in cm): ", coord_v[:,2]*voxelsize )

        L = coord_x - x_center
        print("Type of L:", type(L) )
        print("Size of L: ", L.size)
        print("Shape of L: ", L.shape)
        print("This is L (all coords for this one PMT): ", L)
        print("coord_v[:,1]*voxelsize-120.0 - y_center):", coord_v[:,1]*voxelsize-120.0 - y_center)
        print("coord_v[:,2]*voxelsize - z_center:", coord_v[:,2]*voxelsize - z_center)
        X2 = ( coord_v[:,0]*voxelsize+(2399-3200-(flashTick-3200))*0.5*dv - x_center )**2
        Y2 = ( coord_v[:,1]*voxelsize-120.0 - y_center)**2
        Z2 = ( coord_v[:,2]*voxelsize - z_center )**2
        print("X^2: ", X2)
        print("Y^2: ", Y2 )
        print("Z^2: ", Z2 )
        print("Y^2+Z^2: ",Y2 + Z2 )

        R = np.sqrt(X2 + Y2 + Z2) # this is an array with all coords in the interxn

        print("R is: ", R)

        inverseR2 = 1 / (R**2)

        zy_offset = np.sqrt( ( coord_v[:,1]*voxelsize-120.0 - y_center)**2 + ( coord_v[:,2]*voxelsize - z_center )**2 )  #sqrt(Y^2 + X^2)
        print("This is zy_offset (all coords for this one PMT):", zy_offset)

        SA_list = []
        #SA_list.append(ientry)
        for j in range( L.size ):
            SA = solidAngleCalc(zy_offset[j],radius,L[j])
            print("This is the solid angle calc!", SA)
            SA_list.append(SA)

        print("THIS IS SA LIST: ", SA_list)

        #SA_allPMTs.append(ientry)
        SA_allPMTs.append(SA_list)

        print("THIS IS SA_ALLPMTS: ", SA_allPMTs)

    SA_np = np.array(SA_allPMTs)
    SA_np = SA_np.astype(float)
    print("This is the shape of the SA output: ", SA_np.shape)

    print("Need to take the transpose so shape is (N,32).")
    SA_transpose = np.transpose(SA_np)
    print("Shape is now: ", SA_transpose.shape )

    SA_t = torch.from_numpy( SA_transpose )
    ####SA_t = torch.from_numpy( SA_np )

    return SA_t

# this fn only takes in the coord_np and pmt number
def makeSA(coord_v, ipmt): 

    coord_x = coord_v[:,0]*voxelsize

    ####SA_allPMTs = []

    ####for ipmt in range(0,32): #32

    print("This is PMT #: ", ipmt)

    center = [pmtposmap[ipmt][0]-15.0, pmtposmap[ipmt][1], pmtposmap[ipmt][2] ]
    print("center of pmt is: ", center)
    x_center = center[0]
    y_center = center[1]
    z_center = center[2]
    print("y_center",y_center)
    print("z_center",z_center)

    print("x-coords are (not in cm): ", coord_v[:,0] )
    print("x-coords are (in cm): ", coord_v[:,0]*voxelsize )

    print("y-coords are (not in cm): ", coord_v[:,1] )
    print("y-coords are (in cm): ", coord_v[:,1]*voxelsize )

    print("z-coords are (not in cm): ", coord_v[:,2] )
    print("z-coords are (in cm): ", coord_v[:,2]*voxelsize )

    L = coord_x - x_center
    print("Type of L:", type(L) )
    print("Size of L: ", L.size)
    print("Shape of L: ", L.shape)
    print("This is L (all coords for this one PMT): ", L)
    ##print("coord_v[:,2]*voxelsize - z_center:", coord_v[:,2]*voxelsize - z_center)
    X2 = ( coord_v[:,0]*voxelsize - x_center )**2
    Y2 = ( coord_v[:,1]*voxelsize - y_center)**2
    Z2 = ( coord_v[:,2]*voxelsize - z_center )**2
    print("X^2: ", X2)
    print("Y^2: ", Y2 )
    print("Z^2: ", Z2 )
    print("Y^2+Z^2: ",Y2 + Z2 )

    R = np.sqrt(X2 + Y2 + Z2) # this is an array with all coords in the interxn

    print("R is: ", R)

    inverseR2 = 1 / (R**2)

    zy_offset = np.sqrt( ( coord_v[:,1]*voxelsize - y_center)**2 + ( coord_v[:,2]*voxelsize - z_center )**2 )  #sqrt(Y^2 + X^2)
    print("This is zy_offset (all coords for this one PMT):", zy_offset)

    SA_list = []
    #SA_list.append(ientry)
    print("L.size is: ", L.size)
    for j in range( L.size ):
        SA = solidAngleCalc(zy_offset[j],radius,L[j])
        ####print("This is the solid angle calc!", SA, "for entry: ", j)
        SA_list.append(SA)

    ##print("THIS IS SA LIST: ", SA_list)

    #SA_allPMTs.append(ientry)
    ####SA_allPMTs.append(SA_list)

    ####print("THIS IS SA_ALLPMTS: ", SA_allPMTs)

    ####SA_np = np.array(SA_allPMTs)
    SA_np = np.array(SA_list)
    SA_np = SA_np.astype(float)
    print("This is the shape of the SA output: ", SA_np.shape)

    '''
    print("Need to take the transpose so shape is (N,32).")
    SA_transpose = np.transpose(SA_np)
    print("Shape is now: ", SA_transpose.shape )

    SA_t = torch.from_numpy( SA_transpose )
    '''

    return SA_np ####SA_t

def fullDetectorCoords(voxsize) :

    # detector size in cm
    # adding a buffer of 10 cm 
    x_cm = 256+10
    y_cm = 234+10
    z_cm = 1036+10

    # number of voxels given voxelsize.
    # // for integer division, and then if there's a remainder round up
    total_x_voxels = x_cm // voxelsize + (x_cm % voxelsize > 0)
    print("total_x_voxels: ", total_x_voxels)
    x_voxel_array = np.arange(0, total_x_voxels, 1)
    
    total_y_voxels = y_cm // voxelsize + (y_cm % voxelsize > 0)
    print("total_y_voxels: ", total_y_voxels)
    y_voxel_array = np.arange(0, total_y_voxels, 1)

    total_z_voxels = z_cm // voxelsize + (z_cm % voxelsize > 0)
    #x_voxels = [ x_cm / voxsize ]
    print("total_z_voxels: ", total_z_voxels)
    z_voxel_array = np.arange(0, total_z_voxels, 1)

    SA_allPMTs = []
    #for ipmt in range(0,1): #32

    I = np.arange(total_x_voxels*total_y_voxels*total_z_voxels)
    #print("I: ", I)
    #print("I.shape: ". I.shape)
    coord_np = np.column_stack((np.repeat(x_voxel_array, total_y_voxels*total_z_voxels), 
                              np.tile( np.repeat(y_voxel_array, total_z_voxels), total_x_voxels), 
                              np.tile(z_voxel_array, total_y_voxels*total_x_voxels)))
    print("coord_np: ", coord_np)

    #SA_t = z_voxel_array

    return coord_np

# make csv file that contains SA for 32 PMTs for each voxel, for each entry 
def makeCSVPerEntry(SA_vals, ientry):
    print("This is the SA_vals that go into the CSV: ", SA_vals)
    print("Type of SA_vals: ", type(SA_vals))
    np.savetxt('SA_020624_voxelsize5_allPMTS_entry%d.csv'% (ientry), SA_vals, delimiter=',')
    #np.save('numpy_array_1.npy', SA_vals)

def makeCSV(SA_vals):
    #print("This is the SA_vals that go into the CSV: ", SA_vals)
    #print("Type of SA_vals: ", type(SA_vals))
    np.savetxt('SA_FullDetector_voxelsize5_32PMTs.csv', SA_vals, delimiter=',')
    #np.save('numpy_array_1.npy', SA_vals)

# Test it out!
if __name__=="__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Run solid angle calculator')
    parser.add_argument('-p','--PMTID',type=int,default=0,help="PMT ID to run over (0-31; 0 by default)")
    args = parser.parse_args()

    print("args.PMTID: ", args.PMTID)

    input = "testfm_010724_FMDATA_coords_withErrorFlags_100Events_voxelsize5cm_010724.root"
    ####num = 2

    # test
    print("solidAngleCalc test from class:", solidAngleCalc(0.2,1,1) )

    print("dv: ", dv)
    coords_fullDetector = fullDetectorCoords(voxelsize)
    print("output of the fullDetectorCoordsFunction: ",  coords_fullDetector )
    print("coords_fullDetector.shape ",  coords_fullDetector.shape)
    #coords_fullDetector = np.ones((3,3))
    print("coords_fullDetector.shape ",  coords_fullDetector.shape)

    np.savetxt('fullDetectorCoords_5cmVoxels.csv', coords_fullDetector, delimiter=',')
    
    #####S##A_out = makeSA(coords_fullDetector, args.PMTID)
    ####print("This is the result of calling SA on the full detector coords for PMT number ", args.PMTID, ": ", SA_out )

    # now save this list for 1 PMT to a file
    #######np.savetxt('SA_batchJob_voxelsize5_PMT%d.csv'% (args.PMTID), SA_out, delimiter=',')

    ####makeCSV(SA_out)

    ##SA_tensor = torch.empty((32, 0), dtype=torch.float32)
    ##SA_tensor_t = torch.transpose(SA_tensor, 0, 1)

    '''
    for num in range(2,17): # how many entries do I want total?

        coords, ftick = coordFlashFromFile(input, num)
        print("coords, ftick", coords, ", ", ftick)

        SA_tensor = makeSA(coords, ftick, num)
        print("This is the SA tesnor!", SA_tensor)
    '''
        
    #SA_tensor_temp_t = torch.transpose(SA_tensor_temp, 0, 1)

    ####full_tensor = torch.cat((SA_tensor, SA_tensor_temp), 0)
    ####SA_tensor = full_tensor

    ####SA_tensor_t = torch.transpose(SA_tensor, 0, 1)
            
    ###############makeCSV(SA_tensor, num)

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:",execution_time, " seconds")   
