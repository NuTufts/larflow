# This is the dataloader for the Light Model network.
# Reads ROOT file with FMDATA nd opflash info, returns 
# torch tensors: coords, feat, and truth.
#
# Called by train.py.

import ROOT as rt
import torch
import numpy as np
import time

from larlite import larlite
####import lardly
####from lardly.ubdl.pmtpos import pmtposmap

####from solidAngle import solidAngleCalc

####from larlite import larutil


####dv = larutil.LArProperties.GetME().DriftVelocity()
####voxelsize = 5

'''
# Takes in coordinate array, outputs array of SA values
def calc_solid_angle(coord_v, flashTick):

    print("Called function calc_solid_angle")
    
    radius = 10.16 # PMT disk radius in cm

    print("coord_v: ", coord_v)
    print("coord_v shape: ", coord_v.shape)

    # Grab X component and convert to cm
    coord_x = coord_v[:,0]*voxelsize+(2399-3200-(flashTick-3200))*0.5*dv
    # note anode plane x-coord is 0.0 cm
    #pmtposmap = lardly.ubdl.getPosFromID( 1, origin_at_detcenter=False )
    print("pmt pos map: ", pmtposmap)

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

        np.savetxt('Rvals.csv', R, delimiter=',')
        np.savetxt('inverseR2vals.csv', inverseR2, delimiter=',')

        zy_offset = np.sqrt( ( coord_v[:,1]*voxelsize-120.0 - y_center)**2 + ( coord_v[:,2]*voxelsize - z_center )**2 )  #sqrt(Y^2 + X^2)
        print("This is zy_offset (all coords for this one PMT):", zy_offset)
        #for j in range( len(coord_x) ):
        
        SA_list = []
        for j in range( L.size ):
            SA = solidAngleCalc(zy_offset[j],radius,L[j])
            #print("This is the solid angle calc!", SA)
            SA_list.append(SA)

        SA_allPMTs.append(SA_list)
        
    return SA_allPMTs
'''

# Loads in ROOT trees, outputs torch tensors (coords, feats, truth)
def load_lm_data(input_file, opinput_file, entry, batchsize):

    coordBatch = []
    featBatch = []
    truthBatch = []

    # load tree                                                                                                                                                               
    tfile = rt.TFile(input_file,'open')
    larvoxeltrainingdata  = tfile.Get('larvoxeltrainingdata')
    print("Got tree")   

    opio = larlite.storage_manager( larlite.storage_manager.kREAD )
    opio.add_in_filename( opinput_file )
    opio.open()

    ##tfile2 = rt.TFile('100events_062323_FMDATA_filtered_MCTracks_opflash.root', 'open')
    ##opflashTree = tfile.Get('opflash_simpleFlashCosmic_tree')

    #vector_a = []

    input_files = rt.std.vector("std::string")()
    input_files.push_back(input_file)

    for ibatch in range(entry, entry+batchsize):

        larvoxeltrainingdata.GetEntry(ibatch)
        opio.go_to(ibatch)

        ev_opflash_cosmic = opio.get_data(larlite.data.kOpFlash,"simpleFlashCosmic")
        ev_opflash_beam = opio.get_data(larlite.data.kOpFlash,"simpleFlashBeam")

        #ev_opflash = ev_opflash_cosmic.at(0)

        ##print("Solid angle test: ", solidAngleCalc(0.2,1,1))

        flash = []

        origin = larvoxeltrainingdata.origin
        print("Origin here is (2 for cosmic, 1 for neutrino): ", origin)

        if (origin == 2): # cosmic event
            #print("ev_opflash_cosmic: ", (ev_opflash))
            print("ev_opflash_cosmic.size()", ev_opflash_cosmic.size())
            for i in range(200,232): #range for cosmic channels
                print("ev_opflash_cosmic[0]: ", ev_opflash_cosmic[0].PE(i), " ")
                flash.append( ev_opflash_cosmic[0].PE(i) )
        
        if (origin == 1): # neutrino event
            print("ev_opflash_beam.size()", ev_opflash_beam.size())
            for i in range(0,32): #range for beam channels
                print("ev_opflash_beam[0]: ", ev_opflash_beam[0].PE(i), " ")
                flash.append( ev_opflash_beam[0].PE(i) )

        flash_np = np.array(flash)

        ##print("total PE: ", ev_opflash_cosmic[0].TotalPE())
        ##print("nOpDets(): ", ev_opflash_cosmic[0].nOpDets())
        
        #opflash = ev_opflash_cosmic.at(0)
        #opflash_array = np.array(opflash)
        #print("opflash_array: ", opflash_array)
        #c.append(opflash_array)
        #print("c: ", c)

        ##opflashTree.GetEntry(entry)

        # will loop through to create a batch this many times                                                                                                                     
        ##nentries = 1 # how many batches                                                                                                                                           
        ##batchsize = 8 # how many inside a batch                                                                                                                                   

        #dataloader = larflow.lightmodel.DataLoader(input_files)
        #dataloader.load_entry(ientry)

        #for ientry in range(nentries):
        #data_dict = dataloader.getTrainingDataBatch(batchsize)
        #if data_dict:
                #print("entry[",ientry,"] voxel entries: ",data_dict["coord_t"].shape)

        #print("larvoxeltrainingdata.coord_v",larvoxeltrainingdata.coord_v)
        #coord_array = np.array(larvoxeltrainingdata.coord_v)
        #print("coord_array: ", coord_array)
        #coord_t = torch.from_numpy(np.array(larvoxeltrainingdata.coord_v.data))
        #print("coord_t: ", coord_t)

        coord_v = larvoxeltrainingdata.coord_v
        feat_v = larvoxeltrainingdata.feat_v
        ####flashTick = larvoxeltrainingdata.flashTick
        ####print("FLASHTICK HERE IS: ", flashTick)

        print("coord_v.size()", coord_v.size())
        print("feat_v.size()", feat_v.size())

        print("coord_v", coord_v)
        print("coord_v.size()", coord_v.size())
        coord_np = []
        for i in range( coord_v.size() ):
            print("i: ", i)
            coord = coord_v.at(i)
            coord_np = np.copy( coord.tonumpy() )
            print("This is coord_np: ", coord_np)
            print("This is coord_np.shape: ", coord_np.shape)
            print("This is type(coord_np): ", type(coord_np) )
            coord_t = torch.from_numpy(coord_np)
            #coord_t_v.append(coord_t)
        flash_t = torch.from_numpy(flash_np)

        stdFlash = torch.std(flash_t)
        print("Std value of flash: ", stdFlash )

        print("coord_t: ", coord_t)
        print("coord_t.size()", coord_t.size())

        print("flash_t: ", flash_t)
        print("flash_t.size()", flash_t.size())


        ####SA = calc_solid_angle(coord_np, flashTick)
        ####print("This is from the SA function: ", SA)

        ####SA_np = np.array(SA)

        ####SA_np = SA_np.astype(float)

        ####print("This is the shape of the SA output: ", SA_np.shape)
        ####print("Need to take the transpose so shape is (N,32).")
        ####SA_transpose = np.transpose(SA_np)
        ####print("Shape is now: ", SA_transpose.shape )

        

        ####SA_t = torch.from_numpy( SA_transpose )
        ####print("This is SA_t!!", SA_t)
        ####np.savetxt('SA_LMDATALOADER_020424_voxelsize5_entry%d_allPMTs.csv' % (entry), SA_t, delimiter=',')

        print("feat_v.size()", feat_v.size() )

        for i in range(feat_v.size()):
            feat = feat_v.at(i)
            feat_np = np.copy( feat.tonumpy() )
            feat_t = torch.from_numpy(feat_np)

        print("feat_t: ", feat_t)
        print("feat_t.size()", feat_t.size())
        #print("coord_t_v ", coord_t_v)
        meanFeat = torch.mean(feat_t)
        stdFeat = torch.std(feat_t)
        print("Mean value of ADC: ", meanFeat )
        print("Std value of ADC: ", stdFeat )

        #vector_b = np.array(larvoxeltrainingdata.coord_v)
        #vector_a.append(vector_b)
        #print("vector a: ", vector_a)
        #print("vector b: ", vector_b)

        #numpy_v = [np.asarray(e) for e in larvoxeltrainingdata.coord_v]
        #print("numpy_v", numpy_v)

        #coord_t = torch.from_numpy(np.array(data_dict["coord_t"]))
        #feat_t = torch.from_numpy(np.array(data_dict["feat_t"]))

        ## Do I subtract mean, divid by stdev?

        #data = {"coord_t":coord_t, "feat_t":feat_t, "flash_t":flash_t}

        feat_t = feat_t - meanFeat
        feat_t = feat_t / stdFeat

        #flash_t = flash_t / stdFlash
        flash_t = flash_t / 4000.

        print("This is the new normalized feature tensor: ", feat_t)
        print("This is the new normalized flash tensor: ", flash_t)

        coordBatch.append(coord_t)
        featBatch.append(feat_t)
        truthBatch.append(flash_t)

    return coordBatch, featBatch, truthBatch #SA_t

# Test it out!
if __name__=="__main__":

    start_time = time.time()

    print("hi")

    #finalSAList = []
    #input_file = "100events_062323_FMDATA_coords_withErrorFlags_100Events.root"
    input_file = "testfm_010724_FMDATA_coords_withErrorFlags_100Events_voxelsize5cm_010724.root"
    opfile = "100events_062323_FMDATA_filtered_MCTracks_opflash.root"
    num = 0
    batchnum = 2

    for i in range(0,1):
        coordsBatch, featsBatch, labelBatch = load_lm_data(input_file, opfile, num, batchnum)

    print("This is what got loaded in:")
    print("coords batch: ", coordsBatch)
    print("feats batch: ", featsBatch)
    print("truth/label/flash batch: ", labelBatch)

    print("Grabbing index of 1 from batch:")
    print("labelBatch[1]: ", labelBatch[1])




    ##np.savetxt('SA_012524_test5voxelsize.csv', SA, delimiter=',')
        
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:",execution_time, " seconds")