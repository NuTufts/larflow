# Want to make 3D plot of:
#    Total charge (ADC) for 1 entry (intrxn level)
#    Total PE of that matched flash
#    Average x-position of that mctrack (raw)

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from array import array
from scipy.stats import binned_statistic_2d
from ROOT import TCanvas, TGraph2D, TH1D, TFile
from larlite import larlite
from larflow import larflow
sys.path.append("../")
from lightmodel.lm_dataset import LMDataset

if len(sys.argv) < 4:
    msg  = '\n'
    msg += "Usage 1: %s $INPUT_COORD_FILE $INPUT_MC_FILE $ENTRY\n" % sys.argv[0]
    msg += '\n'
    sys.stderr.write(msg)
    sys.exit(1)

entry = int(sys.argv[3])
    
ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll2 = larlite.storage_manager( larlite.storage_manager.kREAD )
#ioll.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
ioll.add_in_filename( sys.argv[2] )
#ioll2.add_in_filename( sys.argv[1] )
ioll.open()
#ioll2.open()

ll_nentries = ioll.get_entries()
print("ll_nentries: ",ll_nentries)
#ioll.go_to( entry )

# arrays for our 3D plot!
totalADC_v = []
meanX_v = []
totalPE_v = []

dataset = LMDataset( filelist=["missingChargeFlag_100Events_040323_TEST.root"], is_voxeldata=True, random_access=False )

#tfile = TFile(sys.argv[1],'open')
#preppedTree  = tfile.Get('larvoxeltrainingdata')
#adc_v = []
#preppedTree.GetEntry(int(sys.argv[3]))
#print("preppedTree.feat_v: ", type(preppedTree.feat_v))

#nentries_adc = len(dataset)
#print("NENTRIES_ADC: ",nentries_adc)

loader = torch.utils.data.DataLoader( dataset, batch_size=1, collate_fn=LMDataset.collate_fn )
niter = ll_nentries

for iiter in range(niter):
    print("====================================")
    #for ib,data in enumerate(batch):                                                                                                                                                   
    print("ITER[%d]"%(iiter))

    batch = next(iter(loader))
    print("batch keys: ",batch.keys())

    coords = batch["voxcoord"]
    feats  = batch["voxfeat"]
    print("coords: ", coords)
    print("feats: ", feats)

    # for now, we are going to take just the U plane's 
    # value of the ADC charge
    featsU = batch["voxfeat"][:,0]
    print("featsU: ", featsU)

    totalADC = np.sum(featsU)
    print("totalADC: ", totalADC)
    totalADC_v.append(totalADC)

    print("====================================")

##batch = list(iter(loader))[sys.argv[3]]

# Get entry data
#for iientry in range(niter):
#    batch = next(iter(loader))
##nvoxels = batch[0]["voxcoord"].shape[0]
# We need to retrieved the 3d positions
##pos3d = batch[0]["voxcoord"].astype(np.float64)*0.3
#pos3d[:,1] -= 117.0
##print(pos3d.shape)

for entry in range(ll_nentries):
    ioll.go_to( entry )
    print("THIS IS ENTRY: ", entry)
    ev_mctrack = ioll.get_data(larlite.data.kMCTrack,"mcreco")
    ev_mcshower = ioll.get_data(larlite.data.kMCShower, "mcreco")
    ev_opflash_cosmic = ioll.get_data(larlite.data.kOpFlash, "simpleFlashCosmic")
    ev_opflash_beam = ioll.get_data(larlite.data.kOpFlash, "simpleFlashBeam")
    print("Number of tracks in event: ", ev_mctrack.size() )
    print("sizeof ev_opflash_cosmic: ",len(ev_opflash_cosmic))
    print("sizeof ev_opflash_beam: ",len(ev_opflash_beam))
    if len(ev_opflash_cosmic)==1:
        #print("ev_opflash_cosmic.PE(): ", ev_opflash_cosmic.PE() )
        opflash_v = ev_opflash_cosmic.at(0)
        totalPE = opflash_v.TotalPE()
        print("totalPE: ", totalPE )
        totalPE_v.append(totalPE)
    if len(ev_opflash_beam)==1:
        #print("ev_opflash_cosmic.PE(): ", ev_opflash_cosmic.PE() )
        opflash_v = ev_opflash_beam.at(0)
        totalPE = opflash_v.TotalPE()
        print("totalPE: ", totalPE )
        totalPE_v.append(totalPE)
    if len(ev_opflash_beam)==0 and len(ev_opflash_cosmic)==0:
        totalPE_v.append(0)
    
#elif len(ev_opflash_beam)==1:
#    opflash_v = lardly.data.visualize_larlite_opflash_3d( ev_opflash_beam.at(0) )

    # array for grabbing all x-positions for mctracks, mcshowers in intrxn event
    vx = array( 'd' )

    if ev_mctrack:

        print("There are ", ev_mctrack.size(), " track(s) in this intrxn")

        for i in range(ev_mctrack.size()):

            mctrack = ev_mctrack.at(i)

            for mcstep in mctrack:
                vx.append( mcstep.X() )
            print( "mcstep X values: ",mcstep.X() )

    if ev_mcshower:

        print("There are ", ev_mcshower.size(), " shower(s) in this intrxn")

        for i in range(ev_mcshower.size()):

            mcshower = ev_mcshower.at(i)

            vx.append( mcshower.Start().X() )
            vx.append( mcshower.End().X() )

    print(vx)

    # no tracks or showers
    if len(vx)==0:
        meanX_v.append(0)
        continue

    # now take mean of all the points gathered in interxn
    # this is the average x-position for the intrxn event
    mean = np.mean(vx)
    print("The mean x-position of this track is: ", mean)
    meanX_v.append(mean)
    continue

    

#c1 = TCanvas( 'c1', 'Canvas', 200, 10, 700, 500 )
#c1.SetFillColor( 42 )
#c1.SetGrid()
#corr = TH2D("corr","Total charge for 1 entry vs. Total PE of matched flash vs. Avg x pos of track",100,-115,115,100,-115,115)

print("HERE ARE THE ARRAYS TO BE PLOTTED: ")
print("totalADC_v: ", totalADC_v)
totalADC_arr = np.array(totalADC_v)
print("totalADC_arr.shape: ", totalADC_arr.shape)
print("meanX_v: ", meanX_v)
meanX_arr = np.array(meanX_v)
print("meanX_arr.shape: ", meanX_arr.shape)
print("totalPE_v: ", totalPE_v)
totalPE_arr = np.array(totalPE_v)
print("totalPE_arr.shape: ", totalPE_arr.shape)

test1 = [1,2,3,4,5,6,7]
test2 = [4,5,8,3,7,5,9]
test3 = [6,7,4,8,9,0,2]


n1 = np.array(totalADC_v)
n2 = np.array(meanX_v)
print("THIS IS n2: ", n2)
n3 = np.array(totalPE_v)

'''
# Let's scale everything
n1_mean = np.mean(n1, dtype=np.float64)
n1_std = np.std(n1, dtype=np.float64)
n1 = n1 - n1_mean
n1 = n1 / n1_std
n1 = n1*10

print("SCALED totalADC: ", n1)
'''

# don't want to scale x-position, 
# more informative to see the actual value in cm
#
#n2_mean = np.mean(n2, dtype=np.float64)
#n2_std = np.std(n2, dtype=np.float64)
#n2 = n2 - n2_mean
#n2 = n2 / n2_std
#n2 = n2*10
#
#print("SCALED meanX: ", n2)

'''
n3_mean = np.mean(n3, dtype=np.float64)
n3_std = np.std(n3, dtype=np.float64)
n3 = n3 - n3_mean
n3 = n3 / n3_std
n3 = n3*10

print("SCALED totalPE: ", n3)
'''

n1 = n1.astype(int)
np.savetxt('n1.csv', n1, delimiter=',')
n2 = n2.astype(int)
np.savetxt('n2.csv', n2, delimiter=',')
n3 = n3.astype(int)
np.savetxt('n3.csv', n3, delimiter=',')

print("type(n1)): ", type(n1))
print("n1.shape: ", n1.shape)

# Define the bin edges
xbins = np.linspace(np.min(n1), np.max(n1), 106)
ybins = np.linspace(np.min(n2), np.max(n2), 106)

# Compute the statistics for each bin
bin_means, xedges, yedges, binnumber = binned_statistic_2d(n1, n2, n3, statistic='mean', bins=[xbins, ybins])

# Create a 2D plot of the binned data                                                                                         
#plt.pcolor(xedges, yedges, bin_means.T, cmap='viridis')
#plt.colorbar()

'''
# 2D histogram of mean x-position vs. total PE                                                                                   
plt.hist2d(n3, n2)
plt.hist2d(n3, n2, bins=(50,50))
plt.colorbar()

# Set labels and title                                                                                                        
plt.xlabel('Total PE')
plt.ylabel('Mean X-Position')
plt.title('Mean X-Position vs. Total PE')

plt.savefig("x_vs_pe_50bins.jpg")
plt.show()
'''
'''
# 2D histogram of total PE vs. total ADC                                                                                  
plt.hist2d(n1, n3)
plt.hist2d(n1, n3, bins=(50, 50))
plt.colorbar()

# Set labels and title                                                                                                        
plt.xlabel('Total ADC')
plt.ylabel('Total PE')
plt.title('Total PE vs. Total ADC')

plt.savefig("pe_vs_adc_50bins.jpg")
plt.show()
'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(n1, n2, n3, c='r', marker='.')
h, yedges, zedges = np.histogram2d(n2, n3, bins=50)
h = h.transpose()
normalized_map = plt.cm.Blues(h/h.max())

yy, zz = np.meshgrid(yedges, zedges)
xpos = min(n1)-2 # Plane of histogram
xflat = np.full_like(yy, xpos) 

p = ax.plot_surface(xflat, yy, zz, facecolors=normalized_map, rstride=1, cstride=1, shade=False)
plt.savefig("3dscatter.jpg")
plt.show()

'''
plt.scatter(n3, n2)
plt.savefig("x_vs_pe_scatter.jpg")
plt.show()
'''

# Set labels and title                                                                                                        
#plt.xlabel('Total PE')
#plt.ylabel('Mean X Position')
#plt.title('Mean X-Position vs. Total PE')

# Set labels and title                                                                                                        
#plt.xlabel('Total ADC')
#plt.ylabel('Mean X Position')
#plt.title('[Scaled]: Mean X Position vs. Total ADC vs. Total PE')

#plt.scatter(test1, test2, c=test3)
#plt.savefig("output.jpg")
#plt.show()

'''
y, x = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
#nn3 = n3[:-1, :-1]

nn1, nn2 = np.meshgrid(n1, n2)

#plt.plot(test1, test2, test3)
#plt.imshow(, cmap='hot', interpolation='nearest')

n3_min, n3_max = -np.abs(n3).max(), np.abs(n3).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(nn1, nn2, n3, cmap='RdBu', vmin=n3_min, vmax=n3_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([nn1.min(), nn1.max(), nn2.min(), nn2.max()])
fig.colorbar(c, ax=ax)
plt.savefig("output.jpg")
plt.show()
'''

'''
gr = TGraph2D( len(vx), vx, vy, vz )

gr.SetLineColor( 2 )
gr.SetLineWidth( 4 )
gr.SetMarkerColor( 4 )
gr.SetMarkerStyle( 21 )
gr.SetTitle( 'MCTrack' )
gr.GetXaxis().SetTitle( 'X' )

gr.GetYaxis().SetTitle( 'Y' )
gr.GetYaxis().SetTitle( 'Z' )
gr.Draw( 'P' )


input = input("Press enter to continue...")
'''

#my_proc = larlite.ana_processor()
#my_proc.set_io_mode(larlite.storage_manager.kREAD)
#my_proc.set_ana_output_file("myGraph.root");

#event_mctracks=storage.get_data<event_mctrack>("mcreco");
