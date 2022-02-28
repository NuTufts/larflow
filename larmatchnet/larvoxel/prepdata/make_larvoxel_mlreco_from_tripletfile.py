from __future__ import print_function
import os,sys,argparse,json
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet/larvoxel/prepdata/")

parser = argparse.ArgumentParser(description='Run Prep larmatch data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('-i','--fileid',required=True,type=int,help="File ID number to run")
parser.add_argument('input_list',type=str,help="json file that collates triplet and mcinfo files")

args = parser.parse_args()

"""
This script takes the labels prepared in the MicroBooNE Triplet files 
and prepares them in a format to use in SLAC lartpc_mlreco code.

output schema needed

    schema:
      input_data:
        - parse_sparse3d_scn: which returns
          voxels: numpy array(int32) with shape (N,3) Coordinates
          data: numpy array(float32) with shape (N,C) Pixel values/channels
        - sparse3d_pcluster
      segment_label:
        - parse_sparse3d_scn: which returns
          voxels: numpy array(int32) with shape (N,3) Coordinates
          data: numpy array(float32) with shape (N,C) Pixel values/channels
        - sparse3d_pcluster_semantics 
      particles_label:
        - parse_particle_points_with_tagging: which returns
          np_voxels: np.ndarray
                     a numpy array with the shape (N,3) where 3 represents (x,y,z) coordinate
          np_values: np.ndarray
                     a numpy array with the shape (N, 3) where 3 
                        [0] point type: (0=proton,
                                         1=(not gamma and not electron and not proton), 
                                         2=gamma,
                                         3=shower (primary,nCapture,conv), 
                                         4=delta (muIoni,hIoni))
                        [1] part_index: (indexing label)
                        [2] start or end (Tracks only: class 0(protons) or 1(tracks))
        - sparse3d_pcluster
        - particle_corrected
"""

from ctypes import c_int
from array import array
import numpy as np
import ROOT as rt
from ROOT import std
from larlite import larlite
from ROOT import larutil as larutil
from larcv import larcv
from ublarcvapp import ublarcvapp
from larflow import larflow
rt.gSystem.Load('./lib/libSmallClusterRemoval.so')

from ROOT import larvoxelprepdata as larvoxelprepdata

if not os.path.exists(args.input_list):
    print("Could not fine input list: ",args.input_list)
    sys.exit(0)

# FILTER OUT ABS(PDG CODES)
ALLOWED_PDG_CODES = [11,13,22,211,2212,321]
PDG_NAMES = ["electron","muon","gamma","pion","proton","kaon"] # These are larmatch codes
CODE_TO_NAMES = {11:"electron",
                 13:"muon",
                 22:"gamma",
                 211:"pion",
                 2212:"proton",
                 321:"kaon"}

# MLRECO CODES
type_labels = {
    22: 0,  # photon
    11: 1,  # e-
    -11: 1, # e+
    13: 2,  # mu-
    -13: 2, # mu+
    211: 3, # pi+
    -211: 3, # pi-
    2212: 4, # protons
}


# LOAD JSON FILE
f = open(args.input_list,'r')
j = json.load(f)

# This is the file id to run
FILEIDS=[args.fileid]

input_triplet_v = []
input_mcinfo_v = []
for fid in FILEIDS:
    data = j["%d"%(fid)]
    tripletfile = data["triplet"]
    if not os.path.exists(tripletfile):
        print("input file given does not exist: ",tripletfile)
        sys.exit(0)
    input_triplet_v.append(tripletfile)
    for mcinfofile in data["mcinfo"]:
        if not os.path.exists(mcinfofile):
            print("mcinfo file given does not exist: ",mcinfofile)
            sys.exit(0)
        input_mcinfo_v.append(mcinfofile)    
    
print("Loaded %d larmatch triplet files to process"%(len(input_triplet_v)))

io = larlite.storage_manager( larlite.storage_manager.kREAD )
io.set_data_to_read( larlite.data.kMCTrack,  "mcreco" )
io.set_data_to_read( larlite.data.kMCShower, "mcreco" )
io.set_data_to_read( larlite.data.kMCTruth,  "generator" )
for f in input_mcinfo_v:
    io.add_in_filename( f )
io.open()

# c++ class that appends keypoint and ssnet labels to triplet truth information
f_v = std.vector("std::string")()
for f in input_triplet_v:
    f_v.push_back(f)
labeler = larflow.keypoints.LoaderKeypointData( f_v )

# sce correction
sce = larutil.SpaceChargeMicroBooNE()

# c++ class that provides voxels and labels using data in the labeler class
voxelsize_cm = 0.3 # 3 mm, the wire-pitch
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelizer.set_voxel_size_cm( voxelsize_cm )

# small cluster removal algorithm
remover = larvoxelprepdata.SmallClusterRemoval()
voxel_threshold = 10
charge_threshold = 100.0
print(remover)

# Get the number of entries in the tree
nentries = labeler.GetEntries()
ll_nentries = io.get_entries()
if nentries!=ll_nentries and True:
    raise ValueError("Mismatch in triplet and larlite entries: labeler=%d larlite=%d"%(nentries,ll_nentries))
print("Input ready to go!")
if ll_nentries<nentries:
    nentries = ll_nentries

# we're going to loop through the larlite file to make a rse to entry map
# do we need to?
rse_map = {}
for i in range(ll_nentries):
    io.go_to(i)
    rse = ( int(io.run_id()),int(io.subrun_id()),int(io.event_id()) )
    rse_map[rse] = i
    
# output file and trees 
outfile = rt.TFile(args.output,"recreate")
outfile.cd()
        
outtree = rt.TTree("larvoxel_mlreco_tree","lartpc mlreco training data")

# Run, subrun, event
vardict = dict(run    = array('i',[0]), # run
               subrun = array('i',[0]), # subrun
               event  = array('i',[0]), # event
               nupid  = array('i',[0]), # nu PID
               enu    = array('f',[0]), # enu
               evis   = array('f',[0]), # evis
               evistr = array('f',[0]), # evis tracks
               evissh = array('f',[0]), # evis showers
               vtxsce = array('f',[0.0,0.0,0.0]), # nu vtx position
               vtxvox = array('f',[0.0,0.0,0.0]), # nu vtx position               
               nvoxels = array('i',[0]), # number of voxels
               
               # 3D Wire Plane Images, as sparse matrices
               coord_v = std.vector("larcv::NumpyArrayInt")(),
               feat_v  = std.vector("larcv::NumpyArrayFloat")(),
               
               # ssnet class label
               ssnetpid_v = std.vector("larcv::NumpyArrayInt")(),
               ssnetweight_v = std.vector("larcv::NumpyArrayFloat")(),
               
               # particle keypoint pos (x,y,z)
               partpos_xyz_v = std.vector("larcv::NumpyArrayFloat")(),
               # particle keypoint pos in voxels
               partpos_vox_v = std.vector("larcv::NumpyArrayFloat")(),                
               
               # particle keypoint labels
               partlabel_v = std.vector("larcv::NumpyArrayInt")())

print("Make output tree branches")
print("=========================")
for name,var in vardict.items():
    if type(var) is array:
        branch_specs = name+"_branch"
        if len(var)>1:
            branch_specs += "[%d]"%(len(var))
        branch_specs += "/I" if var.typecode=='i' else "/F"
        print(name," ",branch_specs)        
        outtree.Branch(name+"_branch", var, branch_specs)
    else:
        print(name," numpy array")
        outtree.Branch(name+"_branch",var)
print("=========================")
nentries = 8

for ientry in range(nentries):

    print("========== EVENT %d =============="%(ientry))
    
    # Get the first entry (or row) in the tree (i.e. table)
    labeler.load_entry(ientry)
    trip_rse = ( int(labeler.run()), int(labeler.subrun()), int(labeler.event()) )

    if trip_rse not in rse_map:
        raise ValueError("triplet rse not in larlite RSE",trip_rse)

    llentry = rse_map[trip_rse]
    io.go_to(llentry)

    mcpg = ublarcvapp.mctools.MCPixelPGraph()
    mcpg.buildgraphonly( io )

    voxelizer.make_voxeldata( labeler.triplet_v[0] )    
    voxdata = voxelizer.get_full_voxel_labelset_dict( labeler )

    ll_rse = ( int(io.run_id()), int(io.subrun_id()), int(io.event_id()) )
    print("triplet rse: ",trip_rse)
    print("larlite: ",ll_rse)

    if ll_rse != trip_rse:
        raise ValueError("larlite and triplet RSE mismatch! triplet=",trip_rse," larlite=",ll_rse)
        
    print("voxdata keys: ",voxdata.keys())
    #print("voxinstance2id")
    #print(voxdata["voxinstance2id"])

    vardict["run"][0]    = labeler.run()
    vardict["subrun"][0] = labeler.subrun()
    vardict["event"][0]  = labeler.event()

    ev_mctruth = io.get_data( larlite.data.kMCTruth, "generator" )
    mctruth = ev_mctruth.at(0)
    mcnu = mctruth.GetNeutrino()

    # get nu interaciton meta-data (for analysis later)
    vardict["nupid"][0] = mcnu.Nu().PdgCode()
    vardict["enu"][0] = mcnu.Nu().Momentum().E()
    # blank out vars
    vardict["evis"][0] = 0.0
    vardict["evistr"][0] = 0.0
    vardict["evissh"][0] = 0.0
    vardict["nvoxels"][0] = voxdata["voxcoord"].shape[0]
    for i in range(3):
        vardict["vtxsce"][i] = 0.0
        vardict["vtxvox"][i] = 0.0        
    for i in ["coord_v","feat_v","ssnetpid_v","ssnetweight_v","partpos_xyz_v","partpos_vox_v","partlabel_v"]:
        vardict[i].clear()

    # keypoints have already been parsed in the keypoint class
    kp_pos_v = []
    kp_xyz_v  = labeler.get_keypoint_pos()
    kp_type_v = labeler.get_keypoint_types()
    kp_pidtid_v = labeler.get_keypoint_pdg_and_trackid()
    for n in range(kp_xyz_v.size()):
        kpxyz = kp_xyz_v.at(n)
        xyz = std.vector("float")(3)
        for v in range(3):
            xyz[v] = kpxyz[v]
        goodkp = True
        try:
            voxpos = voxelizer.get_voxel_indices( xyz )
        except:
            print("keypoint outside of voxel array")
            goodkp = False

        if kp_type_v.at(n)==0 and goodkp:
            # nu-vtx keypoint
            for v in range(3):            
                vardict["vtxsce"][v] = kpxyz[v]
                vardict["vtxvox"][v] = voxpos[v]
            continue

        if not goodkp:
            continue

        # check the number of voxels associated to this keypoint. Cut it if too small.

        kppt  = kp_pidtid_v.at(n)
        kp_pos_v.append( (kpxyz[0], kpxyz[1], kpxyz[2], kp_type_v.at(n), kppt[0], kppt[1], voxpos[0], voxpos[1], voxpos[2] ) )

    # collect particle info that we need
    # for each particle, we need
    #   (x-start, y-start, z-start)
    #   point type
    #   part_index (re-index)
    #   start or end for tracks    
    np_ppn_pos = np.zeros( (len(kp_pos_v),3), dtype=np.float32 )
    np_ppn_vox = np.zeros( (len(kp_pos_v),3), dtype=np.float32 )    
    np_ppn_tag = np.zeros( (len(kp_pos_v),3), dtype=np.int32 )
    print("KP_POS_V -------")
    for i,kpinfo in enumerate(kp_pos_v):
        print(kpinfo)
        for v in range(3):
            np_ppn_pos[i,v] = kpinfo[v]
            np_ppn_vox[i,v] = kpinfo[6+v]

        if kpinfo[4]==2212:
            np_ppn_tag[i,0] = 0 # proton
            if kpinfo[3]==1:
                np_ppn_tag[i,2] = 0 # track start
            else:
                np_ppn_tag[i,2] = 1 # track end
        elif abs(kpinfo[4])!=11 and abs(kpinfo[4])!=22:
            np_ppn_tag[i,0] = 1 # track (not proton)
            if kpinfo[3]==1:
                np_ppn_tag[i,2] = 0 # track start
            else:
                np_ppn_tag[i,2] = 1 # track end            
        elif kpinfo[4]==22:
            np_ppn_tag[i,0] = 2 # gamma
        else:
            # shower
            if kpinfo[3]==5:
                np_ppn_tag[i,0] = 4 # delta
            else:
                np_ppn_tag[i,0] = 3 # shower
                
        np_ppn_tag[i,1] = kpinfo[5]
        print("-------------")

    for t in range(5):
        print("KP TAGS [%d] n=%d"%(t,np.sum(np_ppn_tag[:,0]==t)))
        
    # collect more meta data
    ev_mctrack  = io.get_data( larlite.data.kMCTrack,  "mcreco" )
    ev_mcshower = io.get_data( larlite.data.kMCShower, "mcreco" )

    for x in ["evis","evistr","evissh"]:
        vardict[x][0] = 0.0
    for (ev_data,s_t) in [(ev_mctrack,1),(ev_mcshower,0)]:
        for ipart in range(ev_data.size()):
            mcpart  = ev_data.at(ipart)
            if mcpart.Origin()!=1:
                continue
            if mcpart.TrackID()!=mcpart.MotherTrackID():
                continue
            mom4 = mcpart.Start().Momentum()
            ke = mom4.E()-mom4.Mag()
            vardict["evis"][0] += ke
            pdgcode = abs(mcpart.PdgCode())
            if pdgcode in [11,22]:
                vardict["evissh"][0] += ke
            else:
                vardict["evistr"][0] += ke

    # ssnet relabel
    ssnet_unique = np.unique( voxdata["ssnet_labels"] )
    ssnet = voxdata["ssnet_labels"]
    print("ssnet unique: ",ssnet_unique)
    ssnet[ ssnet==0 ] = 9  # ghost
    ssnet[ ssnet==2 ] = 10 # gamma    
    ssnet[ ssnet==1 ] = 11 # electron
    ssnet[ ssnet==3 ] = 12 # muon
    ssnet[ ssnet==4 ] = 13 # pion
    ssnet[ ssnet==5 ] = 14 # proton
    ssnet[ ssnet==6 ] = 13 # kaon+other -> pion
    ssnet -= 10 # reindex to [-1,0,1,2,3,4]
    
    # save numpy arrays
    x_coord = larcv.NumpyArrayInt()
    x_coord.store( voxdata["voxcoord"].astype(np.int32) )
    x_feat  = larcv.NumpyArrayFloat()
    x_feat.store( voxdata["voxfeat"].astype(np.float32) )
    x_label = larcv.NumpyArrayInt()
    x_label.store( ssnet.astype(np.int32) )
    x_weight = larcv.NumpyArrayFloat()
    x_weight.store( voxdata["ssnet_weights"].astype(np.float32) )

    # ppn arrays
    x_ppn_pos = larcv.NumpyArrayFloat()
    x_ppn_pos.store( np_ppn_pos )
    x_ppn_vox = larcv.NumpyArrayFloat()
    x_ppn_vox.store( np_ppn_vox )
    x_ppn_tag = larcv.NumpyArrayInt()
    x_ppn_tag.store( np_ppn_tag )
    
    vardict["coord_v"].push_back( x_coord )
    vardict["feat_v"].push_back( x_feat )
    vardict["ssnetpid_v"].push_back( x_label )
    vardict["ssnetweight_v"].push_back( x_weight )
    vardict["partpos_xyz_v"].push_back( x_ppn_pos )
    vardict["partpos_vox_v"].push_back( x_ppn_vox )
    vardict["partlabel_v"].push_back( x_ppn_tag )
            
    outtree.Fill()
            
    if False and ientry>=4:
        # For debug
        break


print("Writing file")
outfile.Write()
    

    
