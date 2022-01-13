from __future__ import print_function
import os,sys,argparse,json
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet/larvoxel/prepdata/")

parser = argparse.ArgumentParser(description='Run Prep larmatch data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('-i','--fileid',required=True,type=int,help="File ID number to run")
parser.add_argument('input_list',type=str,help="json file that collates triplet and mcinfo files")

args = parser.parse_args()

from ctypes import c_int
from array import array
import numpy as np
import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow

if not os.path.exists(args.input_list):
    print("Could not fine input list: ",args.input_list)
    sys.exit(0)

# FILTER OUT ABS(PDG CODES)
ALLOWED_PDG_CODES = [11,13,22,211,2212,321]
PDG_NAMES = ["electron","muon","gamma","pion","proton","kaon"]
CODE_TO_NAMES = {11:"electron",
                 13:"muon",
                 22:"gamma",
                 211:"pion",
                 2212:"proton",
                 321:"kaon"}

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

# c++ class that provides voxels and labels using data in the labeler class
voxelsize_cm = 0.3 # 3 mm, the wire-pitch
voxelizer = larflow.voxelizer.VoxelizeTriplets()
voxelizer.set_voxel_size_cm( voxelsize_cm )

# Get the number of entries in the tree
nentries = labeler.GetEntries()
ll_nentries = io.get_entries()
if nentries!=ll_nentries:
    raise ValueError("Mismatch in triplet and larlite entries: labeler=%d larlite=%d"%(nentries,ll_nentries))
print("Input ready to go!")

# we're going to loop through the larlite file to make a rse to entry map
# do we need to?
rse_map = {}
for i in range(ll_nentries):
    io.go_to(i)
    rse = ( int(io.run_id()),int(io.subrun_id()),int(io.event_id()) )
    rse_map[rse] = i
    
# output container for data
# we split the data into classes. This will make it easier to sample in a balance way.
outfiles = {}
outtrees = {}
vardicts = {}
for pdgname in PDG_NAMES:
    if len(args.output)>len(".root") and args.output[-len(".root"):]==".root":
        fname = args.output[:-len(".root")]+"_"+pdgname+".root"
        print(pdgname," filename: ",fname)
        outfile = rt.TFile(fname,"recreate")
    else:
        outfile = rt.TFile(args.output+"_"+pdgname+".root","recreate")
    outfile.cd()
        
    outtree = rt.TTree("larvoxelpidtrainingdata","LArMatch training data")
    print(pdgname," tree: ",outtree)

    # Run, subrun, event
    treevars = dict(run    = array('i',[0]),
                    subrun = array('i',[0]),
                    event  = array('i',[0]),
                    pid    = array('i',[0]),
                    shower0_or_track1 = array('i',[0]),
                    trackid = array('i',[0]),
                    ke      = array('f',[0]),

                    # 3D Wire Plane Images, as sparse matrices
                    coord_v = std.vector("larcv::NumpyArrayInt")(),
                    feat_v  = std.vector("larcv::NumpyArrayFloat")(),
    
                    # class label
                    pid_v = std.vector("larcv::NumpyArrayInt")(),
    
                    # meta data: true position
                    detpos_v = std.vector("larcv::NumpyArrayFloat")(),
    
                    # meta data: true momentum
                    mom_v = std.vector("larcv::NumpyArrayFloat")())

    outtree.Branch("run",    treevars["run"],    "run/I")
    outtree.Branch("subrun", treevars["subrun"], "subrun/I")
    outtree.Branch("event",  treevars["event"],  "event/I")
    outtree.Branch("pid",    treevars["pid"],    "pid/I")
    outtree.Branch("shower0_or_track1", treevars["shower0_or_track1"], "shower0_or_track1/I")
    outtree.Branch("geant_trackid", treevars["trackid"], "geant_trackid/I")
    outtree.Branch("ke",       treevars["ke"],"ke/F")
    outtree.Branch("coord_v",  treevars["coord_v"])
    outtree.Branch("feat_v",   treevars["feat_v"])
    outtree.Branch("pid_v",    treevars["pid_v"])
    outtree.Branch("detpos_v", treevars["detpos_v"])
    outtree.Branch("mom_v",    treevars["mom_v"])

    outtrees[pdgname] = outtree
    outfiles[pdgname] = outfile
    vardicts[pdgname] = treevars

for ientry in range(nentries):

    print("========== EVENT %d =============="%(ientry))
    
    # Get the first entry (or row) in the tree (i.e. table)
    labeler.load_entry(ientry)
    trip_rse = ( int(labeler.run()), int(labeler.subrun()), int(labeler.event()) )

    if trip_rse not in rse_map:
        raise ValueError("triplet rse not in larlite RSE",trip_rse)

    llentry = rse_map[trip_rse]
    io.go_to(llentry)

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

    for pdgname,vardict in vardicts.items():
        vardict["run"][0]    = labeler.run()
        vardict["subrun"][0] = labeler.subrun()
        vardict["event"][0]  = labeler.event()

    ev_mctrack  = io.get_data( larlite.data.kMCTrack,  "mcreco" )
    ev_mcshower = io.get_data( larlite.data.kMCShower, "mcreco" )

    for (ev_data,s_t) in [(ev_mctrack,1),(ev_mcshower,0)]:
        for ipart in range(ev_data.size()):
            mcpart  = ev_data.at(ipart)
            pdgcode = mcpart.PdgCode()
            if abs(pdgcode) not in ALLOWED_PDG_CODES:
                continue
            if mcpart.Origin()!=1:
                continue
            
            mom = mcpart.Start().Momentum()
            mass = mom.Mag()
            E = mom.E()
            p = mom.P()
            partke = E-mass
            if partke<20:
                continue

            trackid = mcpart.TrackID()
            if trackid not in voxdata["voxinstance2id"]:
                continue

            iid = voxdata["voxinstance2id"][trackid]
            #print("instanceid=",iid)
            indexmatch = voxdata["voxinstance"]==iid
            nvoxels = indexmatch.sum()

            if nvoxels<10:
                continue
            
            print("number of voxels: ",indexmatch.sum())

            # gonna save this particle
            tree = outtrees[ CODE_TO_NAMES[abs(pdgcode)] ]
            print("tree: ",tree)
            vardict = vardicts[ CODE_TO_NAMES[abs(pdgcode)] ]
            
            # clear entry data containers
            vardict["coord_v"].clear()
            vardict["feat_v"].clear()
            vardict["pid_v"].clear()
            vardict["detpos_v"].clear()
            vardict["mom_v"].clear()
            
            vardict["shower0_or_track1"][0] = s_t
            vardict["trackid"][0] = mcpart.TrackID()
            vardict["pid"][0] = mcpart.PdgCode()
            vardict["ke"][0] = partke
            
            pos = mcpart.Start().Position()
            
            print("particle[%d] isshower=%d pid=%d"%(trackid,s_t,pdgcode)," mass=",mass," E=",E," P2=",p," KE=",partke)
            
            iicoord = voxdata["voxcoord"][indexmatch[:],:]
            iifeat  = voxdata["voxfeat"][indexmatch[:],:]
            print("iicoord=",iicoord.shape," iifeat=",iifeat.shape)
            np_pid = np.ones( 1, dtype=np.int32 )*pdgcode
            
            # store data
            np_mom = np.ones(4,dtype=np.float32)
            np_pos = np.ones(4,dtype=np.float32)
            for i in range(4):
                np_mom[i] = mom(i)
                np_pos[i] = pos(i)

            x_mom = larcv.NumpyArrayFloat()
            x_mom.store( np_mom.astype(np.float32) )
            x_pos = larcv.NumpyArrayFloat()
            x_pos.store( np_pos.astype(np.float32) )
            x_pid = larcv.NumpyArrayInt()
            x_pid.store( np_pid.astype(np.int32) )
            x_coord = larcv.NumpyArrayInt()
            x_coord.store( iicoord.astype(np.int32) )
            x_feat  = larcv.NumpyArrayFloat()
            x_feat.store( iifeat.astype(np.float32) )
            #print(x_coord)

            vardict["coord_v"].push_back( x_coord )
            vardict["feat_v"].push_back( x_feat )
            vardict["pid_v"].push_back( x_pid )
            vardict["detpos_v"].push_back( x_pos )
            vardict["mom_v"].push_back( x_mom )

            print("coord_v.size: ",vardict["coord_v"].size())
                
            tree.Fill()
            
    if False and ientry>=4:
        # For debug
        break

for name,f in outfiles.items():
    print("Writing file for ",name)
    f.Write()
    

    
