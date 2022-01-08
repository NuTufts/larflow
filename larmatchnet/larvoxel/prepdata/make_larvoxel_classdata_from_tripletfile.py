from __future__ import print_function
import os,sys,argparse,json
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")

parser = argparse.ArgumentParser(description='Run Prep larmatch data')
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")
parser.add_argument('-s','--single',default=False,action='store_true',help='If flag given, input_list argument is interpretted as a triplet file')
parser.add_argument('input_list',type=str,help="text file with paths to larmatch triplet files to distill")

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
#if os.path.exists(args.output):
#    print("output file already exists. do not overwrite")
#    sys.exit(0)


ALLOWED_PDG_CODES = [11,13,22,211,2212,321]

f = open(args.input_list,'r')
j = json.load(f)
print(type(j))

FILEIDS=[0]

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

# output container for data
outfile = rt.TFile(args.output,"recreate")
outtree = rt.TTree("larvoxelpidtrainingdata","LArMatch training data")

# Run, subrun, event
run    = array('i',[0])
subrun = array('i',[0])
event  = array('i',[0])
pid    = array('i',[0])
shower0_or_track1 = array('i',[0])
trackid = array('i',[0])
ke      = array('f',[0])

# 3D Wire Plane Images, as sparse matrices
coord_v = std.vector("larcv::NumpyArrayInt")()
feat_v  = std.vector("larcv::NumpyArrayFloat")()

# class label
pid_v = std.vector("larcv::NumpyArrayInt")()

# meta data: true position
detpos_v = std.vector("larcv::NumpyArrayFloat")()

# meta data: true momentum
mom_v = std.vector("larcv::NumpyArrayFloat")()

outtree.Branch("run", run, "run/I")
outtree.Branch("subrun", subrun, "subrun/I")
outtree.Branch("event",  event,  "event/I")
outtree.Branch("pid",pid,"pid/I")
outtree.Branch("shower0_or_track1", shower0_or_track1, "shower0_or_track1/I")
outtree.Branch("geant_trackid", trackid, "geant_trackid/I")
outtree.Branch("ke",ke,"ke/F")
outtree.Branch("coord_v",coord_v)
outtree.Branch("feat_v", feat_v)
outtree.Branch("pid_v",pid_v)
outtree.Branch("detpos_v", detpos_v)
outtree.Branch("mom_v", mom_v)

for ientry in range(nentries):

    print("========== EVENT %d =============="%(ientry))
    
    # Get the first entry (or row) in the tree (i.e. table)
    labeler.load_entry(ientry)
    io.go_to(ientry)

    voxelizer.make_voxeldata( labeler.triplet_v[0] )    
    voxdata = voxelizer.get_full_voxel_labelset_dict( labeler )

    print("triplet rse: ",(labeler.run(),labeler.subrun(),labeler.event()))
    print("larlite: ",(io.run_id(),io.subrun_id(),io.event_id()))
    print("voxdata keys: ",voxdata.keys())
    #print("voxinstance2id")
    #print(voxdata["voxinstance2id"])

    run[0]    = labeler.run()
    subrun[0] = labeler.subrun()
    event[0]  = labeler.event()

    ev_mctrack  = io.get_data( larlite.data.kMCTrack,  "mcreco" )
    ev_mcshower = io.get_data( larlite.data.kMCShower, "mcreco" )

    for (ev_data,s_t) in [(ev_mctrack,1),(ev_mcshower,0)]:
        for ipart in range(ev_data.size()):
            
            # clear entry data containers
            coord_v.clear()
            feat_v.clear()
            pid_v.clear()
            detpos_v.clear()
            mom_v.clear()
            
            shower0_or_track1[0] = s_t
            mcpart = ev_data.at(ipart)
            if mcpart.Origin()!=1:
                continue
            trackid[0] = mcpart.TrackID()
            pid[0] = mcpart.PdgCode()
            pos = mcpart.Start().Position()
            mom = mcpart.Start().Momentum()
            mass = mom.Mag()
            E = mom.E()
            p = mom.P()
            ke[0] = E-mass
            if ke[0]<20:
                continue
            if abs(pid[0]) not in ALLOWED_PDG_CODES:
                continue
            
            print("particle[%d] isshower=%d pid=%d"%(trackid[0],shower0_or_track1[0],pid[0])," mass=",mass," E=",E," P2=",p," KE=",ke[0])
            if trackid[0] in voxdata["voxinstance2id"]:
                iid = voxdata["voxinstance2id"][trackid[0]]
                print("instanceid=",iid)
                indexmatch = voxdata["voxinstance"]==iid
                print("number of voxels: ",indexmatch.sum())
                iicoord = voxdata["voxcoord"][indexmatch[:],:]
                iifeat  = voxdata["voxfeat"][indexmatch[:],:]
                print("iicoord=",iicoord.shape," iifeat=",iifeat.shape)
                np_pid = np.ones( 1, dtype=np.int32 )*pid[0]

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

                coord_v.push_back( x_coord )
                feat_v.push_back( x_feat )
                pid_v.push_back( x_pid )
                detpos_v.push_back( x_pos )
                mom_v.push_back( x_mom )

                print("coord_v.size: ",coord_v.size())
                
                outtree.Fill()
            
    if True and ientry>=4:
        # For debug
        break


outfile.Write()
    

    
