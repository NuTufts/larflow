from __future__ import print_function
import os,sys,argparse,time
sys.path.append(os.environ["LARFLOW_BASEDIR"]+"/larmatchnet")

parser = argparse.ArgumentParser(description='deploy larmatch model on microboone larcv/larlite input')
parser.add_argument('-c','--config-file',type=str,default="config.yaml",help="larmatch configuration file")
parser.add_argument('-w','--weights',required=True,type=str,help='weight file')
parser.add_argument('-p','--min-score',type=float,default=0.3,help="Minimum Score to save point [default: 0.3]")
parser.add_argument('-d','--device-name',default="cpu",type=str,help="Name of device. [default: cpu; e.g. cuda:0]")
parser.add_argument('-adc','--adc-name',default="wire",type=str,help="Name of ADC tree [default: wire]")
parser.add_argument('-savelm','--save-input-lmpoints', default=False, action='store_true', 
                        help="If flag given, save input larmatch info used to make clusters")
parser.add_argument('-saveptmc','--save-point-mctruth', default=False, action='store_true', 
                        help="If flag given, save truth labels for each spacepoint saved")
parser.add_argument('-trueedges','--save-true-edges', default=False, action='store_true', 
                        help="If flag given, determine and save the true edges for making shower clusters")
parser.add_argument('-v','--verbose',default=False,action='store_true',help='If flag given, just run 5 events for debugging')
parser.add_argument('-ilcv','--input-larcv', required=True,help="input larcv file")
parser.add_argument('-ill', '--input-larlite', required=True,help="input larlite file")
parser.add_argument('-ao', '--allow-output-overwrite', default=False, help="If flag given, allow output file to overwrite")
parser.add_argument('-tf','--tickforwards',action='store_true',default=False,help="Indicate that input larcv file is tick-forward [default: F]")
parser.add_argument('-o','--output',required=True,type=str,help="Filename stem for output files")

args = parser.parse_args()

# prepare network
import h5py
import torch
import numpy as np
import larmatch.utils.larmatchme_engine as engine
import dlshowermodel.data.define_truth_edges as truth_edge_module

DEVICE=torch.device(args.device_name)
config = engine.load_config_file(  args )
single_model    = engine.get_model( config )
# set to eval-mode for inference
single_model.eval()

# Loading weights
checkpointfile  = engine.get_weightfile( args.weights, config )
checkpoint_data = engine.load_model_weights( single_model, checkpointfile )

single_model.eval()
single_model.to(DEVICE)
print("loaded MODEL on ",DEVICE)
if args.verbose and False:
    print("MODEL")
    print("-----------------------------")
    print(single_model)
    print()
    print("=============================")
    print("Parameters")
    print("------------")
    print("------------")
    for name, par in single_model.named_parameters():
        print("---------------------------------")
        print(name," ",par.shape)
        print(par)



# Setup input and output files

input_larcv   = args.input_larcv
input_larlite = args.input_larlite
outdir = os.path.dirname( args.output )
if outdir=="":
    outdir="./"

# check file paths, input and outpiut
if not os.path.exists( input_larcv ):
    print("LARCV input file does not exist. path given: ",input_larcv)
    sys.exit(1)
if not os.path.exists( input_larlite ):
    print("larlite input file does not exist. path given: ", input_larlite)
    sys.exit(1)
if not args.allow_output_overwrite and os.path.exists( args.output ):
    print("output file exists. not allowing overwrites. path given: ",args.output)
    print("provide True to keyword argument 'allow_output_overwrite'")
    sys.exit(1)
if not os.path.exists( outdir  ):
    print("directory for output file does not exist. Given: ",outdir)
    sys.exit(1)

# setup larcv and larlite interfaces to data
# ROOT-based IO for data objects used in LArSoft (common framework used in LArTPC experiments )
import ROOT
from ROOT import std
from larcv import larcv
from larlite import larlite

ioll = larlite.storage_manager( larlite.storage_manager.kREAD )
ioll.add_in_filename( input_larlite )
ioll.open()

if args.tickforwards:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickForward )
else:
    iolcv = larcv.IOManager( larcv.IOManager.kREAD, "larcv", larcv.IOManager.kTickBackward )

iolcv.add_in_file( input_larcv )
iolcv.reverse_all_products()
iolcv.initialize()

nentries_larcv = iolcv.get_n_entries()
start_entry = 0
num_entries = nentries_larcv
print("Number of entries in file: ",nentries_larcv)
if start_entry>=nentries_larcv:
    print("Asking to start after last entry (%d) in file"%(nentries_larcv-1))
    sys.exit(1)

end_entry = start_entry + nentries_larcv
if num_entries>0:
    end_entry = start_entry + num_entries
if end_entry >= nentries_larcv:
    end_entry = nentries_larcv

print("running entries [",start_entry,",",end_entry,"]")

# event loop

# we use the LArMatchHDFWriter class to help us convert larcv/larlite data into numpy arrays
from dlshowermodel.data.larmatchhit_hdf5_writer import LArMatchHitHDF5Writer
lmwriter = LArMatchHitHDF5Writer()
num_max_spacepoints = 10000000
process_truth_labels = True
triplet_key = 'matchtriplet'

# load utils for processing points into shower clusters that we are saving
from dlshowermodel.utils.cluster_shower_points import ClusterShowerPoints

clustering_alg = ClusterShowerPoints()

output_entries = []

#for ientry in range(start_entry,end_entry):
for ientry in range(start_entry,start_entry+1):
    print("[[ RUN ENTRY %d ]]"%(ientry))

    tprep = time.time()

    ioll.go_to(ientry)
    iolcv.read_entry(ientry)

    ev_adc = iolcv.get_data( larcv.kProductImage2D, args.adc_name )
    ev_chstatus = iolcv.get_data( larcv.kProductChStatus, "wire" )
    adc_v = ev_adc.as_vector()

    # convert the data and store into self.entry_data
    lmwriter.larlite_larcv_to_hdf5_entry( ioll, iolcv, process_truth_labels, num_max_spacepoints )

    entrydata = lmwriter.entry_data.pop()
    print("processed larcv/larlite info dict keys: ",entrydata.keys())
    
    inputdata = LArMatchHitHDF5Writer.prepare_triplet_and_image_arrays_for_network( entrydata, triplet_key=triplet_key )
    # transfer keys from inputdata to entrydata
    for k,i in inputdata.items():
        entrydata[k] = i
        
    batch = [entrydata]
    
    batchsize = len(batch)
    batch_sparsetensors, batch_triplets, batch_coordqueries = \
        LArMatchHitHDF5Writer.make_batch_sparse_tensors( batch, DEVICE, triplet_key=triplet_key )
    dt_prep = time.time()-tprep

    #mcpg = ublarcvapp.mctools.MCPixelPGraph()
    #mcpg.buildgraphonly( ioll )

    with torch.no_grad():
        # run larmatch network
        # input: forward( self, input_wireplane_sparsetensors, matchtriplets, query_v, batch_size ):
        tstart_runnet = time.time()
        larmatchout = single_model( batch_sparsetensors, batch_triplets, batch_coordqueries, batchsize, return_larmatch_features=True )
        dt_runnet = time.time()-tstart_runnet

        # get charge feature
        pixval_v = [ batch_sparsetensors[p].features_at_coordinates( batch_coordqueries[p] ) for p in range(3)  ]

        print("Ran larmatch: time elapsed=",dt_runnet," sec")

        # for each entry we collect:
        lmscores = torch.softmax( larmatchout["lm"][0], dim=0 )
        lmfilter = lmscores[1,:]>args.min_score
        lmscores = lmscores[ 1, lmfilter[:]]
        lmscores = lmscores.reshape( (1,lmscores.shape[0]))
        ssnet = larmatchout['ssnet'][0,:,lmfilter[:]]
        paf = larmatchout['paf'][0,:,lmfilter[:]]
        kpscores = larmatchout['kp'][0,:,lmfilter[:]]
        lmfeats = larmatchout['larmatch_features'][0,:,lmfilter[:]]
        spacepoints = torch.transpose( torch.from_numpy(entrydata['spacepoints']).to(DEVICE) , 1, 0 )[:,lmfilter[:]]
        pixval_t = torch.transpose( torch.cat( pixval_v, dim=1 ), 1, 0 )[:,lmfilter[:]]
        print("lmscores: ",lmscores.shape)
        print("lmfeats: ",lmfeats.shape)
        print("kpscores: ",kpscores.shape)
        print("paf: ",paf.shape)
        print("ssnet predictions: ",ssnet.shape)
        print("spacepoints: ",spacepoints.shape)
        print("pixval_t.shape: ",pixval_t.shape)

        # Truth labels
        origin = torch.unsqueeze( torch.from_numpy(entrydata['origin_label']), 0 ).to(DEVICE)[:,lmfilter[:]]
        instanceids = torch.unsqueeze(torch.from_numpy(entrydata['instanceid_label']),0).to(DEVICE)[:,lmfilter[:]]
        particleids = torch.unsqueeze(torch.from_numpy(entrydata['ssnet_label']),0).to(DEVICE)[:,lmfilter[:]]
        keypoint_truth = torch.from_numpy(entrydata['kplabel']).to(DEVICE)[lmfilter[:],:]
        print("instanceids: ",instanceids.shape)
        print("particleids: ",particleids.shape)
        print("keypoint_truth: ",keypoint_truth.shape)
        print("origin: ",origin.shape)

        entrydata = {'lmfeatures':lmfeats.detach().cpu().numpy(),
                     'lmscores':lmscores.detach().cpu().numpy(),
                     'ssnet':ssnet.detach().cpu().numpy(),
                     'paf':paf.detach().cpu().numpy(),
                     'kpscores':kpscores.detach().cpu().numpy(),
                     'pos':spacepoints.detach().cpu().numpy(),
                     'pixvals':pixval_t.detach().cpu().numpy()}

        truthdata = {'instanceids':instanceids.detach().cpu().numpy(),
                     'particleids':particleids.detach().cpu().numpy(),
                     'keyptlabels':np.transpose(keypoint_truth.detach().cpu().numpy(),(1,0)),
                     'origin':origin.detach().cpu().numpy()}

        results = clustering_alg.process_event_points( torch.transpose(spacepoints,1,0), 
                                    torch.transpose(lmfeats,1,0),
                                    torch.transpose(lmscores,1,0),
                                    torch.transpose(ssnet,1,0) )
        
        # convert output tensors to numpy arrays and then store in dictionary
        for k,arr in results.items():
            results[k] = arr.detach().cpu().numpy()
        lmshower_mask = results["lmshower_selection_mask"]
        clusterdata = {'shower_points':results['shower_points'],
                       'shower_feats':results['shower_feats'],
                       'cluster_labels':results['cluster_labels'],
                       'cluster_sampled_pos':results['cluster_sampled_pos'],
                       'cluster_sampled_feat':results['cluster_sampled_feat']}

        # we also need graph truth
        if args.save_true_edges:
            lmshowerpts_instanceids = truthdata["instanceids"][0,lmshower_mask]
            lmshowerpts_particleids = truthdata["particleids"][0,lmshower_mask]
            lmshowerpts_keyptlabels = truthdata["keyptlabels"][:,lmshower_mask]
            shower_edge_list = truth_edge_module.make_true_edge_list( clusterdata['cluster_labels'],
                                                    lmshowerpts_instanceids,
                                                    lmshowerpts_particleids,
                                                    lmshowerpts_keyptlabels,
                                                    verbose=False )
            clusterdata['showercluster_edge_list'] = shower_edge_list         

        if args.save_input_lmpoints:
            clusterdata.update( entrydata )
            clusterdata["lmshower_selection_mask"] = results["lmshower_selection_mask"]

        if args.save_point_mctruth:
            clusterdata.update( truthdata )

        output_entries.append( clusterdata )

    if True:
        continue

             
# write output

with h5py.File(args.output, 'w') as hf:
    for ientry,entrydict in enumerate(output_entries):
        print("writing entry[",ientry,"]")
        for name in entrydict:
            n = name+"_%d"%(ientry)
            print("  write ",n," ",entrydict[name].shape)
            hf.create_dataset( n, data=entrydict[name], compression='gzip', compression_opts=9 )



    #     # output is a dict with keys being the different output heads
    #     if True:
    #         print("-----------------------------------")
    #         #for ib,pred_dict in enumerate(larmatchout):
    #         pred_dict = larmatchout
    #         print("output: ")
    #         for k,v in pred_dict.items():
    #             print(k,": ",v.shape)
    #         print("-----------------------------------")

    #         if "cuda" in args.device_name:
    #             torch.cuda.synchronize()
    #         sys.stdout.flush()    
            
    #         # EVALUATE LARMATCH SCORES
    #         tstart = time.time()
    #         with torch.no_grad():
    #             lm_prob_t = torch.transpose(  pred_dict["lm"].squeeze(), 1, 0 )
    #             lm_prob_t = 1.0-torch.softmax( lm_prob_t, dim=1 )
    #             print("  lm_prob_t=",lm_prob_t.shape)
    #             #print(lm_prob_t[:10,:])

    #         # EVALUATE SSNET SCORES
    #         if config["RUN_SSNET"]:
    #             with torch.no_grad():
    #                 #print("  pred_dict[ssnet] shape: ",pred_dict["ssnet"].shape)        
    #                 ssnet_pred_t = torch.transpose( pred_dict["ssnet"].squeeze(), 1, 0 )
    #                 ssnet_pred_t = torch.softmax( ssnet_pred_t, dim=1 )
    #                 print("  ssnet_pred_t: ",ssnet_pred_t.shape)

    #         # EVALUATE KP-LABEL SCORES
    #         if config["RUN_KPLABEL"]:
    #             with torch.no_grad():
    #                 #print("  pred_dict[kplabel]: ",pred_dict["kp"].shape)
    #                 kplabel_pred_t = torch.transpose( pred_dict["kp"].squeeze(), 1, 0 )
    #                 print("  kplabel_pred_t: ",kplabel_pred_t.shape)

    #         print("prepare score arrays: ",time.time()-tstart," sec")
            
    #         # EVALUATE PAF SCORES
    #         if config["RUN_PAF"]:
    #             with torch.no_grad():
    #                 paf_pred_t = pred_dict['paf']
    #                 paf_pred_t = paf_pred_t.reshape( (paf_pred_t.shape[1],paf_pred_t.shape[2]) )
    #                 paf_pred_t = torch.transpose( paf_pred_t, 1, 0 )        
    #                 print("  paf-pred: ",paf_pred_t.shape)


    #         # PASS LARMATCH OUTPUTS to hitmaker
    #         matchtriplet_np = batch[0][triplet_key]
    #         #sparse_np_v = [ batch[ib]['wireimage_plane%d'%(p)] for p in range(3) ] 
    #         sparse_np_v = [ batch[0]['coord_%d'%(p)] for p in range(3) ] 
    #         prob_np = lm_prob_t.to(torch.device("cpu")).detach().numpy()
    #         #prob_np[:] = 1.0 # hack to check
    #         print("add larmatch output to hitmaker")
    #         sys.stdout.flush()   
    #         pos_v = std.vector("std::vector<float>")()
    #         hitmaker.add_triplet_match_data( prob_np,
    #                                         matchtriplet_np,
    #                                         sparse_np_v[0],
    #                                         sparse_np_v[1],
    #                                         sparse_np_v[2],
    #                                         pos_v,
    #                                         adc_v )

    #         if config["RUN_SSNET"]:
    #             print("  add ssnet data to hitmaker(...). probshape=",ssnet_pred_t.shape)
    #             sys.stdout.flush()   
    #             ssnet_np = ssnet_pred_t.to(torch.device("cpu")).detach().numpy()
    #             hitmaker.add_triplet_ssnet_scores(  matchtriplet_np, 
    #                                                 sparse_np_v[0],
    #                                                 sparse_np_v[1],
    #                                                 sparse_np_v[2],
    #                                                 adc_v.front().meta(),
    #                                                 ssnet_np )                                      

    #         if config["RUN_KPLABEL"]:
    #             print("  add kplabel to hitmaker(...). probshape=",kplabel_pred_t.shape)
    #             sys.stdout.flush()   
    #             kplabel_np = kplabel_pred_t.to(torch.device("cpu")).detach().numpy()
    #             hitmaker.add_triplet_keypoint_scores(  matchtriplet_np,
    #                                                 sparse_np_v[0],
    #                                                 sparse_np_v[1],
    #                                                 sparse_np_v[2],
    #                                                 adc_v.front().meta(),
    #                                                 kplabel_np )

    #         if config["RUN_PAF"]:
    #             print("  add affinity field prediction to hitmaker(...). probshape=",paf_pred_t.shape)
    #             sys.stdout.flush()   
    #             paf_np = paf_pred_t.to(torch.device("cpu")).detach().numpy()
    #             hitmaker.add_triplet_affinity_field(  matchtriplet_np, 
    #                                                 sparse_np_v[0],
    #                                                 sparse_np_v[1],
    #                                                 sparse_np_v[2],
    #                                                 adc_v.front().meta(),
    #                                                 paf_np )

    #         # make flow hits
    #         hitmaker.make_hits( ev_chstatus, adc_v, evout_lfhits )
    #         hitmaker.make_hits( ev_chstatus, adc_v, evout_lmsp )
    #         dt_make_hits = time.time()-tstart
    #         print("number of hits made: ",evout_lfhits.size())
    #         print("time to run net: ",dt_runnet," secs")
    #         print("time to make hits: ",dt_make_hits," secs")

    #         # End of flow direction loop
    #         outll.set_id( ioll.run_id(), ioll.subrun_id(), ioll.event_id() )
    #         outll.next_event(True)
    #         sys.stdout.flush()
    # print("End of entry[",ientry,"]")
    # if False and ientry>=2:
    #     break

print("Finished")
print("Cleaning up")
ioll.close()
iolcv.finalize()
