from __future__ import print_function
import os,sys,time,argparse

# parser = argparse.ArgumentParser("Run CRT-MATCHING algorithms")
# parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="DL merged file")
# parser.add_argument('-cl','--input-cluster',type=str,required=True,help="PCA cluster file")
# parser.add_argument('-o',"--output",type=str,required=True,help="Output file stem")
# parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
# parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
#
# args = parser.parse_args()

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow
import numpy as np
# Info for running:
# EXTBNB
# 		dlreco_filelist "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3_G1_extbnb_dlana/next_final_ext_dlreco_2.txt"
# 		mrcnn_filelist  "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3_G1_extbnb_dlana/next_final_ext_mrcnn_2.txt"
#       tree name: "Pixel_Removal_on_ExtBnB"
        # have to adjust mrcnn file string to add directory:
		# STRING_MRCNN  = "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3_G1_extbnb_dlana/"+STRING_MRCNN;

# NUEINTRINSICS
# 		dlreco_filelist "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/final_nue_dlreco_2.txt"
# 		mrcnn_filelist  "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/final_nue_mrcnn_2.txt"
# 		tree name: "Pixel_Removal_on_Intrinsic_Nue_Truth_Image"
#
# NUE LOWE INTRINSICS
# 		dlreco_filelist "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_LowE/temp_dlreco_low_fullpaths.txt"
# 		mrcnn_filelist  "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_LowE/temp_mrcnn_low_fullpaths.txt"
# 		tree name: "Pixel_Removal_on_Intrinsic_Nue_Truth_LowE_Image"
#
# BNBNUOVERLAY
# 		dlreco_filelist "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/final_merged_list.txt"
# 		mrcnn_filelist  "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/final_hadd_list.txt"
# 		treename"Pixel_Removal_on_BnB_Nu_Overlay_Truth_Image"
#
# BNB5e19
# 		dlreco_filelist "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_bnb5e19_fixed/partial_bnb5e19_merged.txt"
# 		mrcnn_filelist  "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_bnb5e19_fixed/partial_bnb5e19_mrcnn.txt"
# 		tree name: "Pixel_Removal_on_BnB5e19_Image"
ismc = True
infile_dlreco  = "/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/data/00/01/37/64/merged_dlreco_001361e0-3306-491f-9098-1d08eee8458b.root"
infile_mrcnn   = "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/0/0069/000/2/hadd_mrcnnproposals_merged_dlreco_001361e0-3306-491f-9098-1d08eee8458b.root"
OutFileName    = "test2.root"
OutTreeName    = "Pixel_Removal_Default"

CRVarsMaker = larflow.cosmicremovalsuite.CRVarsMaker()
# CRVarsMaker.run_varsmaker_rootfile(ismc, infile_dlreco,infile_mrcnn,OutFileName,OutTreeName)

io_mrcnn  =  larcv.IOManager(larcv.IOManager.kREAD,"IOManager_MRCNN", larcv.IOManager.kTickBackward)
io_mrcnn.add_in_file(infile_mrcnn)
io_mrcnn.initialize()

io_dlreco  =  larcv.IOManager(larcv.IOManager.kREAD,"IOManager_DLRECO", larcv.IOManager.kTickBackward)
io_dlreco.reverse_all_products()
io_dlreco.add_in_file(infile_dlreco)
io_dlreco.initialize()

ioll_dlreco  = larlite.storage_manager(larlite.storage_manager.kREAD)
ioll_dlreco.add_in_filename(infile_dlreco)
ioll_dlreco.open()
ev_num = 0
for ev_num in range(3):
    print("Running Event", ev_num)
    outvec = CRVarsMaker.run_varsmaker_arrsout(ismc, io_mrcnn, io_dlreco, ioll_dlreco, ev_num)
    # print(outvec)
    nparr = np.zeros((outvec.size()))
    for i in range(outvec.size()):
        nparr[i] = outvec.at(i)
    # print()
    # print(nparr)
