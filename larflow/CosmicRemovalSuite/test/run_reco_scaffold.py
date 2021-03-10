from __future__ import print_function
import os,sys,time,argparse

import ROOT as rt
from ROOT import std
from larlite import larlite
from larcv import larcv
from larflow import larflow
import numpy as np
import glob
import pickle

def main():
    # # Set up args at a later time
    # parser = argparse.ArgumentParser("Run CRT-MATCHING algorithms")
    # parser.add_argument('-dl','--input-dlmerged',type=str,required=True,help="DL merged file")
    # parser.add_argument('-cl','--input-cluster',type=str,required=True,help="PCA cluster file")
    # parser.add_argument('-o',"--output",type=str,required=True,help="Output file stem")
    # parser.add_argument('-n','--num-entries',type=int,default=None,help="Number of entries to run")
    # parser.add_argument('-e','--start-entry',type=int,default=0,help="Starting entry")
    #
    # args = parser.parse_args()

    #Lets assume we start with dlreco file:
    bdtmodelfile="/cluster/tufts/wongjiradlab/jmills09/ubdl_gen2/gen2_checks/bdtweights/cosmictag_BDTweights_test.pickle"
    # BDT Variables
    n_bdtvars = 16
    bdtvars_list = ["adc_pix_count", \
        "wc_frac", \
        "mrcnn_frac_020",\
        "mrcnn_frac_090",\
        "combined_frac_020",\
        "combined_frac_090",\
        "hip_count_050",\
        "hip_count_090",\
        "mip_count_050",\
        "mip_count_090",\
        "shower_count_050",\
        "shower_count_090",\
        "michel_count_050",\
        "michel_count_090",\
        "delta_count_050",\
        "delta_count_090",\
        ]

    # Nue Intrinsic File Example
    infile_dlreco  = "/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/data/00/01/37/64/merged_dlreco_001361e0-3306-491f-9098-1d08eee8458b.root"
    # ExtBnB File Example
    # infile_dlreco = "/cluster/tufts/wongjiradlab/larbys/data/mcc9/mcc9_v29e_dl_run3_G1_extbnb_dlana/data/mcc9_v29e_dl_run3_G1_extbnb_dlana/merged_dlana_d9679e9b-3be3-4411-bc25-6e2cea860827.root"
    # Output Directory for MRCNN Single Event Files
    mrcnn_outdir = "/cluster/tufts/wongjiradlab/jmills09/ubdl_gen2/larflow/larflow/CosmicRemovalSuite/test/"
    # This grabs more variables from the BDTVars Getter script if set to True.
    # Information about the segment image, and a 21x21 box around then neutrino vertex
    ismc = False
    # Dictionaries to go from BDTVars -> idx in the array from the
    # CRVarsMaker output, and reverse
    var_str2idx_dict, var_idx2str_dict = return_dicts()

    # Load in dlreco file into io methods
    io_dlreco  =  larcv.IOManager(larcv.IOManager.kREAD,"IOManager_DLRECO", larcv.IOManager.kTickBackward)
    io_dlreco.reverse_all_products()
    io_dlreco.add_in_file(infile_dlreco)
    io_dlreco.initialize()

    ioll_dlreco  = larlite.storage_manager(larlite.storage_manager.kREAD)
    ioll_dlreco.add_in_filename(infile_dlreco)
    ioll_dlreco.open()
    nentries = io_dlreco.get_n_entries()
    ev_probs = []
    for ev_num in range(0,nentries):
        print("Running Event", ev_num)
        ###########################
        # Create MaskRCNN File for one entry
        # Execute MaskRCNN script. Using MaskRCNN build in jmills09/maskrcnn_gen2/
        # Permissions set to available for all.
        mrcnn_exec_string = "cd /cluster/tufts/wongjiradlab/jmills09/maskrcnn_gen2/ ; python tools/save_output_objects.py --dataset particle --cfg configs/tuftscluster_config_2.yaml --load_ckpt weights/y_plane.pth --input_file "+ infile_dlreco +" --output_dir "+ mrcnn_outdir  + " --one_entry "+ str(ev_num)
        os.system(mrcnn_exec_string)
        # Grab output file path+name
        ev_string = str(ev_num)
        if (len(str(ev_num)) < 2):
            ev_string = "0"+str(ev_num)
            print()
        infile_mrcnn = glob.glob(mrcnn_outdir+"mrcnnproposals_"+str(ev_string)+"*")[0]
        ###########################


        ###########################
        # Calculate Cosmic Removal BDT Variables with CRVarsMaker
        CRVarsMaker = larflow.cosmicremovalsuite.CRVarsMaker()

        io_mrcnn  =  larcv.IOManager(larcv.IOManager.kREAD,"IOManager_MRCNN", larcv.IOManager.kTickBackward)
        io_mrcnn.add_in_file(infile_mrcnn)
        io_mrcnn.initialize()
        io_dlreco.read_entry(ev_num)
        ioll_dlreco.go_to(ev_num)
        io_mrcnn.read_entry(0) #file only has one event, was made in this loop
        outvec = CRVarsMaker.run_varsmaker_arrsout(ismc, io_mrcnn, io_dlreco, ioll_dlreco)
        crvars_np = np.zeros((outvec.size()))
        for i in range(outvec.size()):
            crvars_np[i] = outvec.at(i)
        ############################


        ############################
        # Run Event through BDT Model
        print("Loading BDT Model")
        cosmictag_BDT = pickle.load(open(bdtmodelfile, "rb"))

        bdtvars = np.zeros((1,n_bdtvars))
        for bdtvar_idx in range(n_bdtvars):
            bdtvars[0][bdtvar_idx] = crvars_np[var_str2idx_dict[bdtvars_list[bdtvar_idx]]]
        # Predict Probability, [0] -> Cosmic [1] -> Nue
        ev_prob = cosmictag_BDT.predict_proba(bdtvars)
        print(ev_prob)
        ev_probs.append(ev_prob[0])
        # Simple cut assuming single BDT model, placeholder
        ev_kept = (ev_prob[0][1] > 0.5)
        if ev_kept:
            # Reconstruct Event
            print()
        else:
            # Save proof of event for bookkeeping but ignore event.
            print()

        ##########################
        # Do Later LArMatch
        ####################
    print(ev_probs)
    return "\nFinishing Main"

def return_dicts():
    var_idx2str = {}
    var_str2idx = {}
    var_idx2str[0] = "run"
    var_idx2str[1] = "subrun"
    var_idx2str[2] = "event"
    var_idx2str[3] = "adc_pix_count"
    var_idx2str[4] = "wc_frac"
    var_idx2str[5] = "mrcnn_frac_000"
    var_idx2str[6] = "mrcnn_frac_020"
    var_idx2str[7] = "mrcnn_frac_030"
    var_idx2str[8] = "mrcnn_frac_040"
    var_idx2str[9] = "mrcnn_frac_050"
    var_idx2str[10] = "mrcnn_frac_060"
    var_idx2str[11] = "mrcnn_frac_070"
    var_idx2str[12] = "mrcnn_frac_080"
    var_idx2str[13] = "mrcnn_frac_090"
    var_idx2str[14] = "mrcnn_frac_095"
    var_idx2str[15] = "mrcnn_frac_098"
    var_idx2str[16] = "mrcnn_frac_09999"
    var_idx2str[17] = "combined_frac_000"
    var_idx2str[18] = "combined_frac_020"
    var_idx2str[19] = "combined_frac_030"
    var_idx2str[20] = "combined_frac_040"
    var_idx2str[21] = "combined_frac_050"
    var_idx2str[22] = "combined_frac_060"
    var_idx2str[23] = "combined_frac_070"
    var_idx2str[24] = "combined_frac_080"
    var_idx2str[25] = "combined_frac_090"
    var_idx2str[26] = "combined_frac_095"
    var_idx2str[27] = "combined_frac_098"
    var_idx2str[28] = "combined_frac_09999"
    var_idx2str[29] = "hip_count_030"
    var_idx2str[30] = "hip_count_050"
    var_idx2str[31] = "hip_count_070"
    var_idx2str[32] = "hip_count_090"
    var_idx2str[33] = "mip_count_030"
    var_idx2str[34] = "mip_count_050"
    var_idx2str[35] = "mip_count_070"
    var_idx2str[36] = "mip_count_090"
    var_idx2str[37] = "shower_count_030"
    var_idx2str[38] = "shower_count_050"
    var_idx2str[39] = "shower_count_070"
    var_idx2str[40] = "shower_count_090"
    var_idx2str[41] = "delta_count_030"
    var_idx2str[42] = "delta_count_050"
    var_idx2str[43] = "delta_count_070"
    var_idx2str[44] = "delta_count_090"
    var_idx2str[45] = "michel_count_030"
    var_idx2str[46] = "michel_count_050"
    var_idx2str[47] = "michel_count_070"
    var_idx2str[48] = "michel_count_090"
    var_idx2str[49] = "adc_pix_count_segment"
    var_idx2str[50] = "wc_frac_segment"
    var_idx2str[51] = "mrcnn_frac_segment_000"
    var_idx2str[52] = "mrcnn_frac_segment_020"
    var_idx2str[53] = "mrcnn_frac_segment_030"
    var_idx2str[54] = "mrcnn_frac_segment_040"
    var_idx2str[55] = "mrcnn_frac_segment_050"
    var_idx2str[56] = "mrcnn_frac_segment_060"
    var_idx2str[57] = "mrcnn_frac_segment_070"
    var_idx2str[58] = "mrcnn_frac_segment_080"
    var_idx2str[59] = "mrcnn_frac_segment_090"
    var_idx2str[60] = "mrcnn_frac_segment_095"
    var_idx2str[61] = "mrcnn_frac_segment_098"
    var_idx2str[62] = "mrcnn_frac_segment_09999"
    var_idx2str[63] = "combined_frac_segment_000"
    var_idx2str[64] = "combined_frac_segment_020"
    var_idx2str[65] = "combined_frac_segment_030"
    var_idx2str[66] = "combined_frac_segment_040"
    var_idx2str[67] = "combined_frac_segment_050"
    var_idx2str[68] = "combined_frac_segment_060"
    var_idx2str[69] = "combined_frac_segment_070"
    var_idx2str[70] = "combined_frac_segment_080"
    var_idx2str[71] = "combined_frac_segment_090"
    var_idx2str[72] = "combined_frac_segment_095"
    var_idx2str[73] = "combined_frac_segment_098"
    var_idx2str[74] = "combined_frac_segment_09999"
    var_idx2str[75] = "hip_count_segment_030"
    var_idx2str[76] = "hip_count_segment_050"
    var_idx2str[77] = "hip_count_segment_070"
    var_idx2str[78] = "hip_count_segment_090"
    var_idx2str[79] = "mip_count_segment_030"
    var_idx2str[80] = "mip_count_segment_050"
    var_idx2str[81] = "mip_count_segment_070"
    var_idx2str[82] = "mip_count_segment_090"
    var_idx2str[83] = "shower_count_segment_030"
    var_idx2str[84] = "shower_count_segment_050"
    var_idx2str[85] = "shower_count_segment_070"
    var_idx2str[86] = "shower_count_segment_090"
    var_idx2str[87] = "delta_count_segment_030"
    var_idx2str[88] = "delta_count_segment_050"
    var_idx2str[89] = "delta_count_segment_070"
    var_idx2str[90] = "delta_count_segment_090"
    var_idx2str[91] = "michel_count_segment_030"
    var_idx2str[92] = "michel_count_segment_050"
    var_idx2str[93] = "michel_count_segment_070"
    var_idx2str[94] = "michel_count_segment_090"
    var_idx2str[95] = "adc_pix_count_box21"
    var_idx2str[96] = "wc_frac_box21"
    var_idx2str[97] = "mrcnn_frac_000_box21"
    var_idx2str[98] = "mrcnn_frac_020_box21"
    var_idx2str[99] = "mrcnn_frac_030_box21"
    var_idx2str[100] = "mrcnn_frac_040_box21"
    var_idx2str[101] = "mrcnn_frac_050_box21"
    var_idx2str[102] = "mrcnn_frac_060_box21"
    var_idx2str[103] = "mrcnn_frac_070_box21"
    var_idx2str[104] = "mrcnn_frac_080_box21"
    var_idx2str[105] = "mrcnn_frac_090_box21"
    var_idx2str[106] = "mrcnn_frac_095_box21"
    var_idx2str[107] = "mrcnn_frac_098_box21"
    var_idx2str[108] = "mrcnn_frac_09999_box21"
    var_idx2str[109] = "combined_frac_000_box21"
    var_idx2str[110] = "combined_frac_020_box21"
    var_idx2str[111] = "combined_frac_030_box21"
    var_idx2str[112] = "combined_frac_040_box21"
    var_idx2str[113] = "combined_frac_050_box21"
    var_idx2str[114] = "combined_frac_060_box21"
    var_idx2str[115] = "combined_frac_070_box21"
    var_idx2str[116] = "combined_frac_080_box21"
    var_idx2str[117] = "combined_frac_090_box21"
    var_idx2str[118] = "combined_frac_095_box21"
    var_idx2str[119] = "combined_frac_098_box21"
    var_idx2str[120] = "combined_frac_09999_box21"
    var_idx2str[121] = "mcvtx_x"
    var_idx2str[122] = "mcvtx_y"
    var_idx2str[123] = "mcvtx_z"
    var_idx2str[124] = "mcvtx_contained"

    var_str2idx["run"]		    =	0
    var_str2idx["subrun"]		=	1
    var_str2idx["event"]		=	2
    var_str2idx["adc_pix_count"]=	3
    var_str2idx["wc_frac"]		=	4
    var_str2idx["mrcnn_frac_000"]		=	5
    var_str2idx["mrcnn_frac_020"]		=	6
    var_str2idx["mrcnn_frac_030"]		=	7
    var_str2idx["mrcnn_frac_040"]		=	8
    var_str2idx["mrcnn_frac_050"]		=	9
    var_str2idx["mrcnn_frac_060"]		=	10
    var_str2idx["mrcnn_frac_070"]		=	11
    var_str2idx["mrcnn_frac_080"]		=	12
    var_str2idx["mrcnn_frac_090"]		=	13
    var_str2idx["mrcnn_frac_095"]		=	14
    var_str2idx["mrcnn_frac_098"]		=	15
    var_str2idx["mrcnn_frac_09999"]		=	16
    var_str2idx["combined_frac_000"]		=	17
    var_str2idx["combined_frac_020"]		=	18
    var_str2idx["combined_frac_030"]		=	19
    var_str2idx["combined_frac_040"]		=	20
    var_str2idx["combined_frac_050"]		=	21
    var_str2idx["combined_frac_060"]		=	22
    var_str2idx["combined_frac_070"]		=	23
    var_str2idx["combined_frac_080"]		=	24
    var_str2idx["combined_frac_090"]		=	25
    var_str2idx["combined_frac_095"]		=	26
    var_str2idx["combined_frac_098"]		=	27
    var_str2idx["combined_frac_09999"]		=	28
    var_str2idx["hip_count_030"]		=	29
    var_str2idx["hip_count_050"]		=	30
    var_str2idx["hip_count_070"]		=	31
    var_str2idx["hip_count_090"]		=	32
    var_str2idx["mip_count_030"]		=	33
    var_str2idx["mip_count_050"]		=	34
    var_str2idx["mip_count_070"]		=	35
    var_str2idx["mip_count_090"]		=	36
    var_str2idx["shower_count_030"]		=	37
    var_str2idx["shower_count_050"]		=	38
    var_str2idx["shower_count_070"]		=	39
    var_str2idx["shower_count_090"]		=	40
    var_str2idx["delta_count_030"]		=	41
    var_str2idx["delta_count_050"]		=	42
    var_str2idx["delta_count_070"]		=	43
    var_str2idx["delta_count_090"]		=	44
    var_str2idx["michel_count_030"]		=	45
    var_str2idx["michel_count_050"]		=	46
    var_str2idx["michel_count_070"]		=	47
    var_str2idx["michel_count_090"]		=	48
    var_str2idx["adc_pix_count_segment"]=	49
    var_str2idx["wc_frac_segment"]		=	50
    var_str2idx["mrcnn_frac_segment_000"]		=	51
    var_str2idx["mrcnn_frac_segment_020"]		=	52
    var_str2idx["mrcnn_frac_segment_030"]		=	53
    var_str2idx["mrcnn_frac_segment_040"]		=	54
    var_str2idx["mrcnn_frac_segment_050"]		=	55
    var_str2idx["mrcnn_frac_segment_060"]		=	56
    var_str2idx["mrcnn_frac_segment_070"]		=	57
    var_str2idx["mrcnn_frac_segment_080"]		=	58
    var_str2idx["mrcnn_frac_segment_090"]		=	59
    var_str2idx["mrcnn_frac_segment_095"]		=	60
    var_str2idx["mrcnn_frac_segment_098"]		=	61
    var_str2idx["mrcnn_frac_segment_09999"]		=	62
    var_str2idx["combined_frac_segment_000"]		=	63
    var_str2idx["combined_frac_segment_020"]		=	64
    var_str2idx["combined_frac_segment_030"]		=	65
    var_str2idx["combined_frac_segment_040"]		=	66
    var_str2idx["combined_frac_segment_050"]		=	67
    var_str2idx["combined_frac_segment_060"]		=	68
    var_str2idx["combined_frac_segment_070"]		=	69
    var_str2idx["combined_frac_segment_080"]		=	70
    var_str2idx["combined_frac_segment_090"]		=	71
    var_str2idx["combined_frac_segment_095"]		=	72
    var_str2idx["combined_frac_segment_098"]		=	73
    var_str2idx["combined_frac_segment_09999"]		=	74
    var_str2idx["hip_count_segment_030"]		=	75
    var_str2idx["hip_count_segment_050"]		=	76
    var_str2idx["hip_count_segment_070"]		=	77
    var_str2idx["hip_count_segment_090"]		=	78
    var_str2idx["mip_count_segment_030"]		=	79
    var_str2idx["mip_count_segment_050"]		=	80
    var_str2idx["mip_count_segment_070"]		=	81
    var_str2idx["mip_count_segment_090"]		=	82
    var_str2idx["shower_count_segment_030"]		=	83
    var_str2idx["shower_count_segment_050"]		=	84
    var_str2idx["shower_count_segment_070"]		=	85
    var_str2idx["shower_count_segment_090"]		=	86
    var_str2idx["delta_count_segment_030"]		=	87
    var_str2idx["delta_count_segment_050"]		=	88
    var_str2idx["delta_count_segment_070"]		=	89
    var_str2idx["delta_count_segment_090"]		=	90
    var_str2idx["michel_count_segment_030"]		=	91
    var_str2idx["michel_count_segment_050"]		=	92
    var_str2idx["michel_count_segment_070"]		=	93
    var_str2idx["michel_count_segment_090"]		=	94
    var_str2idx["adc_pix_count_box21"]		=	95
    var_str2idx["wc_frac_box21"]		    =	96
    var_str2idx["mrcnn_frac_000_box21"]		=	97
    var_str2idx["mrcnn_frac_020_box21"]		=	98
    var_str2idx["mrcnn_frac_030_box21"]		=	99
    var_str2idx["mrcnn_frac_040_box21"]		=	100
    var_str2idx["mrcnn_frac_050_box21"]		=	101
    var_str2idx["mrcnn_frac_060_box21"]		=	102
    var_str2idx["mrcnn_frac_070_box21"]		=	103
    var_str2idx["mrcnn_frac_080_box21"]		=	104
    var_str2idx["mrcnn_frac_090_box21"]		=	105
    var_str2idx["mrcnn_frac_095_box21"]		=	106
    var_str2idx["mrcnn_frac_098_box21"]		=	107
    var_str2idx["mrcnn_frac_09999_box21"]		=	108
    var_str2idx["combined_frac_000_box21"]		=	109
    var_str2idx["combined_frac_020_box21"]		=	110
    var_str2idx["combined_frac_030_box21"]		=	111
    var_str2idx["combined_frac_040_box21"]		=	112
    var_str2idx["combined_frac_050_box21"]		=	113
    var_str2idx["combined_frac_060_box21"]		=	114
    var_str2idx["combined_frac_070_box21"]		=	115
    var_str2idx["combined_frac_080_box21"]		=	116
    var_str2idx["combined_frac_090_box21"]		=	117
    var_str2idx["combined_frac_095_box21"]		=	118
    var_str2idx["combined_frac_098_box21"]		=	119
    var_str2idx["combined_frac_09999_box21"]    =	120
    var_str2idx["mcvtx_x"]		=	121
    var_str2idx["mcvtx_y"]		=	122
    var_str2idx["mcvtx_z"]		=	123
    var_str2idx["mcvtx_contained"]		=	124

    return var_str2idx, var_idx2str


if __name__ == "__main__":
    print(main())
