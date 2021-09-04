#ifndef __CRVARSMAKER_CXX__
#define __CRVARSMAKER_CXX__

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <math.h>
// ROOT
#include "CRVarsMaker.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"

// larutil
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/ClockConstants.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventClusterMask.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "larcv/core/Base/larcv_logger.h"
// larlite
#include "larlite/DataFormat/storage_manager.h"

// ublarcvapp
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "ublarcvapp/dbscan/sDBScan.h"
// Misc

namespace larflow {
namespace cosmicremovalsuite {

int CRVarsMaker::run_varsmaker(int mode, int file_limit, int start_file){
	// Mode Notes: 0 ExtBnB, 1 NueIntrinsic, 2 NueIntrinsic Bckgrd, 3 NueLowE, 4 lowebckgrd, 5 BnB Overlay , 6 BnB Overlay Bckgrd, 7 Data5e19
	// int mode = 1; // 0 for Cosmic ExtBnB Running
	// int file_limit = 1;
	// int start_file =-1;

	// Unusused right now
	bool IsMC                  = true; // Tells the cosmicmasking whether to look at the true image alongside the wire image.
	std::string mergedfilelist = "";
	std::string mrcnnfilelist  = "";
	std::string OutFileName    = "";
	std::string OutTreeName    = "";
	std::string SampleName     = "";


	// if (nargs == 3){
	// 	std::cout << "Using Custom Args\n";
	// 	char* p;
	// 	mode = strtol(argv[1], &p, 10);
	// 	file_limit = strtol(argv[2], &p, 10);
	// 	std::cout << mode << " Mode Chosen\n";
	// 	std::cout << file_limit << " Files Being Run over\n";
	// }
	// else if (nargs == 4){
	// 	std::cout << "Using Custom Args\n";
	// 	char* p;
	// 	mode = strtol(argv[1], &p, 10);
	// 	start_file = strtol(argv[2], &p, 10);
	// 	file_limit = strtol(argv[3], &p, 10);
	// 	std::cout << mode << " Mode Chosen\n";
	// 	std::cout << start_file << " Starting File Number\n";
	// 	std::cout << file_limit << " Ending File Number\n";
	// }
	// else{
	// 	std::cout << "Using Default Args\n";
	// 	std::cout << mode << " Mode Chosen\n";
	// 	std::cout << file_limit << " Files Being Run over\n";
	// }
	// if ((mode==2)||(mode==4)||(mode==6)){
	// 	std::cout << "Background Modes on MC are not coded yet. Use the ExtBnB to estimate performance on real cosmics\n";
	// 	return -1;
	// }

	std::cout << "Hello world " << "\n";
	std::string outfile_name ="jobsoutput/Cosmic_Masking_Output_test_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";
	if (mode == 0){outfile_name ="jobsoutput/Pixel_Removal_ExtBnB_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	else if (mode == 1){outfile_name ="jobsoutput/Pixel_Removal_NueIntrinsic_Truth_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	else if (mode == 2){outfile_name ="jobsoutput/Pixel_Removal_NueIntrinsic_Background_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	else if (mode == 3){outfile_name ="jobsoutput/Pixel_Removal_NueIntrinsic_Truth_LowE_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	else if (mode == 5){outfile_name ="jobsoutput/Pixel_Removal_BnB_Nu_Overlay_Truth_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	else if (mode == 7){outfile_name ="jobsoutput/Pixel_Removal_BnB5e19_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	if (start_file != -1){
		if (mode == 0){outfile_name ="jobsoutput/ExtBnB/Pixel_Removal_ExtBnB_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
		else if (mode == 1){outfile_name ="jobsoutput/NueIntrinsicTruth/Pixel_Removal_NueIntrinsic_Truth_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
		else if (mode == 2){outfile_name ="jobsoutput/NueIntrinsicBackground/Pixel_Removal_NueIntrinsic_Background_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
		else if (mode == 3){outfile_name ="jobsoutput/NueIntrinsicTruth_LowE/Pixel_Removal_NueIntrinsic_Truth_LowE_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
		else if (mode == 5){outfile_name ="jobsoutput/BnB_Nu_OverlayTruth/Pixel_Removal_BnB_Nu_Overlay_Truth_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
		else if (mode == 7){outfile_name ="jobsoutput/BnB5e19/Pixel_Removal_BnB5e19_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";}
	}
	outfile_name = "testout_"+std::to_string(start_file)+"_"+std::to_string(file_limit)+".root";
	TFile *myfile = new TFile(outfile_name.c_str(),"RECREATE");
	myfile->cd();
	TTree *mytree = new TTree("Pixel_Removal_Default","Tagging_Fractions_Default");
	std::ifstream infile_mrcnn;
	std::ifstream infile_dlreco;
	std::string STRING_MRCNN;
	std::string STRING_DLRECO;
	if (mode == 0){
		std::cout << "Running over ExtBnB \n";
		mytree->SetName("Pixel_Removal_on_ExtBnB");
		mytree->SetTitle("Pixel_Removal_on_ExtBnB");
		infile_dlreco.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3_G1_extbnb_dlana/next_final_ext_dlreco_2.txt");
		infile_mrcnn.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3_G1_extbnb_dlana/next_final_ext_mrcnn_2.txt");
	}
	else if ((mode == 1) || (mode==2)){
		std::cout << "Running over Intrinsic Nue \n";
		infile_dlreco.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/final_nue_dlreco_2.txt");
		infile_mrcnn.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_overlay_nocrtremerge/final_nue_mrcnn_2.txt");
		if (mode ==1){
			std::cout << "Checking Pixel Removal of Truth Image\n";
			mytree->SetName("Pixel_Removal_on_Intrinsic_Nue_Truth_Image");
			mytree->SetTitle("Pixel_Removal_on_Intrinsic_Nue_Truth_Image");
		}
		else if (mode==2){
			std::cout << "Checking Pixel Removal of Overlay Image Background\n";
			std::cout << "This Check is not yet implemented, returning failure\n";
			return -1;
			mytree->SetName("Pixel_Removal_on_Intrinsic_Nue_Background_Image");
			mytree->SetTitle("Pixel_Removal_on_Intrinsic_Nue_Background_Image");
		}
	}
	else if ((mode == 3)  || (mode==4)){
		std::cout << "Running over Intrinsic Nue LowE \n";
		infile_dlreco.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_LowE/temp_dlreco_low_fullpaths.txt");
		infile_mrcnn.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_intrinsic_nue_LowE/temp_mrcnn_low_fullpaths.txt");
		if (mode ==3){
			std::cout << "Checking Pixel Removal of LowE Truth Image\n";
			mytree->SetName("Pixel_Removal_on_Intrinsic_Nue_Truth_LowE_Image");
			mytree->SetTitle("Pixel_Removal_on_Intrinsic_Nue_Truth_LowE_Image");
		}
		else if (mode==4){
			std::cout << "Checking Pixel Removal of Overlay LowE Image Background\n";
			std::cout << "This Check is not yet implemented, returning failure\n";
			return -1;
			mytree->SetName("Pixel_Removal_on_Intrinsic_Nue_Background_LowE_Image");
			mytree->SetTitle("Pixel_Removal_on_Intrinsic_Nue_Background_LowE_Image");
		}
	}
	else if ((mode == 5) || (mode ==6)){
		std::cout << "Running over BnB Nu Overlay Truth \n";
		infile_dlreco.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/final_merged_list.txt");
		infile_mrcnn.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3b_bnb_nu_overlay_nocrtremerge/final_hadd_list.txt");
		if (mode ==5){
			std::cout << "Checking Pixel Removal of BnB Nu Overlay Truth Image\n";
			mytree->SetName("Pixel_Removal_on_BnB_Nu_Overlay_Truth_Image");
			mytree->SetTitle("Pixel_Removal_on_BnB_Nu_Overlay_Truth_Image");
		}
		else if (mode==6){
			std::cout << "Checking Pixel Removal of Overlay Image Background\n";
			std::cout << "This Check is not yet implemented, returning failure\n";
			return -1;
			mytree->SetName("Pixel_Removal_on_BnB_Nu_Overlay_Background_Image");
			mytree->SetTitle("Pixel_Removal_on_BnB_Nu_Overlay_Background_Image");
		}
	}
	else if (mode == 7) {
		std::cout << "Running over BnB5e19 \n";
		infile_dlreco.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_bnb5e19_fixed/partial_bnb5e19_merged.txt");
		infile_mrcnn.open ("/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_bnb5e19_fixed/partial_bnb5e19_mrcnn.txt");
		mytree->SetName("Pixel_Removal_on_BnB5e19_Image");
		mytree->SetTitle("Pixel_Removal_on_BnB5e19_Image");
	}
	std::vector<double> cosmic_thresh_v = {0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98};

	int run=-1;
	int subrun=-1;
	int event=-1;
	int adc_pix_count=-1;
	double wc_frac=-1;
	double mrcnn_frac_000=-1;
	double mrcnn_frac_020=-1;
	double mrcnn_frac_030=-1;
	double mrcnn_frac_040=-1;
	double mrcnn_frac_050=-1;
	double mrcnn_frac_060=-1;
	double mrcnn_frac_070=-1;
	double mrcnn_frac_080=-1;
	double mrcnn_frac_090=-1;
	double mrcnn_frac_095=-1;
	double mrcnn_frac_098=-1;
	double mrcnn_frac_09999=-1;
	double combined_frac_000=-1;
	double combined_frac_020=-1;
	double combined_frac_030=-1;
	double combined_frac_040=-1;
	double combined_frac_050=-1;
	double combined_frac_060=-1;
	double combined_frac_070=-1;
	double combined_frac_080=-1;
	double combined_frac_090=-1;
	double combined_frac_095=-1;
	double combined_frac_098=-1;
	double combined_frac_09999=-1;

	int hip_count_030=-1;
	int hip_count_050=-1;
	int hip_count_070=-1;
	int hip_count_090=-1;
	int mip_count_030=-1;
	int mip_count_050=-1;
	int mip_count_070=-1;
	int mip_count_090=-1;
	int shower_count_030=-1;
	int shower_count_050=-1;
	int shower_count_070=-1;
	int shower_count_090=-1;
	int delta_count_030=-1;
	int delta_count_050=-1;
	int delta_count_070=-1;
	int delta_count_090=-1;
	int michel_count_030=-1;
	int michel_count_050=-1;
	int michel_count_070=-1;
	int michel_count_090=-1;

	int adc_pix_count_segment=-1;
	double wc_frac_segment=-1;
	double mrcnn_frac_segment_000=-1;
	double mrcnn_frac_segment_020=-1;
	double mrcnn_frac_segment_030=-1;
	double mrcnn_frac_segment_040=-1;
	double mrcnn_frac_segment_050=-1;
	double mrcnn_frac_segment_060=-1;
	double mrcnn_frac_segment_070=-1;
	double mrcnn_frac_segment_080=-1;
	double mrcnn_frac_segment_090=-1;
	double mrcnn_frac_segment_095=-1;
	double mrcnn_frac_segment_098=-1;
	double mrcnn_frac_segment_09999=-1;
	double combined_frac_segment_000=-1;
	double combined_frac_segment_020=-1;
	double combined_frac_segment_030=-1;
	double combined_frac_segment_040=-1;
	double combined_frac_segment_050=-1;
	double combined_frac_segment_060=-1;
	double combined_frac_segment_070=-1;
	double combined_frac_segment_080=-1;
	double combined_frac_segment_090=-1;
	double combined_frac_segment_095=-1;
	double combined_frac_segment_098=-1;
	double combined_frac_segment_09999=-1;

	int hip_count_segment_030=-1;
	int hip_count_segment_050=-1;
	int hip_count_segment_070=-1;
	int hip_count_segment_090=-1;
	int mip_count_segment_030=-1;
	int mip_count_segment_050=-1;
	int mip_count_segment_070=-1;
	int mip_count_segment_090=-1;
	int shower_count_segment_030=-1;
	int shower_count_segment_050=-1;
	int shower_count_segment_070=-1;
	int shower_count_segment_090=-1;
	int delta_count_segment_030=-1;
	int delta_count_segment_050=-1;
	int delta_count_segment_070=-1;
	int delta_count_segment_090=-1;
	int michel_count_segment_030=-1;
	int michel_count_segment_050=-1;
	int michel_count_segment_070=-1;
	int michel_count_segment_090=-1;

	int adc_pix_count_box21=-1;
	double wc_frac_box21=-1;
	double mrcnn_frac_000_box21=-1;
	double mrcnn_frac_020_box21=-1;
	double mrcnn_frac_030_box21=-1;
	double mrcnn_frac_040_box21=-1;
	double mrcnn_frac_050_box21=-1;
	double mrcnn_frac_060_box21=-1;
	double mrcnn_frac_070_box21=-1;
	double mrcnn_frac_080_box21=-1;
	double mrcnn_frac_090_box21=-1;
	double mrcnn_frac_095_box21=-1;
	double mrcnn_frac_098_box21=-1;
	double mrcnn_frac_09999_box21=-1;
	double combined_frac_000_box21=-1;
	double combined_frac_020_box21=-1;
	double combined_frac_030_box21=-1;
	double combined_frac_040_box21=-1;
	double combined_frac_050_box21=-1;
	double combined_frac_060_box21=-1;
	double combined_frac_070_box21=-1;
	double combined_frac_080_box21=-1;
	double combined_frac_090_box21=-1;
	double combined_frac_095_box21=-1;
	double combined_frac_098_box21=-1;
	double combined_frac_09999_box21=-1;

	mytree->Branch("run",&run);
	mytree->Branch("subrun",&subrun);
	mytree->Branch("event",&event);
	mytree->Branch("adc_pix_count",&adc_pix_count);
	mytree->Branch("wc_frac",&wc_frac);
	mytree->Branch("mrcnn_frac_000",&mrcnn_frac_000);
	mytree->Branch("mrcnn_frac_020",&mrcnn_frac_020);
	mytree->Branch("mrcnn_frac_030",&mrcnn_frac_030);
	mytree->Branch("mrcnn_frac_040",&mrcnn_frac_040);
	mytree->Branch("mrcnn_frac_050",&mrcnn_frac_050);
	mytree->Branch("mrcnn_frac_060",&mrcnn_frac_060);
	mytree->Branch("mrcnn_frac_070",&mrcnn_frac_070);
	mytree->Branch("mrcnn_frac_080",&mrcnn_frac_080);
	mytree->Branch("mrcnn_frac_090",&mrcnn_frac_090);
	mytree->Branch("mrcnn_frac_095",&mrcnn_frac_095);
	mytree->Branch("mrcnn_frac_098",&mrcnn_frac_098);
	mytree->Branch("mrcnn_frac_09999",&mrcnn_frac_09999);
	mytree->Branch("combined_frac_000",&combined_frac_000);
	mytree->Branch("combined_frac_020",&combined_frac_020);
	mytree->Branch("combined_frac_030",&combined_frac_030);
	mytree->Branch("combined_frac_040",&combined_frac_040);
	mytree->Branch("combined_frac_050",&combined_frac_050);
	mytree->Branch("combined_frac_060",&combined_frac_060);
	mytree->Branch("combined_frac_070",&combined_frac_070);
	mytree->Branch("combined_frac_080",&combined_frac_080);
	mytree->Branch("combined_frac_090",&combined_frac_090);
	mytree->Branch("combined_frac_095",&combined_frac_095);
	mytree->Branch("combined_frac_098",&combined_frac_098);
	mytree->Branch("combined_frac_09999",&combined_frac_09999);

	mytree->Branch("hip_count_030", &hip_count_030 );
	mytree->Branch("hip_count_050", &hip_count_050 );
	mytree->Branch("hip_count_070", &hip_count_070 );
	mytree->Branch("hip_count_090", &hip_count_090 );
	mytree->Branch("mip_count_030", &mip_count_030 );
	mytree->Branch("mip_count_050", &mip_count_050 );
	mytree->Branch("mip_count_070", &mip_count_070 );
	mytree->Branch("mip_count_090", &mip_count_090 );
	mytree->Branch("shower_count_030", &shower_count_030 );
	mytree->Branch("shower_count_050", &shower_count_050 );
	mytree->Branch("shower_count_070", &shower_count_070 );
	mytree->Branch("shower_count_090", &shower_count_090 );
	mytree->Branch("delta_count_030", &delta_count_030 );
	mytree->Branch("delta_count_050", &delta_count_050 );
	mytree->Branch("delta_count_070", &delta_count_070 );
	mytree->Branch("delta_count_090", &delta_count_090 );
	mytree->Branch("michel_count_030", &michel_count_030 );
	mytree->Branch("michel_count_050", &michel_count_050 );
	mytree->Branch("michel_count_070", &michel_count_070 );
	mytree->Branch("michel_count_090", &michel_count_090 );

	if ((mode == 1) || (mode==3) || (mode==5)){
		mytree->Branch("adc_pix_count_box21",&adc_pix_count_box21);
		mytree->Branch("wc_frac_box21",&wc_frac_box21);
		mytree->Branch("mrcnn_frac_000_box21",&mrcnn_frac_000_box21);
		mytree->Branch("mrcnn_frac_020_box21",&mrcnn_frac_020_box21);
		mytree->Branch("mrcnn_frac_030_box21",&mrcnn_frac_030_box21);
		mytree->Branch("mrcnn_frac_040_box21",&mrcnn_frac_040_box21);
		mytree->Branch("mrcnn_frac_050_box21",&mrcnn_frac_050_box21);
		mytree->Branch("mrcnn_frac_060_box21",&mrcnn_frac_060_box21);
		mytree->Branch("mrcnn_frac_070_box21",&mrcnn_frac_070_box21);
		mytree->Branch("mrcnn_frac_080_box21",&mrcnn_frac_080_box21);
		mytree->Branch("mrcnn_frac_090_box21",&mrcnn_frac_090_box21);
		mytree->Branch("mrcnn_frac_095_box21",&mrcnn_frac_095_box21);
		mytree->Branch("mrcnn_frac_098_box21",&mrcnn_frac_098_box21);
		mytree->Branch("mrcnn_frac_09999_box21",&mrcnn_frac_09999_box21);
		mytree->Branch("combined_frac_000_box21",&combined_frac_000_box21);
		mytree->Branch("combined_frac_020_box21",&combined_frac_020_box21);
		mytree->Branch("combined_frac_030_box21",&combined_frac_030_box21);
		mytree->Branch("combined_frac_040_box21",&combined_frac_040_box21);
		mytree->Branch("combined_frac_050_box21",&combined_frac_050_box21);
		mytree->Branch("combined_frac_060_box21",&combined_frac_060_box21);
		mytree->Branch("combined_frac_070_box21",&combined_frac_070_box21);
		mytree->Branch("combined_frac_080_box21",&combined_frac_080_box21);
		mytree->Branch("combined_frac_090_box21",&combined_frac_090_box21);
		mytree->Branch("combined_frac_095_box21",&combined_frac_095_box21);
		mytree->Branch("combined_frac_098_box21",&combined_frac_098_box21);
		mytree->Branch("combined_frac_09999_box21",&combined_frac_09999_box21);

		mytree->Branch("adc_pix_count_segment",&adc_pix_count_segment);
		mytree->Branch("wc_frac_segment",&wc_frac_segment);
		mytree->Branch("mrcnn_frac_segment_000",&mrcnn_frac_segment_000);
		mytree->Branch("mrcnn_frac_segment_020",&mrcnn_frac_segment_020);
		mytree->Branch("mrcnn_frac_segment_030",&mrcnn_frac_segment_030);
		mytree->Branch("mrcnn_frac_segment_040",&mrcnn_frac_segment_040);
		mytree->Branch("mrcnn_frac_segment_050",&mrcnn_frac_segment_050);
		mytree->Branch("mrcnn_frac_segment_060",&mrcnn_frac_segment_060);
		mytree->Branch("mrcnn_frac_segment_070",&mrcnn_frac_segment_070);
		mytree->Branch("mrcnn_frac_segment_080",&mrcnn_frac_segment_080);
		mytree->Branch("mrcnn_frac_segment_090",&mrcnn_frac_segment_090);
		mytree->Branch("mrcnn_frac_segment_095",&mrcnn_frac_segment_095);
		mytree->Branch("mrcnn_frac_segment_098",&mrcnn_frac_segment_098);
		mytree->Branch("mrcnn_frac_segment_09999",&mrcnn_frac_segment_09999);
		mytree->Branch("combined_frac_segment_000",&combined_frac_segment_000);
		mytree->Branch("combined_frac_segment_020",&combined_frac_segment_020);
		mytree->Branch("combined_frac_segment_030",&combined_frac_segment_030);
		mytree->Branch("combined_frac_segment_040",&combined_frac_segment_040);
		mytree->Branch("combined_frac_segment_050",&combined_frac_segment_050);
		mytree->Branch("combined_frac_segment_060",&combined_frac_segment_060);
		mytree->Branch("combined_frac_segment_070",&combined_frac_segment_070);
		mytree->Branch("combined_frac_segment_080",&combined_frac_segment_080);
		mytree->Branch("combined_frac_segment_090",&combined_frac_segment_090);
		mytree->Branch("combined_frac_segment_095",&combined_frac_segment_095);
		mytree->Branch("combined_frac_segment_098",&combined_frac_segment_098);
		mytree->Branch("combined_frac_segment_09999",&combined_frac_segment_09999);

		mytree->Branch("hip_count_segment_030", &hip_count_segment_030 );
		mytree->Branch("hip_count_segment_050", &hip_count_segment_050 );
		mytree->Branch("hip_count_segment_070", &hip_count_segment_070 );
		mytree->Branch("hip_count_segment_090", &hip_count_segment_090 );
		mytree->Branch("mip_count_segment_030", &mip_count_segment_030 );
		mytree->Branch("mip_count_segment_050", &mip_count_segment_050 );
		mytree->Branch("mip_count_segment_070", &mip_count_segment_070 );
		mytree->Branch("mip_count_segment_090", &mip_count_segment_090 );
		mytree->Branch("shower_count_segment_030", &shower_count_segment_030 );
		mytree->Branch("shower_count_segment_050", &shower_count_segment_050 );
		mytree->Branch("shower_count_segment_070", &shower_count_segment_070 );
		mytree->Branch("shower_count_segment_090", &shower_count_segment_090 );
		mytree->Branch("delta_count_segment_030", &delta_count_segment_030 );
		mytree->Branch("delta_count_segment_050", &delta_count_segment_050 );
		mytree->Branch("delta_count_segment_070", &delta_count_segment_070 );
		mytree->Branch("delta_count_segment_090", &delta_count_segment_090 );
		mytree->Branch("michel_count_segment_030", &michel_count_segment_030 );
		mytree->Branch("michel_count_segment_050", &michel_count_segment_050 );
		mytree->Branch("michel_count_segment_070", &michel_count_segment_070 );
		mytree->Branch("michel_count_segment_090", &michel_count_segment_090 );
	}

	int file_idx = 0;
	while(std::getline(infile_mrcnn,STRING_MRCNN)){ // To get you all the lines.
		std::getline(infile_dlreco,STRING_DLRECO);
		if (file_idx < start_file) {
			file_idx++;
			continue;
		}
		if (file_idx >= file_limit) break;
		// For string formatting of file, most are full path strings, some are just filename, no path
		if (mode == 0){
			STRING_MRCNN  = "/cluster/tufts/wongjiradlab/jmills09/mrcnn_processed_outs/mcc9_v29e_dl_run3_G1_extbnb_dlana/"+STRING_MRCNN;
			STRING_DLRECO = STRING_DLRECO;
		}
		else if ((mode==1) || (mode==2)){
			STRING_MRCNN = STRING_MRCNN;
			STRING_DLRECO = STRING_DLRECO;
		}
		else if ((mode==3) || (mode==4)){
			STRING_MRCNN = STRING_MRCNN;
			STRING_DLRECO = STRING_DLRECO;
		}
		else if ((mode==5) || (mode==6)){
			STRING_MRCNN = STRING_MRCNN;
			STRING_DLRECO = STRING_DLRECO;
		}
		else if (mode==7){
			STRING_MRCNN = STRING_MRCNN;
			STRING_DLRECO = STRING_DLRECO;
		}
		std::cout << file_idx << " " << STRING_MRCNN << "\n";
		std::cout << file_idx << " " << STRING_DLRECO << "\n";
		std::cout << "\n\n";
		file_idx++;
		gStyle->SetOptStat(0);

		larcv::IOManager* io_mrcnn  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_MRCNN", larcv::IOManager::kTickBackward);
		// io_mrcnn->reverse_all_products();
		std::string infile_mrcnn = STRING_MRCNN;
		io_mrcnn->add_in_file(infile_mrcnn);
		io_mrcnn->initialize();

		larcv::IOManager* io_dlreco  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_DLRECO", larcv::IOManager::kTickBackward);
		io_dlreco->reverse_all_products();
		std::string infile_dlreco = STRING_DLRECO;
		io_dlreco->add_in_file(infile_dlreco);
		io_dlreco->initialize();

		larlite::storage_manager ioll_dlreco  = larlite::storage_manager(larlite::storage_manager::kREAD);
    ioll_dlreco.add_in_filename(STRING_DLRECO);
    ioll_dlreco.open();

		int nentries_mrcnn = io_mrcnn->get_n_entries();
		int nentries_dlreco = io_dlreco->get_n_entries();

		std::cout << "\n\n\n";
		std::cout << nentries_mrcnn << " " << nentries_dlreco  << "	Entries\n";
		Cosmic_Products cosmic_products_getter;
		std::vector< larcv::ClusterMask > cluster_v ;

		larcv::logger larcv_logger;
		std::cout << "Level Setting:\n";
		std::cout << larcv_logger.level() << "\n";
		larcv::msg::Level_t log_level = larcv::msg::kCRITICAL;
		larcv_logger.force_level(log_level);
		std::cout << larcv_logger.level() << "\n";
		std::cout << "Level Set.\n";

		for (int i = 0;i<nentries_dlreco;i++){
		// for (int i = 4;i<5;i++){
			io_mrcnn->read_entry(i);
			io_dlreco->read_entry(i);
			ioll_dlreco.go_to(i);
			// LArCV Imports
			std::cout << "Entry	" << i << "\n";
			std::string producer_adc;
			// if (mode==0){
			// 	producer_adc = "wire";
			// }
			// else if (mode==1){
			// 	producer_adc = "segment";
			// }

			larcv::EventImage2D* ev_adc           = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "wire"));
			larcv::EventImage2D* ev_wc = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "thrumu"));
			larcv::EventClusterMask* ev_cmask     = (larcv::EventClusterMask*)(io_mrcnn->get_data(larcv::kProductClusterMask, "mask_proposals_y"));
			larcv::EventImage2D* ev_segment ;
			larcv::EventSparseImage* ev_spr_ssnet = (larcv::EventSparseImage*)(io_dlreco->get_data(larcv::kProductSparseImage, "sparseuresnetout"));

			run = ev_adc->run();
			subrun = ev_adc->subrun();
			event = ev_adc->event();
			std::vector< larcv::Image2D > img_v = ev_adc->Image2DArray();
			std::vector< larcv::Image2D > img_segment_v;
			std::vector< larcv::Image2D > img_ssnet_yplane_v = ev_spr_ssnet->SparseImageArray().at(2).as_Image2D();


			// int maxkd = 2;
			// int minsize = 2;
		 	// float maxdist = 2;
			// std::vector< cluster_t > cluster_v;
			// std::vector<std::vector<float> > hits_vv;
			// cluster_sdbscan_spacepoints(hits_vv, cluster_v, maxdist,minsize,maxkd);
			// return -1;

			// 0 -> HIP (Pions+ProtonsTruth)
			// 1 -> MIP (Muons)
			// 2 -> Shower
			// 3 -> Delta Ray
			// 4 -> Michel





			if ((mode==1) || (mode==3) || (mode==5)){
				ev_segment = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "segment"));
				img_segment_v = ev_segment->Image2DArray();
			}
			std::vector< larcv::Image2D > wc_v = ev_wc->Image2DArray();
			std::vector< std::vector< larcv::ClusterMask >> cmask_v = ev_cmask->as_vector();
			double adc_thresh = 10.0;
			double segment_thresh = 0.0;
			if ((mode==0) || (mode==7)) {adc_thresh = 10.0;}
			else if ((mode==1)||(mode==3) || (mode==5)) {adc_thresh = 10.0;}
			else if ((mode==2)||(mode==4) || (mode==6)) {std::cout << "Mode2/4/6 not implemented\n"; return -1;}

			// Look at ADC wire image
			std::vector< std::vector<double> > adc_pts_v = get_list_nonzero_pts(img_v[2],adc_thresh);
			larcv::Image2D maskrcnn_image(1008,3456);
			cosmic_products_getter.get_maskrcnn_image(maskrcnn_image,img_v[2],cmask_v[0],true);
			adc_pix_count = adc_pts_v.size();
			wc_frac = pixel_removal_fraction(adc_pts_v, wc_v[2],adc_thresh);
			mrcnn_frac_000 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.00);
			mrcnn_frac_020 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.20);
			mrcnn_frac_030 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.30);
			mrcnn_frac_040 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.40);
			mrcnn_frac_050 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.50);
			mrcnn_frac_060 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.60);
			mrcnn_frac_070 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.70);
			mrcnn_frac_080 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.80);
			mrcnn_frac_090 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.90);
			mrcnn_frac_095 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.95);
			mrcnn_frac_098 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.98);
			mrcnn_frac_09999 =pixel_removal_fraction(adc_pts_v, maskrcnn_image,adc_thresh,0.9999);
			combined_frac_000 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.0);
			combined_frac_020 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.20);
			combined_frac_030 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.30);
			combined_frac_040 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.40);
			combined_frac_050 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.50);
			combined_frac_060 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.60);
			combined_frac_070 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.70);
			combined_frac_080 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.80);
			combined_frac_090 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.90);
			combined_frac_095 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.95);
			combined_frac_098 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.98);
			combined_frac_09999 =pixel_removal_fraction(adc_pts_v, wc_v[2], maskrcnn_image,adc_thresh,0.0,0.9999);
			std::vector<double> ssnet_threshes = {0.3, 0.5, 0.7, 0.9};
			std::vector<int> hip_counts=    SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[0], adc_pts_v);
			std::vector<int> mip_counts=    SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[1], adc_pts_v);
			std::vector<int> shower_counts= SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[2], adc_pts_v);
			std::vector<int> delta_counts=  SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[3], adc_pts_v);
			std::vector<int> michel_counts= SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[4], adc_pts_v);
			hip_count_030=hip_counts[0];
			hip_count_050=hip_counts[1];
			hip_count_070=hip_counts[2];
			hip_count_090=hip_counts[3];
			mip_count_030=mip_counts[0];
			mip_count_050=mip_counts[1];
			mip_count_070=mip_counts[2];
			mip_count_090=mip_counts[3];
			shower_count_030=shower_counts[0];
			shower_count_050=shower_counts[1];
			shower_count_070=shower_counts[2];
			shower_count_090=shower_counts[3];
			delta_count_030=delta_counts[0];
			delta_count_050=delta_counts[1];
			delta_count_070=delta_counts[2];
			delta_count_090=delta_counts[3];
			michel_count_030=michel_counts[0];
			michel_count_050=michel_counts[1];
			michel_count_070=michel_counts[2];
			michel_count_090=michel_counts[3];



			if ((mode==1)||(mode==3) || (mode==5)){
				// Set up for looking at only 21x21 box around neutrino vertex
				ublarcvapp::mctools::NeutrinoVertex NeutrinoVertexFinder;
				std::vector<int> wire_tick = NeutrinoVertexFinder.getImageCoords(ioll_dlreco);
				larcv::Image2D maskrcnn_image_segment(1008,3456);
				cosmic_products_getter.get_maskrcnn_image(maskrcnn_image_segment,img_segment_v[2],cmask_v[0],true);
				if ((wire_tick[3] < 8448) && (wire_tick[3] >= 2400)){
					std::vector<int> row_cols = {(int)wc_v[2].meta().row(wire_tick[3]), wire_tick[0], wire_tick[1] , wire_tick[2]};
					std::vector< std::vector<double> > adc_pts_v_box21 = get_list_nonzero_pts(img_segment_v[2],segment_thresh,row_cols[0]-10,row_cols[0]+11,row_cols[3]-10, row_cols[3]+11);
					adc_pix_count_box21 = adc_pts_v_box21.size();
					wc_frac_box21 = pixel_removal_fraction(adc_pts_v_box21, wc_v[2],segment_thresh);
					mrcnn_frac_000_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.00);
					mrcnn_frac_020_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.20);
					mrcnn_frac_030_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.30);
					mrcnn_frac_040_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.40);
					mrcnn_frac_050_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.50);
					mrcnn_frac_060_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.60);
					mrcnn_frac_070_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.70);
					mrcnn_frac_080_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.80);
					mrcnn_frac_090_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.90);
					mrcnn_frac_095_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.95);
					mrcnn_frac_098_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.98);
					mrcnn_frac_09999_box21 =pixel_removal_fraction(adc_pts_v_box21, maskrcnn_image_segment,segment_thresh,0.9999);
					combined_frac_000_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.0);
					combined_frac_020_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.20);
					combined_frac_030_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.30);
					combined_frac_040_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.40);
					combined_frac_050_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.50);
					combined_frac_060_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.60);
					combined_frac_070_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.70);
					combined_frac_080_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.80);
					combined_frac_090_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.90);
					combined_frac_095_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.95);
					combined_frac_098_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.98);
					combined_frac_09999_box21 =pixel_removal_fraction(adc_pts_v_box21, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.9999);
				}
				std::vector< std::vector<double> > adc_pts_v_segment = get_list_nonzero_pts(img_segment_v[2],segment_thresh);
				adc_pix_count_segment = adc_pts_v_segment.size();
				wc_frac_segment = pixel_removal_fraction(adc_pts_v_segment, wc_v[2],segment_thresh);
				mrcnn_frac_segment_000 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.00);
				mrcnn_frac_segment_020 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.20);
				mrcnn_frac_segment_030 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.30);
				mrcnn_frac_segment_040 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.40);
				mrcnn_frac_segment_050 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.50);
				mrcnn_frac_segment_060 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.60);
				mrcnn_frac_segment_070 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.70);
				mrcnn_frac_segment_080 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.80);
				mrcnn_frac_segment_090 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.90);
				mrcnn_frac_segment_095 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.95);
				mrcnn_frac_segment_098 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.98);
				mrcnn_frac_segment_09999 =pixel_removal_fraction(adc_pts_v_segment, maskrcnn_image_segment,segment_thresh,0.9999);
				combined_frac_segment_000 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.0);
				combined_frac_segment_020 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.20);
				combined_frac_segment_030 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.30);
				combined_frac_segment_040 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.40);
				combined_frac_segment_050 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.50);
				combined_frac_segment_060 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.60);
				combined_frac_segment_070 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.70);
				combined_frac_segment_080 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.80);
				combined_frac_segment_090 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.90);
				combined_frac_segment_095 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.95);
				combined_frac_segment_098 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.98);
				combined_frac_segment_09999 =pixel_removal_fraction(adc_pts_v_segment, wc_v[2], maskrcnn_image_segment,segment_thresh,0.0,0.9999);

				std::vector<int> hip_counts_segment=    SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[0], adc_pts_v_segment);
				std::vector<int> mip_counts_segment=    SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[1], adc_pts_v_segment);
				std::vector<int> shower_counts_segment= SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[2], adc_pts_v_segment);
				std::vector<int> delta_counts_segment=  SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[3], adc_pts_v_segment);
				std::vector<int> michel_counts_segment= SSNET_Thresh_Counts(ssnet_threshes, img_ssnet_yplane_v[4], adc_pts_v_segment);
				hip_count_segment_030=hip_counts_segment[0];
				hip_count_segment_050=hip_counts_segment[1];
				hip_count_segment_070=hip_counts_segment[2];
				hip_count_segment_090=hip_counts_segment[3];
				mip_count_segment_030=mip_counts_segment[0];
				mip_count_segment_050=mip_counts_segment[1];
				mip_count_segment_070=mip_counts_segment[2];
				mip_count_segment_090=mip_counts_segment[3];
				shower_count_segment_030=shower_counts_segment[0];
				shower_count_segment_050=shower_counts_segment[1];
				shower_count_segment_070=shower_counts_segment[2];
				shower_count_segment_090=shower_counts_segment[3];
				delta_count_segment_030=delta_counts_segment[0];
				delta_count_segment_050=delta_counts_segment[1];
				delta_count_segment_070=delta_counts_segment[2];
				delta_count_segment_090=delta_counts_segment[3];
				michel_count_segment_030=michel_counts_segment[0];
				michel_count_segment_050=michel_counts_segment[1];
				michel_count_segment_070=michel_counts_segment[2];
				michel_count_segment_090=michel_counts_segment[3];
			}


			mytree->Fill();
			// std::cout << adc_pts_v.size() << "	Num adc pts\n";
			// std::cout << "Wirecell    Frac:	" << wc_frac  << "\n";
			// std::cout << "MRCNN   050 Frac:	" << mrcnn_frac_050  << "\n";
			// std::cout << "Combine 050 Frac:	" << combined_frac_050  << "\n";

			// make_event_disp(img_trueadc_v[2],"test_adc_raw");
			// make_event_disp(img_v[2],"test_segment_raw");
			// make_event_disp(wc_v[2],"test_wc");
			// make_event_disp(maskrcnn_image, "test_mrcnn",1.0);
			// make_event_disp(maskrcnn_image_full, "test_mrcnn_full",1.0);

		}//end of entry loop
		io_mrcnn->finalize();
		delete io_mrcnn;
		io_dlreco->finalize();
		delete io_dlreco;

		std::cout << "\n\n\n";
	}//End of mrcnn file

	myfile->cd();
	mytree->Write();
	myfile->Close();
	// delete mytree;
	delete myfile;
	print_signal();
	return 0;
	}//End of main

void print_signal(){
	std::cout << "\n\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "\n\n";

	std::cout << "         _==/            i     i           \\==_ \n";
	std::cout << "        /XX/             |\\___/|            \\XX\\    \n";
	std::cout << "       /XXXX\\            |XXXXX|            /XXXX\\   \n";
	std::cout << "      |XXXXXX\\_         _XXXXXXX_         _/XXXXXX|   \n";
	std::cout << "     XXXXXXXXXXXxxxxxxxXXXXXXXXXXXxxxxxxxXXXXXXXXXXX   \n";
	std::cout << "    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|  \n";
	std::cout << "    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   \n";
	std::cout << "    |XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|   \n";
	std::cout << "     XXXXXX/^^^^^\\XXXXXXXXXXXXXXXXXXXXX/^^^^^\\XXXXXX    \n";
	std::cout << "      |XXX|       \\XXX/^^\\XXXXX/^^\\XXX/       |XXX|    \n";
	std::cout << "       \\XX\\        \\X/    \\XXX/    \\X/       /XX/    \n";
	std::cout << "           \\        |      \\X/      |       /     \n";
	std::cout << "\n\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "/////////////////////////////////////////////////////////////\n";
	std::cout << "\n\n";
	return;
	}

	// TH1D wc_masking_eff       =TH1D("WC_Masking_Eff","WC_Masking_Eff ",100,0,1.0001);
	// TH1D mrcnn_masking_eff    =TH1D("MRCNN_Masking_Eff","MRCNN_Masking_Eff ",100,0,1.0001);
	// TH1D combined_masking_eff =TH1D("Combined_Masking_Eff","Combined_Masking_Eff ",100,0,1.0001);
	// wc_masking_eff.SetTitle("WC_Masking_Eff");
	// wc_masking_eff.SetXTitle("Efficiency");
	// wc_masking_eff.SetYTitle("Event Count");
	// wc_masking_eff.SetLineColor(2);
	// mrcnn_masking_eff.SetTitle("MRCNN_Masking_Eff");
	// mrcnn_masking_eff.SetXTitle("Efficiency");
	// mrcnn_masking_eff.SetYTitle("Event Count");
	// mrcnn_masking_eff.SetLineColor(4);
	// combined_masking_eff.SetTitle("Combined_Masking_Eff");
	// combined_masking_eff.SetXTitle("Efficiency");
	// combined_masking_eff.SetYTitle("Event Count");
	// combined_masking_eff.SetLineColor(1);
}
}//End namespaces
#endif
