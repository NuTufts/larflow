#ifndef __CRVARSMAKER_CXX__
#define __CRVARSMAKER_CXX__


#include "CRVarsMaker.h"

// Misc

namespace larflow {
namespace cosmicremovalsuite {

int CRVarsMaker::run_varsmaker_rootfile(bool IsMC,
				std::string infile_dlreco,
				std::string infile_mrcnn,
				std::string OutFileName,
				std::string OutTreeName
			){


	std::cout << "Hello world " << "\n";

	TFile *myfile = new TFile(OutFileName.c_str(),"RECREATE");
	myfile->cd();
	TTree *mytree = new TTree(OutTreeName.c_str(),OutTreeName.c_str());


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
	double mcvtx_x =-9999;
	double mcvtx_y =-9999;
	double mcvtx_z =-9999;
	double mcvtx_contained =-1;



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

	if (IsMC == true){
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

		mytree->Branch("mcvtx_x", &mcvtx_x );
		mytree->Branch("mcvtx_y", &mcvtx_y );
		mytree->Branch("mcvtx_z", &mcvtx_z );
		mytree->Branch("mcvtx_contained", &mcvtx_contained );

	}

	std::cout  << "MRCNN File:   " << infile_mrcnn << "\n";
	std::cout  << "DLRECO FILE:  " << infile_dlreco << "\n";
	std::cout << "\n";
	gStyle->SetOptStat(0);

	larcv::IOManager* io_mrcnn  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_MRCNN", larcv::IOManager::kTickBackward);
	io_mrcnn->add_in_file(infile_mrcnn);
	io_mrcnn->initialize();

	larcv::IOManager* io_dlreco  = new larcv::IOManager(larcv::IOManager::kREAD,"IOManager_DLRECO", larcv::IOManager::kTickBackward);
	io_dlreco->reverse_all_products();
	io_dlreco->add_in_file(infile_dlreco);
	io_dlreco->initialize();

	larlite::storage_manager ioll_dlreco  = larlite::storage_manager(larlite::storage_manager::kREAD);
  ioll_dlreco.add_in_filename(infile_dlreco);
  ioll_dlreco.open();

	int nentries_mrcnn = io_mrcnn->get_n_entries();
	int nentries_dlreco = io_dlreco->get_n_entries();

	std::cout << "\n\n\n";
	std::cout << nentries_mrcnn << " " << nentries_dlreco  << "	Entries\n";
	Cosmic_Products cosmic_products_getter;
	std::vector< larcv::ClusterMask > cluster_v ;

	larcv::logger larcv_logger;
	std::cout << "LArCV Logger Level Setting:\n";
	std::cout << larcv_logger.level() << "\n";
	larcv::msg::Level_t log_level = larcv::msg::kCRITICAL;
	larcv_logger.force_level(log_level);
	std::cout << larcv_logger.level() << "\n";
	std::cout << "LArCV Logger Level Set.\n\n";

	for (int i = 0;i<nentries_dlreco;i++){
		io_mrcnn->read_entry(i);
		io_dlreco->read_entry(i);
		ioll_dlreco.go_to(i);
		// LArCV Imports
		std::cout << "Entry	" << i << "\n";
		std::string producer_adc;

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





		if (IsMC == true){
			ev_segment = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "segment"));
			img_segment_v = ev_segment->Image2DArray();
		}
		std::vector< larcv::Image2D > wc_v = ev_wc->Image2DArray();
		std::vector< std::vector< larcv::ClusterMask >> cmask_v = ev_cmask->as_vector();
		double adc_thresh = 10.0;
		double segment_thresh = 0.0;


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



		if (IsMC == true){
			// Set up for looking at only 21x21 box around neutrino vertex
			ublarcvapp::mctools::NeutrinoVertex NeutrinoVertexFinder;
			std::vector<int> wire_tick = NeutrinoVertexFinder.getImageCoords(ioll_dlreco);

			larlite::event_mctruth* ev_mctruth = (larlite::event_mctruth*)ioll_dlreco.get_data(larlite::data::kMCTruth,"generator");
			auto const& mctruth = ev_mctruth->at(0);
			const larlite::mcstep& start = mctruth.GetNeutrino().Nu().Trajectory().front();
			std::vector<double> pos3d = {start.X(), start.Y(), start.Z()};
			mcvtx_x = pos3d[0];
			mcvtx_y = pos3d[1];
			mcvtx_z = pos3d[2];
			mcvtx_contained = IsInsideBoundaries(pos3d);



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

	}//end of entry loop
	io_mrcnn->finalize();
	delete io_mrcnn;
	io_dlreco->finalize();
	delete io_dlreco;

	std::cout << "\n\n\n";

	myfile->cd();
	mytree->Write();
	myfile->Close();
	// delete mytree;
	delete myfile;
	print_signal();
	return 0;
}//End of run_varsmaker_rootfile

std::vector<double> CRVarsMaker::run_varsmaker_arrsout(bool IsMC,
				larcv::IOManager* io_mrcnn,
				larcv::IOManager* io_dlreco,
				larlite::storage_manager& ioll_dlreco,
				int entry_num
			){
	std::vector<double> cosmic_thresh_v = {0.0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98};

	double run=-1;
	double subrun=-1;
	double event=-1;
	double adc_pix_count=-1;
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

	double hip_count_030=-1;
	double hip_count_050=-1;
	double hip_count_070=-1;
	double hip_count_090=-1;
	double mip_count_030=-1;
	double mip_count_050=-1;
	double mip_count_070=-1;
	double mip_count_090=-1;
	double shower_count_030=-1;
	double shower_count_050=-1;
	double shower_count_070=-1;
	double shower_count_090=-1;
	double delta_count_030=-1;
	double delta_count_050=-1;
	double delta_count_070=-1;
	double delta_count_090=-1;
	double michel_count_030=-1;
	double michel_count_050=-1;
	double michel_count_070=-1;
	double michel_count_090=-1;

	double adc_pix_count_segment=-1;
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

	double hip_count_segment_030=-1;
	double hip_count_segment_050=-1;
	double hip_count_segment_070=-1;
	double hip_count_segment_090=-1;
	double mip_count_segment_030=-1;
	double mip_count_segment_050=-1;
	double mip_count_segment_070=-1;
	double mip_count_segment_090=-1;
	double shower_count_segment_030=-1;
	double shower_count_segment_050=-1;
	double shower_count_segment_070=-1;
	double shower_count_segment_090=-1;
	double delta_count_segment_030=-1;
	double delta_count_segment_050=-1;
	double delta_count_segment_070=-1;
	double delta_count_segment_090=-1;
	double michel_count_segment_030=-1;
	double michel_count_segment_050=-1;
	double michel_count_segment_070=-1;
	double michel_count_segment_090=-1;

	double adc_pix_count_box21=-1;
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
	double mcvtx_x =-9999;
	double mcvtx_y =-9999;
	double mcvtx_z =-9999;
	double mcvtx_contained =-1;

	std::vector< larcv::ClusterMask > cluster_v ;

	larcv::logger larcv_logger;
	// std::cout << "LArCV Logger Level Setting:\n";
	// std::cout << larcv_logger.level() << "\n";
	larcv::msg::Level_t log_level = larcv::msg::kCRITICAL;
	larcv_logger.force_level(log_level);
	// std::cout << larcv_logger.level() << "\n";
	// std::cout << "LArCV Logger Level Set.\n\n";


	// LArCV Imports
	if (entry_num != -1){
		std::cout << "Setting Entry to	" << entry_num << "\n";
		io_mrcnn->read_entry(entry_num);
		io_dlreco->read_entry(entry_num);
		ioll_dlreco.go_to(entry_num);
	}
	std::string producer_adc;

	larcv::EventImage2D* ev_adc           = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "wire"));
	larcv::EventImage2D* ev_wc            = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "thrumu"));
	larcv::EventClusterMask* ev_cmask     = (larcv::EventClusterMask*)(io_mrcnn->get_data(larcv::kProductClusterMask, "mask_proposals_y"));
	larcv::EventSparseImage* ev_spr_ssnet = (larcv::EventSparseImage*)(io_dlreco->get_data(larcv::kProductSparseImage, "sparseuresnetout"));
	larcv::EventImage2D* ev_segment ;

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

	if (IsMC == true){
		ev_segment = (larcv::EventImage2D*)(io_dlreco->get_data(larcv::kProductImage2D, "segment"));
		img_segment_v = ev_segment->Image2DArray();
	}
	std::vector< larcv::Image2D > wc_v = ev_wc->Image2DArray();
	std::vector< std::vector< larcv::ClusterMask >> cmask_v = ev_cmask->as_vector();
	double adc_thresh = 10.0;
	double segment_thresh = 0.0;

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



	if (IsMC == true){
		// Set up for looking at only 21x21 box around neutrino vertex
		ublarcvapp::mctools::NeutrinoVertex NeutrinoVertexFinder;
		std::vector<int> wire_tick = NeutrinoVertexFinder.getImageCoords(ioll_dlreco);

		larlite::event_mctruth* ev_mctruth = (larlite::event_mctruth*)ioll_dlreco.get_data(larlite::data::kMCTruth,"generator");
		auto const& mctruth = ev_mctruth->at(0);
		const larlite::mcstep& start = mctruth.GetNeutrino().Nu().Trajectory().front();
		std::vector<double> pos3d = {start.X(), start.Y(), start.Z()};
		mcvtx_x = pos3d[0];
		mcvtx_y = pos3d[1];
		mcvtx_z = pos3d[2];
		mcvtx_contained = IsInsideBoundaries(pos3d);



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

	std::vector<double> output_vector(125);


	output_vector[0]		=		run;
	output_vector[1]		=		subrun;
	output_vector[2]		=		event;
	output_vector[3]		=		adc_pix_count;
	output_vector[4]		=		wc_frac;
	output_vector[5]		=		mrcnn_frac_000;
	output_vector[6]		=		mrcnn_frac_020;
	output_vector[7]		=		mrcnn_frac_030;
	output_vector[8]		=		mrcnn_frac_040;
	output_vector[9]		=		mrcnn_frac_050;
	output_vector[10]		=		mrcnn_frac_060;
	output_vector[11]		=		mrcnn_frac_070;
	output_vector[12]		=		mrcnn_frac_080;
	output_vector[13]		=		mrcnn_frac_090;
	output_vector[14]		=		mrcnn_frac_095;
	output_vector[15]		=		mrcnn_frac_098;
	output_vector[16]		=		mrcnn_frac_09999;
	output_vector[17]		=		combined_frac_000;
	output_vector[18]		=		combined_frac_020;
	output_vector[19]		=		combined_frac_030;
	output_vector[20]		=		combined_frac_040;
	output_vector[21]		=		combined_frac_050;
	output_vector[22]		=		combined_frac_060;
	output_vector[23]		=		combined_frac_070;
	output_vector[24]		=		combined_frac_080;
	output_vector[25]		=		combined_frac_090;
	output_vector[26]		=		combined_frac_095;
	output_vector[27]		=		combined_frac_098;
	output_vector[28]		=		combined_frac_09999;
	output_vector[29]		=		hip_count_030;
	output_vector[30]		=		hip_count_050;
	output_vector[31]		=		hip_count_070;
	output_vector[32]		=		hip_count_090;
	output_vector[33]		=		mip_count_030;
	output_vector[34]		=		mip_count_050;
	output_vector[35]		=		mip_count_070;
	output_vector[36]		=		mip_count_090;
	output_vector[37]		=		shower_count_030;
	output_vector[38]		=		shower_count_050;
	output_vector[39]		=		shower_count_070;
	output_vector[40]		=		shower_count_090;
	output_vector[41]		=		delta_count_030;
	output_vector[42]		=		delta_count_050;
	output_vector[43]		=		delta_count_070;
	output_vector[44]		=		delta_count_090;
	output_vector[45]		=		michel_count_030;
	output_vector[46]		=		michel_count_050;
	output_vector[47]		=		michel_count_070;
	output_vector[48]		=		michel_count_090;
	output_vector[49]		=		adc_pix_count_segment;
	output_vector[50]		=		wc_frac_segment;
	output_vector[51]		=		mrcnn_frac_segment_000;
	output_vector[52]		=		mrcnn_frac_segment_020;
	output_vector[53]		=		mrcnn_frac_segment_030;
	output_vector[54]		=		mrcnn_frac_segment_040;
	output_vector[55]		=		mrcnn_frac_segment_050;
	output_vector[56]		=		mrcnn_frac_segment_060;
	output_vector[57]		=		mrcnn_frac_segment_070;
	output_vector[58]		=		mrcnn_frac_segment_080;
	output_vector[59]		=		mrcnn_frac_segment_090;
	output_vector[60]		=		mrcnn_frac_segment_095;
	output_vector[61]		=		mrcnn_frac_segment_098;
	output_vector[62]		=		mrcnn_frac_segment_09999;
	output_vector[63]		=		combined_frac_segment_000;
	output_vector[64]		=		combined_frac_segment_020;
	output_vector[65]		=		combined_frac_segment_030;
	output_vector[66]		=		combined_frac_segment_040;
	output_vector[67]		=		combined_frac_segment_050;
	output_vector[68]		=		combined_frac_segment_060;
	output_vector[69]		=		combined_frac_segment_070;
	output_vector[70]		=		combined_frac_segment_080;
	output_vector[71]		=		combined_frac_segment_090;
	output_vector[72]		=		combined_frac_segment_095;
	output_vector[73]		=		combined_frac_segment_098;
	output_vector[74]		=		combined_frac_segment_09999;
	output_vector[75]		=		hip_count_segment_030;
	output_vector[76]		=		hip_count_segment_050;
	output_vector[77]		=		hip_count_segment_070;
	output_vector[78]		=		hip_count_segment_090;
	output_vector[79]		=		mip_count_segment_030;
	output_vector[80]		=		mip_count_segment_050;
	output_vector[81]		=		mip_count_segment_070;
	output_vector[82]		=		mip_count_segment_090;
	output_vector[83]		=		shower_count_segment_030;
	output_vector[84]		=		shower_count_segment_050;
	output_vector[85]		=		shower_count_segment_070;
	output_vector[86]		=		shower_count_segment_090;
	output_vector[87]		=		delta_count_segment_030;
	output_vector[88]		=		delta_count_segment_050;
	output_vector[89]		=		delta_count_segment_070;
	output_vector[90]		=		delta_count_segment_090;
	output_vector[91]		=		michel_count_segment_030;
	output_vector[92]		=		michel_count_segment_050;
	output_vector[93]		=		michel_count_segment_070;
	output_vector[94]		=		michel_count_segment_090;
	output_vector[95]		=		adc_pix_count_box21;
	output_vector[96]		=		wc_frac_box21;
	output_vector[97]		=		mrcnn_frac_000_box21;
	output_vector[98]		=		mrcnn_frac_020_box21;
	output_vector[99]		=		mrcnn_frac_030_box21;
	output_vector[100]		=		mrcnn_frac_040_box21;
	output_vector[101]		=		mrcnn_frac_050_box21;
	output_vector[102]		=		mrcnn_frac_060_box21;
	output_vector[103]		=		mrcnn_frac_070_box21;
	output_vector[104]		=		mrcnn_frac_080_box21;
	output_vector[105]		=		mrcnn_frac_090_box21;
	output_vector[106]		=		mrcnn_frac_095_box21;
	output_vector[107]		=		mrcnn_frac_098_box21;
	output_vector[108]		=		mrcnn_frac_09999_box21;
	output_vector[109]		=		combined_frac_000_box21;
	output_vector[110]		=		combined_frac_020_box21;
	output_vector[111]		=		combined_frac_030_box21;
	output_vector[112]		=		combined_frac_040_box21;
	output_vector[113]		=		combined_frac_050_box21;
	output_vector[114]		=		combined_frac_060_box21;
	output_vector[115]		=		combined_frac_070_box21;
	output_vector[116]		=		combined_frac_080_box21;
	output_vector[117]		=		combined_frac_090_box21;
	output_vector[118]		=		combined_frac_095_box21;
	output_vector[119]		=		combined_frac_098_box21;
	output_vector[120]		=		combined_frac_09999_box21;
	output_vector[121]		=		mcvtx_x;
	output_vector[122]		=		mcvtx_y;
	output_vector[123]		=		mcvtx_z;
	output_vector[124]		=		mcvtx_contained;


	std::cout << "\n\n\n";


	print_signal();
	return output_vector;
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

}
}//End namespaces
#endif
