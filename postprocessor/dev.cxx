#include <iostream>
#include <string>

// ROOT
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include <TApplication.h>

// larlite
#include "DataFormat/hit.h"
#include "DataFormat/opflash.h"
#include "DataFormat/spacepoint.h"
#include "DataFormat/larflow3dhit.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

// larlitecv
#include "Base/DataCoordinator.h"

// #ifdef USE_OPENCV
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #endif

// ContourTools
#include "ContourTools/ContourCluster.h"

// FlowContourMatching
#include "FlowContourMatching/FlowContourMatch.h"




void event_changeout( larlite::storage_manager& dataco_output,
		      larlitecv::DataCoordinator& dataco_whole,
		      larlitecv::DataCoordinator& dataco_larlite,
		      larflow::FlowContourMatch& matching_algo,
		      const int runid,
		      const int subrunid,
		      const int eventid,
		      bool makehits_useunmatched,
		      bool makehits_require_3dconsistency,
		      bool hasopreco, bool hasmcreco) {
  
  std::cout << "event_changeout." << std::endl;
  // save the larflow3dhit vector
  larlite::event_larflow3dhit* ev_larflowhit = (larlite::event_larflow3dhit*)dataco_output.get_data(larlite::data::kLArFlow3DHit,"flowhits");
  auto evout_opflash_beam   = (larlite::event_opflash*)dataco_output.get_data(larlite::data::kOpFlash, "simpleFlashBeam" );
  auto evout_opflash_cosmic = (larlite::event_opflash*)dataco_output.get_data(larlite::data::kOpFlash, "simpleFlashCosmic" );
    
  // get mctrack
  const larlite::event_mctrack* ev_track = nullptr;
  if ( hasmcreco ) {
    dataco_larlite.goto_event( runid, subrunid, eventid, "larlite" );
    ev_track = (larlite::event_mctrack*)dataco_larlite.get_larlite_data(larlite::data::kMCTrack, "mcreco");
  }
  
  // get supera images
  dataco_whole.goto_event( runid, subrunid, eventid, "larcv" );
  larcv::EventImage2D* ev_wholeimg  = (larcv::EventImage2D*) dataco_whole.get_larcv_data("image2d","wire");
  // fill mctruth
  if ( hasmcreco )
    matching_algo.mctrack_match(*ev_track,ev_wholeimg->as_vector());
  
  // get opreco to save into output file
  if ( hasopreco ) {
    auto ev_opflash_beam   = (larlite::event_opflash*)dataco_larlite.get_larlite_data(larlite::data::kOpFlash, "simpleFlashBeam" );
    auto ev_opflash_cosmic = (larlite::event_opflash*)dataco_larlite.get_larlite_data(larlite::data::kOpFlash, "simpleFlashCosmic" );
    
    for ( auto& flash : *ev_opflash_beam )
      evout_opflash_beam->emplace_back( std::move(flash) );
    for ( auto& flash : *ev_opflash_cosmic )
      evout_opflash_cosmic->emplace_back( std::move(flash) );
  }
  
  // get the final hits made from flow
  std::vector< larlite::larflow3dhit > whole_event_hits3d_v = matching_algo.get3Dhits_2pl( makehits_useunmatched, makehits_require_3dconsistency );
  
  std::cout << "Number of 3D (2-flow) hits: " << whole_event_hits3d_v.size() << std::endl;
  for ( auto& flowhit : whole_event_hits3d_v ) {
    // form spacepoints
    ev_larflowhit->emplace_back( std::move(flowhit) );
  }
  
  dataco_output.set_id( runid, subrunid, eventid );
  dataco_output.next_event();
  
  return;
}

larcv::Image2D stitch_infill(larlitecv::DataCoordinator& dataco_crop,
			     larlitecv::DataCoordinator& dataco_whole,
			     larflow::FlowContourMatch& matching_algo,
			     int run, int subrun, int event, int entry){

  //infill prediction (unmasked)
  larcv::EventImage2D* ev_infill    = (larcv::EventImage2D*) dataco_crop.get_larcv_data("image2d", "infillCropped");;

  // supera images
  larcv::EventImage2D* ev_wholeimg  = (larcv::EventImage2D*) dataco_whole.get_larcv_data("image2d","wire");
  //const std::vector<larcv::Image2D>& whole_v = ev_wholeimg->image2d_array();

  //chstatus
  const larcv::EventChStatus& ev_chstatus = *(larcv::EventChStatus*) dataco_whole.get_larcv_data("chstatus","wire");

  larcv::Image2D trusted(ev_wholeimg->as_vector()[2].meta());
  larcv::Image2D infill_whole(ev_wholeimg->as_vector()[2].meta());

  int nentries = dataco_crop.get_nentries( "larcv" );
  for (int ientry=entry; ientry<nentries; ientry++) {

    dataco_crop.goto_entry(ientry,"larcv");
    int runid    = dataco_crop.run();
    int subrunid = dataco_crop.subrun();
    int eventid  = dataco_crop.event();
    // load up the whole-view images from the supera file
    dataco_whole.goto_event( runid, subrunid, eventid, "larcv" );
    if(runid!=run || subrunid!=subrun || eventid!=event) break;
    matching_algo.stitchInfill(ev_infill->as_vector()[2],trusted,infill_whole,ev_chstatus);
  }
  return infill_whole;
}

int main( int nargs, char** argv ) {

  gStyle->SetOptStat(0);
  TApplication app ("app",&nargs,argv);
 
  std::cout << "larflow post-processor dev" << std::endl;

  // use hard-coded test-paths for now
  /// whole view examplex
  // std::string input_larflow_file = "../testdata/larflow_test_8541376_98.root";
  // std::string input_reco2d_file  = "../testdata/larlite_reco2d_8541376_98.root";
  // cropped examples
  // -----------------
  // common source files
  std::string input_supera_file       = "../testdata/larcv_5482426_95.root";  
  std::string input_reco2d_file       = "../testdata/larlite_reco2d_5482426_95.root";
  std::string input_opreco_file       = "../testdata/larlite_opreco_5482426_95.root";
  std::string input_mcinfo_file       = "../testdata/larlite_mcinfo_5482426_95.root";
  std::string input_dlcosmictag_file  = "../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root";
  
  // extbnb (mcc9) example
  // std::string input_supera_file       = "../testdata/larcv1_data/larcv2_wholeview_bnbext_mcc9.root";
  // std::string input_reco2d_file       = "../testdata/larcv1_data/larlite_reco2d_a5b67944-607c-49df-9e7d-8881f88bcc0c.root";
  // std::string input_opreco_file       = "";
  // std::string input_mcinfo_file       = "";
  // std::string input_dlcosmictag_file  = "../testdata/larcv1_data/larcv2_larflow_bnbext_mcc9.root";
  
  std::string output_larlite_file     = "output_flowmatch_larlite.root";
  
  bool kVISUALIZE = false;
  bool kINSPECT   = false;
  bool use_hits   = false;
  bool use_truth  = true;
  bool has_opreco = true;
  bool has_mcreco = true;
  bool has_infill = true;
  bool has_ssnet  = true;
  bool makehits_useunmatched = false;
  bool makehits_require_3dconsistency = false;
  int process_num_events = 1;

  if (use_truth && use_hits)
    output_larlite_file = "output_truthmatch_larlite.root";
  else if (!use_truth && use_hits) 
    output_larlite_file = "output_flowmatch_larlite.root";
  else if (use_truth && !use_hits)
    output_larlite_file = "output_truthpixmatch_larlite.root";
  else if (!use_truth && !use_hits)
    output_larlite_file = "output_pixmatch_larlite.root";

  using flowdir = larflow::FlowContourMatch;

  // data from larflow output: sequence of cropped images
  larlitecv::DataCoordinator dataco;
  // 150 event sample
  // -----------------
  // dataco.add_inputfile( input_larflow_y2u_file, "larcv" );
  // dataco.add_inputfile( input_larflow_y2v_file, "larcv" );
  dataco.add_inputfile( input_dlcosmictag_file, "larcv" );
  dataco.initialize();

  // data from whole-view image
  larlitecv::DataCoordinator dataco_whole;
  dataco_whole.add_inputfile( input_supera_file, "larcv" );
  dataco_whole.initialize();
  
  // hit (and mctruth) event data
  larlitecv::DataCoordinator dataco_hits;
  if ( !input_reco2d_file.empty() ) 
    dataco_hits.add_inputfile( input_reco2d_file,  "larlite" );
  if ( !input_opreco_file.empty() ) {
    dataco_hits.add_inputfile( input_opreco_file,  "larlite" );
    has_opreco = true;
  }
  else
    has_opreco = false;
  if ( !input_mcinfo_file.empty() ) {
    dataco_hits.add_inputfile( input_mcinfo_file,  "larlite" );
    has_mcreco = true;
  }
  else
    has_mcreco = false;
  dataco_hits.initialize();

  // output: 3D track hits
  larlite::storage_manager dataco_output( larlite::storage_manager::kWRITE );
  dataco_output.set_out_filename( output_larlite_file );
  dataco_output.open();
  
  // cluster algo
  larlitecv::ContourCluster cluster_algo;
  larflow::FlowContourMatch matching_algo;
  larlite::event_hit pixhits_v;
    
  int nentries = dataco.get_nentries( "larcv" );
  std::cout << "Number of entries in cropped file: " << nentries << std::endl;
  
  int current_runid    = -1;
  int current_subrunid = -1;
  int current_eventid  = -1;
  int nevents = 0;

  //whole image infill
  larcv::Image2D infill_whole_y;

  for (int ientry=0; ientry<nentries; ientry++) {

    dataco.goto_entry(ientry,"larcv");
    
    int runid    = dataco.run();
    int subrunid = dataco.subrun();
    int eventid  = dataco.event();
    std::cout << "Loading entry: " << ientry << " (rse)=(" << runid << "," << subrunid << "," << eventid << ")" << std::endl;
    if ( ientry==0 ) {
      // first entry, set the current_runid
      current_runid    = runid;
      current_subrunid = subrunid;
      current_eventid  = eventid;
    }

    if ( current_runid!=runid || current_subrunid!=subrunid || current_eventid!=eventid ) {

      // if we are breaking, we cut out now, using the event_changeout all at end of file
      std::cout << "new event: (" << runid << "," << subrunid << "," << eventid << ")" << std::endl;
      nevents++;      
      if ( nevents>=process_num_events )
	break;

      event_changeout( dataco_output, dataco_whole, dataco_hits, matching_algo,
		       current_runid, current_subrunid, current_eventid,
		       makehits_useunmatched, makehits_require_3dconsistency, has_opreco, has_mcreco );

      // clear the algo
      matching_algo.clear();
      pixhits_v.clear();
      
      // set the current rse
      current_runid    = runid;
      current_subrunid = subrunid;
      current_eventid  = eventid;

      //clear whole infill
      //infill_whole_y.paint(0.0);

      std::cout << "Event turn over. [enter] to continue." << std::endl;
      std::cin.get();

    }
    // get stitched infill ->should create image once per event
    if( pixhits_v.size()==0 && has_infill ) infill_whole_y = stitch_infill(dataco,dataco_whole,matching_algo,current_runid,current_subrunid,current_eventid,ientry);
    
    // sync up larlite data
    dataco_hits.goto_event( runid, subrunid, eventid, "larlite" );

    // load up the whole-view images from the supera file
    dataco_whole.goto_event( runid, subrunid, eventid, "larcv" );
  
    // larflow input data (assumed to be cropped subimages)
    larcv::EventImage2D* ev_wire      = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "adc");
    larcv::EventImage2D* ev_flow[larflow::FlowContourMatch::kNumFlowDirs] = {NULL};
    ev_flow[flowdir::kY2U] = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "larflow_y2u");
    ev_flow[flowdir::kY2V] = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "larflow_y2v");
    const std::vector<larcv::Image2D>& wire_v = ev_wire->image2d_array();    
    bool hasFlow[2] = { false, false };
    for (int i=0; i<2; i++)
      hasFlow[i] = ( ev_flow[i]->valid() ) ? true : false;

    // hack for debug
    //hasFlow[0] = false;
    //hasFlow[1] = false;
    
    // endpt+segment info
    larcv::EventImage2D* ev_trackimg  = nullptr;
    larcv::EventImage2D* ev_showerimg = nullptr;
    larcv::EventImage2D* ev_endptimg  = nullptr;
    if ( has_ssnet ) {
      ev_trackimg  = (larcv::EventImage2D*)  dataco.get_larcv_data("image2d", "ssnetCropped_track");
      ev_showerimg = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "ssnetCropped_shower");
      ev_endptimg  = (larcv::EventImage2D*)  dataco.get_larcv_data("image2d", "ssnetCropped_endpt");
    }
    
    //infill prediction (unmasked)
    larcv::EventImage2D* ev_infill = nullptr;
    if ( has_infill )
      ev_infill = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "infillCropped");
    
    // For whole-view data, should avoid reloading, repeatedly
    // supera images
    larcv::EventImage2D* ev_wholeimg  = (larcv::EventImage2D*) dataco_whole.get_larcv_data("image2d","wire");
    const std::vector<larcv::Image2D>& whole_v = ev_wholeimg->image2d_array();

    //chstatus
    const larcv::EventChStatus& ev_chstatus = *(larcv::EventChStatus*) dataco_whole.get_larcv_data("chstatus","wire");

    // event data
    const larlite::event_hit*  ev_hit = nullptr;
    if ( !input_reco2d_file.empty() ) {
      ev_hit = ((larlite::event_hit*)dataco_hits.get_larlite_data(larlite::data::kHit, "gaushit"));
      std::cout << "Number of hits: " << ev_hit->size() << std::endl;
    }

    // truth
    larcv::EventImage2D* ev_trueflow  = nullptr;
    const std::vector<larcv::Image2D>* true_v = nullptr;
    std::vector<larcv::Image2D> flow_v;    
    if ( use_truth ) {
      ev_trueflow = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "pixflow");
      true_v = &(ev_trueflow->image2d_array());      
    }
    // merged flow predictions
    if ( hasFlow[flowdir::kY2U] )
      flow_v.emplace_back( std::move(ev_flow[flowdir::kY2U]->modimgat(0) ) );
    else
      flow_v.push_back( larcv::Image2D() ); // dummy image
    if ( hasFlow[flowdir::kY2V] )
      flow_v.emplace_back( std::move(ev_flow[flowdir::kY2V]->modimgat(0) ) );
    else
      flow_v.push_back( larcv::Image2D() ); // dummy image
    
    // Set RSE
    runid    = dataco.run();
    subrunid = dataco.subrun();
    eventid  = dataco.event();
    
    // make badch image (make blanks for now)
    // make blank infill images
    // copy adc images
    std::vector<larcv::Image2D> badch_v;
    std::vector<larcv::Image2D> infill_v;
    std::vector<larcv::Image2D> img_fill_v;
    for ( auto const& img : wire_v ) {
      larcv::Image2D badch( img.meta() );
      badch.paint(0.0);
      badch_v.emplace_back( std::move(badch) );
      larcv::Image2D infill( img.meta() );
      larcv::Image2D img_fill( img.meta() );
      infill.paint(0.0);
      img_fill = img;
      //mask infill and add to adc
      if ( has_infill ) {
	matching_algo.maskInfill(ev_infill->as_vector()[img.meta().id()], ev_chstatus, 0.96, infill );
	matching_algo.addInfill(infill, ev_chstatus, 20.0,  img_fill );
      }
      //fill the vectors
      infill_v.emplace_back( std::move(infill) );
      img_fill_v.emplace_back( std::move(img_fill) );
    }

    // get cluster atomics for cropped u,v,y ADC image    
    cluster_algo.clear();    
    //cluster_algo.analyzeImages( img_v, badch_v, 20.0, 3 );
    cluster_algo.analyzeImages( img_fill_v, badch_v, 20.0, 3 );

    //mask and add whole image infill
    larcv::Image2D whole_y_cp = whole_v[2];
    std::cout <<whole_y_cp.meta().dump() << std::endl;
    if ( has_infill ) {
      matching_algo.maskInfill(infill_whole_y, ev_chstatus, 0.96, infill_whole_y );
      matching_algo.addInfill(infill_whole_y, ev_chstatus, 10.0, whole_y_cp );
    }

    if ( use_hits ) {      
      if ( !use_truth ) {
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, flow_v, *ev_hit, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
      }
      else
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, *true_v, *ev_hit, 10.0, true, true );
    }
    else {
      // make hits from whole image
      if ( pixhits_v.size()==0 )
	//matching_algo.makeHitsFromWholeImagePixels( whole_v[2], pixhits_v, 10.0 );
	matching_algo.makeHitsFromWholeImagePixels( whole_y_cp, pixhits_v, 10.0 );
      
      if ( !use_truth ) {
	//matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, flow_v, pixhits_v, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
	matching_algo.fillPlaneHitFlow(  cluster_algo, img_fill_v[2], img_fill_v, flow_v, pixhits_v, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
      }
      else
	//matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, true_v, pixhits_v, 10.0, true, true );
	matching_algo.fillPlaneHitFlow(  cluster_algo, img_fill_v[2], img_fill_v, *true_v, pixhits_v, 10.0, true, true );
    }

    // update ssnet info
    if ( has_ssnet && ev_trackimg->valid() && ev_showerimg->valid() && ev_endptimg->valid() ) {
      matching_algo.integrateSSNetEndpointOutput( ev_trackimg->as_vector(), ev_showerimg->as_vector(), ev_endptimg->as_vector() );
    }
    else {
      std::cout << "Shower/track/endpt info not valid" << std::endl;
    }
    // update infill labels
    if ( has_infill && ev_infill->valid() && infill_v.size()==3 ) {
      matching_algo.labelInfillHits( infill_v );
      std::cout << "infill labels here" << std::endl;
    }
    else {
      std::cout << "Infill info not valid" << std::endl;
    }

    if ( kVISUALIZE ) {

      std::vector< larlite::larflow3dhit > hits3d_v = matching_algo.get3Dhits_2pl();
      //std::vector< larlite::larflow3dhit > hits3d_v = matching_algo.get3Dhits_1pl( flowdir::kY2U );
      
      // TCanvas c("c","scorematrix",1600,1200);
      // matching_algo.plotScoreMatrix().Draw("colz");
      // c.Update();
      // c.SaveAs("score_matrix.png");
      
      // flip through matches
      TCanvas c("c","matched clusters",1200,400);
      TH2D hsrc = larcv::as_th2d( wire_v[2], "hsource_y" );
      hsrc.SetTitle("Source: Y-plane;wires;ticks");
      hsrc.SetMinimum(10.0);
      hsrc.SetMaximum(255.0);    
      TH2D htar_u = larcv::as_th2d( wire_v[0], "htarget_u" );
      htar_u.SetTitle("Target u-plane;wires;ticks");
      htar_u.SetMinimum(10.0);
      htar_u.SetMaximum(255.0);    
      TH2D htar_v = larcv::as_th2d( wire_v[1], "htarget_v" );
      htar_v.SetTitle("Target v-plane;wires;ticks");
      htar_v.SetMinimum(10.0);
      htar_v.SetMaximum(255.0);    
      
      const larcv::ImageMeta& src_meta = wire_v[2].meta();
      const larcv::ImageMeta* tar_meta[2] = { &(wire_v[0].meta()), &(wire_v[1].meta()) };
      
      std::cout << "Source Matches: " << std::endl;
      for (int i=0; i<matching_algo.m_src_ncontours; i++) {

	// we make a list of tgraphs for each contour
	// thickness determined by score
	// >0.8 = 5 (red)
	// >0.6 = 4 (orange)
	// >0.4 = 3 (yellow)
	// >0.2 = 2 (blue)
	// <0.1 = 1 (black)
	
	const larlitecv::Contour_t& src_ctr = cluster_algo.m_plane_atomics_v[2][ i ];
	TGraph src_graph( src_ctr.size()+1 );
	for ( int n=0; n<src_ctr.size(); n++) {
	  float col = src_meta.pos_x( src_ctr[n].x );
	  float row = src_meta.pos_y( src_ctr[n].y );
	  src_graph.SetPoint(n,col,row);
	}
	src_graph.SetPoint(src_ctr.size(), src_meta.pos_x(src_ctr[0].x), src_meta.pos_y(src_ctr[0].y));
	src_graph.SetLineWidth(5);
	src_graph.SetLineColor(kRed);
	
	// graph the predicted and true flow locations
	TGraph src_flowpix[2];
	TGraph src_truthflow[2];
	int nflowpoints = 0;
	for (int iflow=0; iflow<2; iflow++) {
	  auto it_src_targets = matching_algo.m_src_targets[iflow].find( i );
	  if ( it_src_targets!=matching_algo.m_src_targets[iflow].end() ) {
	    larflow::FlowContourMatch::ContourTargets_t& targetlist = it_src_targets->second;
	    nflowpoints = targetlist.size();
	    src_flowpix[iflow].Set( targetlist.size() );
	    if ( use_truth )
	      src_truthflow[iflow].Set( targetlist.size() );
	    int ipixt = 0;
	    for ( auto& pix_t : targetlist ) {
	      src_flowpix[iflow].SetPoint(ipixt, tar_meta[iflow]->pos_x( pix_t.col ), tar_meta[iflow]->pos_y(pix_t.row) );
	      if ( use_truth ) {
		int truthflow = -10000;
		truthflow = (*true_v)[iflow].pixel( pix_t.row, pix_t.srccol );	  
		try {
		  //std::cout << "truth flow @ (" << pix_t.col << "," << pix_t.row << "): " << truthflow << std::endl;
		  src_truthflow[iflow].SetPoint(ipixt, tar_meta[iflow]->pos_x( pix_t.srccol+truthflow ), tar_meta[iflow]->pos_y(pix_t.row) );
		}
		catch (...) {
		  std::cout << "bad flow @ (" << pix_t.srccol << "," << pix_t.row << ") "
			    << "== " << truthflow << " ==>> (" << pix_t.srccol+truthflow << "," << pix_t.row << ")" << std::endl;
		}
	      }
	      ipixt++;
	    }//end of loop over src_targets
	  }//if src targets
	}//end of loop over flowdir
	std::cout << "SourceIDX[" << i << "]  ";
	// graph the target contours
	std::vector< TGraph > tar_graphs[2];
	for (int iflow=0; iflow<2; iflow++) {
	  for (int j=0; j<matching_algo.m_tar_ncontours[iflow]; j++) {
	    float score = matching_algo.m_score_matrix[ i*matching_algo.m_tar_ncontours[iflow] + j ];
	    if ( score<0.01 )
	      continue;
	    
	    std::cout << "[" << j << "]=" << score << " ";
	    
	    float width = 1;
	    int color = 0;
	    if ( score>0.8 ) {
	      width = 5;
	      color = kRed;
	    }
	    else if ( score>0.6 ) {
	      width = 4;
	      color = kRed-9;
	    }
	    else if ( score>0.3 ) {
	      width = 3;
	      color = kOrange+1;
	    }
	    else if ( score>0.1 ) {
	      width = 2;
	      color = kOrange-9;
	    }
	    else {
	      width = 1;
	      color = kBlack;
	    }
	    
	    const larlitecv::Contour_t& tar_ctr = cluster_algo.m_plane_atomics_v[iflow][ j ];
	    TGraph tar_graph( tar_ctr.size()+1 );
	    for ( int n=0; n<tar_ctr.size(); n++) {
	      float col = tar_meta[iflow]->pos_x( tar_ctr[n].x );
	      float row = tar_meta[iflow]->pos_y( tar_ctr[n].y );
	      tar_graph.SetPoint(n,col,row);
	    }
	    tar_graph.SetPoint(tar_ctr.size(), tar_meta[iflow]->pos_x(tar_ctr[iflow].x),tar_meta[iflow]->pos_y(tar_ctr[0].y));	
	    tar_graph.SetLineWidth(width);
	    tar_graph.SetLineColor(color);
	    
	    tar_graphs[iflow].emplace_back( std::move(tar_graph) );
	  }
	}
	std::cout << std::endl;
	

	// plot matched hits
	TGraph gsrchits( hits3d_v.size() );
	for ( int ihit=0; ihit<(int)hits3d_v.size(); ihit++ ) {
	  larlite::larflow3dhit& hit3d = hits3d_v[ihit];
	  float x = hit3d.srcwire;
	  float y = hit3d.tick;
	  //std::cout << "src hit[" << ihit << "] (r,c)=(" << y << "," << x << ")" << std::endl;
	  gsrchits.SetPoint( ihit, x, y );
	}
	gsrchits.SetMarkerSize(1);
	gsrchits.SetMarkerStyle(21);
	gsrchits.SetMarkerColor(kBlack);

	// plot matched hits
	TGraph gtarhits( hits3d_v.size() );
	for ( int ihit=0; ihit<(int)hits3d_v.size(); ihit++ ) {
	  larlite::larflow3dhit& hit3d = hits3d_v[ihit];

	  // bool goody2u = true;
	  // bool goody2v = true;
	  // if ( hit3d.targetwire[0]<0 || hit3d.targetwire[0]>=2399 )
	  //   goody2u = false;
	  // if ( hit3d.targetwire[1]<0 || hit3d.targetwire[1]>=2399 )
	  //   goody2v = false;
	  // if ( !goody2u && !goody2v )
	  //   continue;
	  // int useflow = (goody2u) ? 0 : 1;
	  	  
	  gtarhits.SetPoint( ihit, hit3d.targetwire[0], hit3d.tick );
	}
	gtarhits.SetMarkerSize(1);
	gtarhits.SetMarkerStyle(21);
	gtarhits.SetMarkerColor(kBlack);

	// SETUP CANVAS
	// -------------
	c.Clear();
	c.Divide(3,1);
	
	// source
	c.cd(1);
	hsrc.Draw("colz");
	//gsrchits.Draw("P");      	
	src_graph.Draw("L");

	
	// target (u)
	c.cd(2);
	htar_u.Draw("colz");
	for ( auto& g : tar_graphs[0] ) {
	  g.Draw("L");
	}
	//gtarhits.Draw("P");
	if ( use_truth ) {
	  src_truthflow[0].SetMarkerStyle(25);      
	  src_truthflow[0].SetMarkerSize(0.2);
	  src_truthflow[0].SetMarkerColor(kMagenta);
	  src_truthflow[0].Draw("P");
	}
	src_flowpix[0].SetMarkerStyle(25);      
	src_flowpix[0].SetMarkerSize(0.2);
	src_flowpix[0].SetMarkerColor(kCyan);            
	src_flowpix[0].Draw("P");

	// target (v)
	c.cd(3);
	htar_v.Draw("colz");
	for ( auto& g : tar_graphs[1] ) {
	  g.Draw("L");
	}
	//gtarhits.Draw("P");
	if ( use_truth ) {
	  src_truthflow[1].SetMarkerStyle(25);      
	  src_truthflow[1].SetMarkerSize(0.2);
	  src_truthflow[1].SetMarkerColor(kMagenta);
	  src_truthflow[1].Draw("P");
	}
	src_flowpix[1].SetMarkerStyle(25);      
	src_flowpix[1].SetMarkerSize(0.2);
	src_flowpix[1].SetMarkerColor(kCyan);            
	src_flowpix[1].Draw("P");
	
	
	c.Update();
	c.Draw();
	if ( kINSPECT )	 {
	  std::cout << "[ENTER] for next contour." << std::endl;
	  std::cin.get();
	}
	else {
	  if ( nflowpoints>20 ) {
	    char pngname[100];
	    sprintf( pngname, "dumpdev/flow_entry%d_cluster%d.png",ientry,i);
	    c.SaveAs( pngname );
	  }
	}
      }
    }//end of visualize

    //break;
    //std::cout << "[ENTER] for next entry." << std::endl;
    //std::cin.get();

  }//end of entry loop

  // save the data from the last event
  event_changeout( dataco_output, dataco_whole, dataco_hits, matching_algo,
		   current_runid, current_subrunid, current_eventid,
		   makehits_useunmatched, makehits_require_3dconsistency, has_opreco, has_mcreco );
  
  std::cout << "Finalize output." << std::endl;
  dataco_output.close();
  
  return 0;

}
