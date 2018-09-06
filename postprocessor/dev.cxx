#include <iostream>
#include <string>

// ROOT
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include <TApplication.h>

// larlite
#include "DataFormat/hit.h"
#include "DataFormat/spacepoint.h"
#include "DataFormat/larflow3dhit.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
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
		      larflow::FlowContourMatch& matching_algo,
		      const int runid,
		      const int subrunid,
		      const int eventid ) {

  std::cout << "event_changeout." << std::endl;
  // save the larflow3dhit vector
  larlite::event_larflow3dhit* ev_larflowhit = (larlite::event_larflow3dhit*)dataco_output.get_data(larlite::data::kLArFlow3DHit,"flowhits");

  // larlite geometry tool: now called in FlowContour
  //const larutil::Geometry* geo = larutil::Geometry::GetME();
  //const float cm_per_tick      = larutil::LArProperties::GetME()->DriftVelocity()*0.5; // cm/usec * usec/tick
  
  // get the final hits made from flow
  std::vector< larlite::larflow3dhit > whole_event_hits3d_v = matching_algo.get3Dhits_2pl( true, false );

  std::cout << "Number of 3D (2-flow) hits: " << whole_event_hits3d_v.size() << std::endl;
  for ( auto& flowhit : whole_event_hits3d_v ) {
    // form spacepoints
    ev_larflowhit->emplace_back( std::move(flowhit) );
  }
  
  dataco_output.set_id( runid, subrunid, eventid );
  dataco_output.next_event();

  return;
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
  // small 5 event sample for DL output:
  std::string input_dlcosmictag_file  = "../testdata/smallsample/larcv_dlcosmictag_5482426_95_smallsample082918.root";
  // larger 150 event test sample for DL output
  // std::string input_larflow_y2u_file  = "../testdata/larcv_larflow_y2u_5482426_95_testsample082918.root";
  // std::string input_larflow_y2v_file  = "../testdata/larcv_larflow_y2v_5482426_95_testsample082918.root";
  // std::string input_infill_file       = "../testdata/larcv_infill_5482426_95_testsample082918.root";
  // std::string input_endptssnet_file   = "../testdata/larcv_endptssnet_5482426_95_testsample082918.root";  
  
  std::string output_larlite_file     = "output_flowmatch_larlite.root";
  
  bool kVISUALIZE = false;
  bool use_hits   = true;
  bool use_truth  = false;
  int process_num_events = 1;

  if (use_truth)
    output_larlite_file = "output_truthmatch_larlite.root";

  // I'm lazy
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
  dataco_hits.add_inputfile( input_reco2d_file,  "larlite" );
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

      event_changeout( dataco_output, matching_algo, current_runid, current_subrunid, current_eventid );

      // clear the algo
      matching_algo.clear();
      pixhits_v.clear();
      
      // set the current rse
      current_runid    = runid;
      current_subrunid = subrunid;
      current_eventid  = eventid;


      std::cout << "Event turn over. [enter] to continue." << std::endl;
      std::cin.get();

    }
    
    // sync up larlite data
    dataco_hits.goto_event( runid, subrunid, eventid, "larlite" );

    // load up the whole-view images from the supera file
    dataco_whole.goto_event( runid, subrunid, eventid, "larcv" );
    
  
    // larflow input data (assumed to be cropped subimages)
    larcv::EventImage2D* ev_wire      = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "adc");
    larcv::EventImage2D* ev_flow[larflow::FlowContourMatch::kNumFlowDirs] = {NULL};
    ev_flow[flowdir::kY2U] = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "larflow_y2u");
    ev_flow[flowdir::kY2V] = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "larflow_y2v");
    const std::vector<larcv::Image2D>& img_v = ev_wire->image2d_array();
    bool hasFlow[2] = { false, false };
    for (int i=0; i<2; i++)
      hasFlow[i] = ( ev_flow[i]->valid() ) ? true : false;

    // endpt+segment info
    larcv::EventImage2D* ev_trackimg = (larcv::EventImage2D*)  dataco.get_larcv_data("image2d", "ssnetCropped_track");
    larcv::EventImage2D* ev_showerimg = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "ssnetCropped_shower");
    larcv::EventImage2D* ev_endptimg = (larcv::EventImage2D*)  dataco.get_larcv_data("image2d", "ssnetCropped_endpt");
    

    // For whole-view data, should avoid reloading, repeatedly
    // supera images
    larcv::EventImage2D* ev_wholeimg  = (larcv::EventImage2D*) dataco_whole.get_larcv_data("image2d","wire");
    const std::vector<larcv::Image2D>& whole_v = ev_wholeimg->image2d_array();
    
    // event data
    const larlite::event_hit&  ev_hit = *((larlite::event_hit*)dataco_hits.get_larlite_data(larlite::data::kHit, "gaushit"));
    std::cout << "Number of hits: " << ev_hit.size() << std::endl;

    // truth
    larcv::EventImage2D* ev_trueflow  = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "pixflow");
    const std::vector<larcv::Image2D>& wire_v = ev_wire->image2d_array();
    const std::vector<larcv::Image2D>& true_v = ev_trueflow->image2d_array();

    // merged flow predictions
    std::vector<larcv::Image2D> flow_v;
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
    std::vector<larcv::Image2D> badch_v;
    for ( auto const& img : img_v ) {
      larcv::Image2D badch( img.meta() );
      badch.paint(0.0);
      badch_v.emplace_back( std::move(badch) );
    }

    // get cluster atomics for cropped u,v,y ADC image    
    cluster_algo.clear();    
    cluster_algo.analyzeImages( img_v, badch_v, 20.0, 3 );
    
    if ( use_hits ) {      
      if ( !use_truth )
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, flow_v, ev_hit, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
      else
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, true_v, ev_hit, 10.0, true, true );
    }
    else {
      // make hits from whole image
      if ( pixhits_v.size()==0 )
	matching_algo.makeHitsFromWholeImagePixels( whole_v[2], pixhits_v, 10.0 );
      
      if ( !use_truth )
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, flow_v, pixhits_v, 10.0, hasFlow[flowdir::kY2U], hasFlow[flowdir::kY2V] );
      else
	matching_algo.fillPlaneHitFlow(  cluster_algo, wire_v[2], wire_v, true_v, pixhits_v, 10.0, true, true );
    }

    // update ssnet info
    if ( ev_trackimg->valid() && ev_showerimg->valid() && ev_endptimg->valid() ) {
      matching_algo.integrateSSNetEndpointOutput( ev_trackimg->as_vector(), ev_showerimg->as_vector(), ev_endptimg->as_vector() );
    }
    else {
      std::cout << "Shower/track/endpt info not valid" << std::endl;
    }

    if ( kVISUALIZE ) {
    
      std::vector< larlite::larflow3dhit > hits3d_v = matching_algo.get3Dhits_2pl();
      //std::vector< larlite::larflow3dhit > hits3d_v = matching_algo.get3Dhits_1pl( flowdir::kY2U );
      
      // TCanvas c("c","scorematrix",1600,1200);
      // matching_algo.plotScoreMatrix().Draw("colz");
      // c.Update();
      // c.SaveAs("score_matrix.png");
      
      // flip through matches
      TCanvas c("c","matched clusters",1600,800);
      TH2D hsrc = larcv::as_th2d( wire_v[2], "hsource_y" );
      hsrc.SetTitle("Source: Y-plane;wires;ticks");
      hsrc.SetMinimum(10.0);
      hsrc.SetMaximum(255.0);    
      TH2D htar = larcv::as_th2d( wire_v[0], "htarget_u" );
      htar.SetTitle("Target: U-plane;wires;ticks");
      htar.SetMinimum(10.0);
      htar.SetMaximum(255.0);    
      htar.Draw("colz");
      
      const larcv::ImageMeta& src_meta = wire_v[2].meta();
      const larcv::ImageMeta& tar_meta = wire_v[0].meta();    
      
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
	TGraph src_flowpix;
	TGraph src_truthflow;
	auto it_src_targets = matching_algo.m_src_targets.find( i );
	if ( it_src_targets!=matching_algo.m_src_targets.end() ) {
	  larflow::FlowContourMatch::ContourTargets_t& targetlist = it_src_targets->second;
	  src_flowpix.Set( targetlist.size() );
	  src_truthflow.Set( targetlist.size() );
	  int ipixt = 0;
	  for ( auto& pix_t : targetlist ) {
	    src_flowpix.SetPoint(ipixt, tar_meta.pos_x( pix_t.col ), tar_meta.pos_y(pix_t.row) );
	    int truthflow = -10000;
	    truthflow = true_v[0].pixel( pix_t.row, pix_t.srccol );	  
	    try {
	      //std::cout << "truth flow @ (" << pix_t.col << "," << pix_t.row << "): " << truthflow << std::endl;
	      src_truthflow.SetPoint(ipixt, tar_meta.pos_x( pix_t.srccol+truthflow ), tar_meta.pos_y(pix_t.row) );
	    }
	    catch (...) {
	      std::cout << "bad flow @ (" << pix_t.srccol << "," << pix_t.row << ") "
			<< "== " << truthflow << " ==>> (" << pix_t.srccol+truthflow << "," << pix_t.row << ")" << std::endl;
	    }
	    ipixt++;
	  }
	}
	
	std::cout << "SourceIDX[" << i << "]  ";
	// graph the target contours
	std::vector< TGraph > tar_graphs;
	for (int j=0; j<matching_algo.m_tar_ncontours; j++) {
	  float score = matching_algo.m_score_matrix[ i*matching_algo.m_tar_ncontours + j ];
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
	  
	  const larlitecv::Contour_t& tar_ctr = cluster_algo.m_plane_atomics_v[0][ j ];
	  TGraph tar_graph( tar_ctr.size()+1 );
	  for ( int n=0; n<tar_ctr.size(); n++) {
	    float col = tar_meta.pos_x( tar_ctr[n].x );
	    float row = tar_meta.pos_y( tar_ctr[n].y );
	    tar_graph.SetPoint(n,col,row);
	  }
	  tar_graph.SetPoint(tar_ctr.size(), tar_meta.pos_x(tar_ctr[0].x),tar_meta.pos_y(tar_ctr[0].y));	
	  tar_graph.SetLineWidth(width);
	  tar_graph.SetLineColor(color);
	  
	  tar_graphs.emplace_back( std::move(tar_graph) );
	}
	std::cout << std::endl;
      
	c.Clear();
	c.Divide(2,1);
	
	// source
	c.cd(1);
	hsrc.Draw("colz");
	src_graph.Draw("L");

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
	gsrchits.Draw("P");      
	
	// target
	c.cd(2);
	htar.Draw("colz");
	for ( auto& g : tar_graphs ) {
	  g.Draw("L");
	}
	src_truthflow.SetMarkerStyle(25);      
	src_truthflow.SetMarkerSize(0.2);
	src_truthflow.SetMarkerColor(kMagenta);
	src_truthflow.Draw("P");      
	src_flowpix.SetMarkerStyle(25);      
	src_flowpix.SetMarkerSize(0.2);
	src_flowpix.SetMarkerColor(kCyan);            
	src_flowpix.Draw("P");
	
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
	gtarhits.Draw("P");
	
	c.Update();
	c.Draw();
	std::cout << "[ENTER] for next contour." << std::endl;
	std::cin.get();
      }
    }//end of visualize

    //break;
    //std::cout << "[ENTER] for next entry." << std::endl;
    //std::cin.get();

  }//end of entry loop

  // save the data from the last event
  event_changeout( dataco_output, matching_algo, current_runid, current_subrunid, current_eventid );
  
  std::cout << "Finalize output." << std::endl;
  dataco_output.close();
  
  return 0;

}
