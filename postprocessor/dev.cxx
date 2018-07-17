#include <iostream>
#include <string>

// ROOT
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include <TApplication.h>

// larlite
#include "DataFormat/hit.h"

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
//#include "ContourTools/ContourShapeMeta.h"
#include "ContourTools/ContourCluster.h"

#include "FlowContourMatching/FlowContourMatch.h"


int main( int nargs, char** argv ) {

  gStyle->SetOptStat(0);
  TApplication app ("app",&nargs,argv);
 
  std::cout << "larflow post-processor dev" << std::endl;

  // use hard-coded test-paths for now
  std::string input_larflow_file = "../testdata/larflow_test_8541376_98.root";
  std::string input_reco2d_file  = "../testdata/larlite_reco2d_8541376_98.root";

  larlitecv::DataCoordinator dataco;
  dataco.add_inputfile( input_larflow_file, "larcv" );
  dataco.add_inputfile( input_reco2d_file,  "larlite" );
  dataco.initialize();

  // cluster algo
  larlitecv::ContourCluster cluster_algo;
  larflow::FlowContourMatch matching_algo;
  
  int nentries = dataco.get_nentries( "larcv" );

  for (int ientry=0; ientry<nentries; ientry++) {

    dataco.goto_entry(ientry,"larcv");
  
    // input data
    larcv::EventImage2D* ev_wire      = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "wire");
    larcv::EventImage2D* ev_flow      = (larcv::EventImage2D*) dataco.get_larcv_data("image2d", "larflow_y2u");
    const larlite::event_hit&  ev_hit = *((larlite::event_hit*)dataco.get_larlite_data(larlite::data::kHit, "gaushit"));
  
    const std::vector<larcv::Image2D>& wire_v = ev_wire->image2d_array();
    const std::vector<larcv::Image2D>& flow_v = ev_flow->image2d_array();
    
    // get cluster atomics for u and y ADC image
    const std::vector<larcv::Image2D>& img_v = ev_wire->image2d_array();

    // make badch image (make blanks for now)
    std::vector<larcv::Image2D> badch_v;
    for ( auto const& img : img_v ) {
      larcv::Image2D badch( img.meta() );
      badch.paint(0.0);
      badch_v.emplace_back( std::move(badch) );
    }
    
    cluster_algo.analyzeImages( img_v, badch_v, 20.0, 3 );
    matching_algo.createMatchData( cluster_algo, flow_v[0], wire_v[2], wire_v[0] );
    //matching_algo.dumpMatchData();
    matching_algo.scoreMatches( cluster_algo, 2, 0 );

    // TCanvas c("c","scorematrix",1600,1200);
    // matching_algo.plotScoreMatrix().Draw("colz");
    // c.Update();
    // c.SaveAs("score_matrix.png");

    matching_algo.greedyMatch();

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

      TGraph src_flowpix;
      auto it_src_targets = matching_algo.m_src_targets.find( i );
      if ( it_src_targets!=matching_algo.m_src_targets.end() ) {
	larflow::FlowContourMatch::ContourTargets_t& targetlist = it_src_targets->second;
	src_flowpix.Set( targetlist.size() );
	int ipixt = 0;
	for ( auto& pix_t : targetlist ) {
	  src_flowpix.SetPoint(ipixt, src_meta.pos_x( pix_t.col ), src_meta.pos_y(pix_t.row) );
	}
      }

      std::cout << "SourceIDX[" << i << "]  ";
      
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
	  float col = src_meta.pos_x( tar_ctr[n].x );
	  float row = src_meta.pos_y( tar_ctr[n].y );
	  tar_graph.SetPoint(n,col,row);
	}
	tar_graph.SetPoint(tar_ctr.size(), src_meta.pos_x(tar_ctr[0].x),src_meta.pos_y(tar_ctr[0].y));	
	tar_graph.SetLineWidth(width);
	tar_graph.SetLineColor(color);
	
	tar_graphs.emplace_back( std::move(tar_graph) );
      }
      std::cout << std::endl;
      
      c.Clear();
      c.Divide(2,1);
      c.cd(1);
      hsrc.Draw("colz");
      src_graph.Draw("L");
      c.cd(2);
      htar.Draw("colz");
      for ( auto& g : tar_graphs ) {
	g.Draw("L");
      }
      src_flowpix.SetMarkerStyle(20);      
      src_flowpix.SetMarkerSize(1.0);
      src_flowpix.Draw("P");
      c.Update();
      c.Draw();
      std::cout << "[ENTER] to continue" << std::endl;
      std::cin.get();
    }

    break;
  }
  
  
  
  return 0;

}
