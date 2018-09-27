#include <iostream>
#include <string>
#include <cmath>
// ROOT
#include "TCanvas.h"
#include "TGraph.h"
#include "TStyle.h"
#include <TApplication.h>

// larlite
#include "DataFormat/hit.h"
#include "DataFormat/spacepoint.h"
#include "DataFormat/larflow3dhit.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/TimeService.h"

// larcv
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/app/UBWireTool/UBWireTool.h"
#include "larcv/core/ROOTUtil/ROOTUtils.h"

// larlitecv
#include "Base/DataCoordinator.h"

// ContourTools
#include "ContourTools/ContourCluster.h"

// FlowContourMatching
#include "FlowContourMatching/FlowContourMatch.h"

int main( int nargs, char** argv ) {

  gStyle->SetOptStat(0);
  TApplication app ("app",&nargs,argv);

  std::cout << "test mctrack projection" << std::endl;

  //space charge
  larutil::SpaceChargeMicroBooNE* sce = new larutil::SpaceChargeMicroBooNE;
  //time service
  const ::larutil::TimeService* tsv = ::larutil::TimeService::GetME();
  //lar properties
  float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;

  std::string input_supera_file       = "../testdata/larcv_5482426_95.root";
  std::string input_reco2d_file       = "../testdata/larlite_reco2d_5482426_95.root";
  std::string input_mcinfo_file       = "../testdata/larlite_mcinfo_5482426_95.root";

  std::string output_larcv_file     = "test_mctrack_out.root";

  // data from whole-view image
  larlitecv::DataCoordinator dataco_img;
  dataco_img.add_inputfile( input_supera_file, "larcv" );
  dataco_img.initialize();

  // hit (and mctruth) event data
  larlitecv::DataCoordinator dataco_mc;
  dataco_mc.add_inputfile( input_reco2d_file,  "larlite" );
  dataco_mc.add_inputfile( input_mcinfo_file, "larlite" );
  dataco_mc.initialize();

  // output: 3D track hits
  //larlite::storage_manager dataco_output( larlite::storage_manager::kWRITE );
  larcv::IOManager dataco_output( larcv::IOManager::kWRITE );
  dataco_output.set_out_file( output_larcv_file );
  dataco_output.initialize();

  // cluster algo
  larlitecv::ContourCluster cluster_algo;
  larflow::FlowContourMatch matching_algo;
  larlite::event_hit pixhits_v;
  //larlite::event_mctrack mctrack_v;

  int nentries = dataco_img.get_nentries( "larcv" );
  nentries=1;
  std::cout << "Number of entries in file: " << nentries << std::endl;

  for (int ientry=0; ientry<nentries; ientry++) {

    dataco_img.goto_entry(ientry,"larcv");

    //clear pixhits
    pixhits_v.clear();
    
    // Set RSE
    int runid    = dataco_img.run();
    int subrunid = dataco_img.subrun();
    int eventid  = dataco_img.event();
    
    // sync up larlite data
    //dataco_mc.goto_event( runid, subrunid, eventid, "larlite" );
    dataco_mc.goto_entry( ientry, "larlite" );

    // out
    larcv::EventImage2D* ev_out_mc = (larcv::EventImage2D*)dataco_output.get_data("image2d","mcproj");

    // supera images
    larcv::EventImage2D* ev_wholeimg  = (larcv::EventImage2D*) dataco_img.get_larcv_data("image2d","wire");
    const std::vector<larcv::Image2D>& whole_v = ev_wholeimg->image2d_array();

    //chstatus
    const larcv::EventChStatus& ev_chstatus = *(larcv::EventChStatus*) dataco_img.get_larcv_data("chstatus","wire");

    // event data
    const larlite::event_hit&  ev_hit = *((larlite::event_hit*)dataco_mc.get_larlite_data(larlite::data::kHit, "gaushit"));
    std::cout << "Number of hits: " << ev_hit.size() << std::endl;
    const larlite::event_mctrack&  ev_track = *((larlite::event_mctrack*)dataco_mc.get_larlite_data(larlite::data::kMCTrack, "mcreco"));

    matching_algo.makeHitsFromWholeImagePixels( whole_v[2], pixhits_v, 10.0 );

    // blank track images: trackid, x, y, z, E with Y meta
    std::vector<larcv::Image2D> trackimg_v;
    for(int i=0; i<5; i++){
      larcv::Image2D trackimg(whole_v[2].meta());
      trackimg.paint(-1.0);
      trackimg_v.emplace_back(std::move(trackimg));
    }
    //larlite::mctrack truthtrack = ev_track.at(0);
    for(const auto& truthtrack : ev_track){
      //initialize internal vectors
      std::vector<std::vector<float>> xyz;
      std::vector<unsigned int> trackid;
      std::vector<float> E;
      const larlite::mcstep start = truthtrack.Start();
      bool isStart=true;
      larlite::mcstep prev_step = start;
      for(auto const& step : truthtrack){
	std::vector<float> dr(4); // x,y,z,t
	std::vector<float> pos(3);
	if(isStart){isStart=false; continue;}

	dr[0] = -prev_step.X()+step.X();
	dr[1] = -prev_step.Y()+step.Y();
	dr[2] = -prev_step.Z()+step.Z();
	dr[3] = -prev_step.T()+step.T();
	float dR = sqrt(pow(dr[0],2)+pow(dr[1],2)+pow(dr[2],2));
	// segment theta: aingle in (z,y) plane starting from y axis
	// sin(theta) = dz/dR
	float theta = asin(dr[2]/dR);
	// segment phi: angle in (x,y) starting from x axis
	// sin(phi) = dy/sqrt(dx^2+dy^2) = dy/(dR*cos(theta))
	float phi = asin(dr[1]/(dR*cos(theta)));
	// we divide dR=sqrt(dx^2+dy^2+dz^2) in 0.15cm steps
	const int N = dR/0.15;
	for(int i=0; i<N; i++){
	  pos[0] = prev_step.X()+N*cos(phi)*cos(theta)*0.15; // dx = cos(phi)*cos(theta)*stepsize
	  pos[1] = prev_step.Y()+N*sin(phi)*cos(theta)*0.15; // dy = sin(phi)*cos(theta)*stepsize
	  pos[2] = prev_step.Z()+N*sin(theta)*0.15; // dz = sin(theta)*stepsize
	  // for time use linear approximation
	  float t = prev_step.T() + N*(0.15*::larutil::LArProperties::GetME()->DriftVelocity()*1.0e3); // cm * cm/usec * usec/ns
	  prev_step = step;
	  std::vector<double> pos_offset = sce->GetPosOffsets( pos[0], pos[1], pos[2] );
	  pos[0] = pos[0]-pos_offset[0]+0.7;
	  pos[1] += pos_offset[1];
	  pos[2] += pos_offset[2];
	  //time tick      
	  float tick = tsv->TPCG4Time2Tick(t) + pos[0]/cm_per_tick;
	  pos[0] = (tick + tsv->TriggerOffsetTPC()/0.5)*cm_per_tick; // x in cm
	  xyz.push_back(pos);
	  trackid.push_back(truthtrack.TrackID());
	  E.push_back(step.E());
	}
      }

      //now translate to image row, col
      std::vector<std::vector<int>> imgpath;
      imgpath.reserve(xyz.size());
      for (auto const& pos : xyz ) {
	//std::vector<int> crossing_imgcoords = larcv::UBWireTool::getProjectedImagePixel( pos, whole_v[2].meta(), 3 );
	std::vector<int> crossing_imgcoords = matching_algo.getProjectedPixel( pos, whole_v[2].meta(), 3 );
	imgpath.push_back( crossing_imgcoords );
      }
      int istep = 0;
      // note: pix are image row, col numbers
      for(auto const& pix : imgpath ){
	if(pix[0]==-1 || pix[3]==-1){istep++; continue;}
	double prevE = trackimg_v[4].pixel(pix[0],pix[3]);
	if( prevE>0 && prevE > E.at(istep) ){ istep++; continue;} //fill only highest E deposit
	trackimg_v[0].set_pixel(pix[0],pix[3],(float)trackid.at(istep));// trackid
	trackimg_v[1].set_pixel(pix[0],pix[3],(float)xyz.at(istep)[0]); // x
	trackimg_v[2].set_pixel(pix[0],pix[3],(float)xyz.at(istep)[1]); // y
	trackimg_v[3].set_pixel(pix[0],pix[3],(float)xyz.at(istep)[2]); // z
	trackimg_v[4].set_pixel(pix[0],pix[3],(float)E.at(istep)); // E
	istep++;
      }
      
      
    }    
    // save the data from the last event
    for ( auto& img : trackimg_v ) {
      larcv::Image2D cp = img;
      ev_out_mc->emplace( std::move(cp) );
    }

    dataco_output.set_id( runid, subrunid, eventid );
    dataco_output.save_entry();
    

    std::cout << "Finalize output." << std::endl;
    dataco_output.finalize();

    return 0;
  }


}
    
    





