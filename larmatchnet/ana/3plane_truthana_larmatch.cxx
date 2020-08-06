#include <iostream>
#include <string>
#include <cmath>
#include <map>
#include <utility>
#include <iterator>
#include <algorithm>
#include <vector>
#include "TFile.h"
#include "TH2D.h"


#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/larflow3dhit.h"
#include "larlite/core/DataFormat/mctrack.h"
#include "larlite/core/DataFormat/mcshower.h"
#include "larlite/core/DataFormat/mctruth.h"

//#include "larflow/Reco/geofuncs.h"

#include "larlite/core/LArUtil/TimeService.h"
#include "larlite/core/LArUtil/SpaceChargeMicroBooNE.h"
#include "larlite/core/LArUtil/LArProperties.h"

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/UBWireTool/UBWireTool.h"
#include "ublarcvapp/UBWireTool/WireData.h"


TVector3 convert_point(const larlite::mcstep& mcstep, larutil::SpaceChargeMicroBooNE* sce, const larutil::TimeService* tsv  ){

  TVector3 point;
  float x,y,z,t;
  float tick;

  const float cm_per_tick = ::larutil::LArProperties::GetME()->DriftVelocity()*0.5;
  t = mcstep.T();
  x = mcstep.X();
  y = mcstep.Y();
  z = mcstep.Z();

  std::vector<double> pos_offset = sce->GetPosOffsets( x, y, z );
  x = x - pos_offset[0]+ 0.6;
  y = y + pos_offset[1];
  z = z + pos_offset[2];
  
  tick = tsv->TPCG4Time2Tick(t) + x/cm_per_tick;
  x  = (tick - 3200)*cm_per_tick;

  point.SetXYZ(x,y,z);
  return point;
}

float DistFromLine(TVector3 x0, TVector3 x1, TVector3 x2){

  float num=((x0 - x1).Cross((x0-x2))).Mag();
  float denom = (x2 - x1).Mag();

  num /=denom;
  return num;
}

int main( int nargs, char** argv ) {

  typedef struct hit_t{
    int istruth;

    int tick;
    int U;
    int V;
    int Y;

    std::array<float,3> xyz;
    float score;

    hit_t()
      : istruth(0),
	tick(0),U(0),V(0),Y(0),
	score(0)
    {};
  } hit_t;

  typedef struct pixmeta_t{
    int type; //trk or shwr
    int vid; //index in mctrack/shower vector
    int pdg;
    TVector3 start;
    TVector3 end;
    
    pixmeta_t()
      : type(0),vid(0),pdg(0)
    {};
  } pixmeta_t;


  //SERVICES
  ::larutil::SpaceChargeMicroBooNE* sce = new ::larutil::SpaceChargeMicroBooNE;
  const ::larutil::TimeService* tsv = ::larutil::TimeService::GetME();

  
  std::cout << "larfow truth data" << std::endl;
  if ( nargs==1 ) {
    std::cout << "=== ARGUMENTS ===" << std::endl;
    std::cout <<  "truthana_larmatch [larmatch] [larcv] [mcinfo] [start entry] [num entries]" << std::endl;
    return 0;
  }

  std::string input_larmatch = argv[1];
  std::string input_larcv    = argv[2];
  std::string input_mcinfo   = argv[3];
  std::string adc_name = "wiremc";
  std::string chstatus_name = "wiremc";
  int startentry = atoi(argv[4]);
  int maxentries = atoi(argv[5]);

  larcv::IOManager io( larcv::IOManager::kREAD, "io", larcv::IOManager::kTickBackward );
  io.add_in_file( input_larcv );
  //io.set_out_file( "ana_temp.root" );
  io.reverse_all_products();
  io.initialize();
  
  
  larlite::storage_manager llio( larlite::storage_manager::kREAD );
  llio.add_in_filename( input_larmatch );
  llio.add_in_filename( input_mcinfo );
  llio.open();

  int nentries = llio.get_entries();

  // output
  TFile* outfile = new TFile(Form("cosmnu_sr0001_evt%d-%d.root",startentry,startentry+maxentries-1),"recreate");

  // DEFINE HISTOGRAMS
  const int nhists = 3;
  // strings for histo names
  std::string str1[4] = {"u","v","y","3d"};
  std::string str2[4] = {"dx","dy","dz","3d"};
  std::string str3[3] = {"track","shower","all"};
  
  // score output versus dist
  TH2D* hprob_v_dist[ nhists ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hprob_v_dist_%s", str3[n].c_str() );
    hprob_v_dist[n] = new TH2D( name,  ";distance from true target wire (cm); match score", 2000, 0, 1000, 100, 0.0, 1.0 );
  }

  // score output versus dist bestmatch
  TH2D* hprob_v_dist_best[ nhists ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hprob_v_dist_bestmatch_%s", str3[n].c_str() );
    hprob_v_dist_best[n] = new TH2D( name,  ";distance from true target wire (cm); match score", 2000, 0, 1000, 100, 0.0, 1.0 );
  }

    //theta all only
  TH1D* htheta[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "htheta_%s", str3[n].c_str() );
    htheta[n] = new TH1D( name,  ";polar angle;", 32, 0, 3.14 );
  }

  //phi all only
  TH1D* hphi[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hphi_%s", str3[n].c_str() );
    hphi[n] = new TH1D( name,  ";azimuth angle;", 32, -3.14, 3.14 );
  }

  // dist vs track/shower theta 
  TH2D* hdist_v_theta[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hdist_v_theta_%s", str3[n].c_str() );
    hdist_v_theta[n] = new TH2D( name,  ";polar angle (rad); distance to true triplet (cm)", 16, 0, 3.14, 600, 0.0, 300 );
  }

  // dist vs track/shower theta best
  TH2D* hdist_v_theta_best[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hdist_v_theta_best_%s", str3[n].c_str() );
    hdist_v_theta_best[n] = new TH2D( name,  ";polar angle (rad); distance to true triplet (cm)", 16, 0, 3.14, 600, 0.0, 300 );
  }

  // dist vs track/shower phi 
  TH2D* hdist_v_phi[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hdist_v_phi_%s", str3[n].c_str() );
    hdist_v_phi[n] = new TH2D( name,  ";azimuth angle (rad); distance to true triplet (cm)", 32, -3.14, 3.14, 600, 0.0, 300 );
  }

  // dist vs track/shower phi best
  TH2D* hdist_v_phi_best[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hdist_v_phi_best_%s", str3[n].c_str() );
    hdist_v_phi_best[n] = new TH2D( name,  ";azimuth angle (rad); distance to true triplet (cm)", 32, -3.14, 3.14, 600, 0.0, 300 );
  }
  
  // dist vs radial dist
  TH2D* hdist_v_radius[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hdist_v_radius_%s", str3[n].c_str() );
    hdist_v_radius[n] = new TH2D( name,  ";distance to true triplet (cm);distance from axis (cm)", 900, 0, 300, 1200, 0, 400 );
  }

  // dist vs radial dist best,atch
  TH2D* hdist_v_radius_best[ 3 ] = {nullptr};
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hdist_v_radius_best_%s", str3[n].c_str() );
    hdist_v_radius_best[n] = new TH2D( name,  ";distance to true triplet (cm);distance from  axis (cm)", 900, 0, 300, 1200, 0, 400 );
  }

  // error in flow 
  TH1D* herrflow[ 3 ] = { nullptr };
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "herrflow_%s", str3[n].c_str() );
    herrflow[n] = new TH1D(name, ";distance to true triplet (cm)", 2000, 0, 1000 );

  }
  // error in flow bestmatch
  TH1D* herrflow_best[ 3 ] = { nullptr };
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "herrflow_best_%s", str3[n].c_str() );
    herrflow_best[n] = new TH1D(name, ";distance to true triplet (cm)", 2000, 0, 1000 );

  }
  // error in flow bestmatch
  TH1D* hrad_best[ 3 ] = { nullptr };
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hrad_best_%s", str3[n].c_str() );
    hrad_best[n] = new TH1D(name, ";distance to axis (cm)", 2000, 0, 1000 );

  }
  // error in flow bestmatch
  TH1D* hrad_best_goodT[ 3 ] = { nullptr };
  for (int n=0; n<3; n++ ) {
    char name[100];
    sprintf( name, "hrad_best_goodT_%s", str3[n].c_str() );
    hrad_best_goodT[n] = new TH1D(name, ";distance to axis (cm)", 2000, 0, 1000 );

  }

    
  // MCPG
  ublarcvapp::mctools::MCPixelPGraph mcpg;
  mcpg.set_adc_treename( adc_name );

  
  // LOOP OVER EVENTS 
  if(startentry+maxentries > nentries) maxentries = nentries - startentry;
  for (int ientry=startentry; ientry<startentry+maxentries; ientry++ ) {

    std::cout << "===========================================" << std::endl;
    std::cout << "[ Entry " << ientry << " ]" << std::endl;

    llio.go_to(ientry);
    io.read_entry(ientry);

    larlite::event_larflow3dhit* lfhit_v = (larlite::event_larflow3dhit*)llio.get_data(larlite::data::kLArFlow3DHit,"larmatch");
    larlite::event_mctrack* evmctrack   = (larlite::event_mctrack*)llio.get_data(larlite::data::kMCTrack,"mcreco");
    larlite::event_mcshower* evmcshower = (larlite::event_mcshower*)llio.get_data(larlite::data::kMCShower,"mcreco");
    
    mcpg.buildgraph( io, llio);
    //mcpg.printGraph();

    std::map<int, hit_t> map1;
    std::multimap<int,int> mmap1;
    std::multimap<std::pair<int,int>,int> mmap2;
    typedef std::multimap<int,int>::iterator Iterator1;
    typedef std::multimap<std::pair<int,int>,int>::iterator Iterator2;
    
    std::multimap<std::pair<int,int>,pixmeta_t> pixm;
    typedef std::multimap<std::pair<int,int>,pixmeta_t>::iterator Iterator3;
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> primaries = mcpg.getPrimaryParticles();
    
    for(int nid=0; nid<primaries.size(); nid++){
      const ublarcvapp::mctools::MCPixelPGraph::Node_t node = *(primaries.at(nid));
      TVector3 start;
      TVector3 end;
      //int tid = node.tid;
      //std::vector<std::vector<int>> pix_vv = mcpg.getPixelsFromParticleAndDaughters(tid);
      larlite::mctrack* trk = NULL;
      larlite::mcshower* shwr = NULL;
      //if(node.origin !=1) continue;  //OPTION OFR NU ONLY
      if(node.type==0) trk = &evmctrack->at( node.vidx );
      else if(node.type==1) shwr = &evmcshower->at( node.vidx );
      else {};

      if(trk){
	start = convert_point(trk->Start(), sce, tsv);
	end = convert_point(trk->End(), sce, tsv);
      }
      else if(shwr){
	start = convert_point(shwr->Start(), sce, tsv);
	end = convert_point(shwr->End(), sce, tsv);
      }
      else{
	start.SetXYZ(15000.,15000.,15000.);
	end.SetXYZ(15000.,15000.,15000.);
      }

      // fill lookup map
      for(int j=0; j<node.pix_vv[2].size()/2; j++){
	pixmeta_t meta;
	meta.type = node.type;
	meta.vid = node.vidx;
	meta.pdg = node.pid;
	meta.start = start;
	meta.end = end;
	
	pixm.insert(std::make_pair(std::make_pair(node.pix_vv[2][2*j],node.pix_vv[2][2*j+1]),meta));
      }
    }
    
    // loop over hits
    for ( size_t ihit=0; ihit< lfhit_v->size(); ihit++ ) {
      const larlite::larflow3dhit& lfhit = lfhit_v->at(ihit);
      
      hit_t hit;
      hit.tick = lfhit.tick;
      hit.U = lfhit.targetwire[0];
      hit.V = lfhit.targetwire[1];
      hit.Y = lfhit.targetwire[2];
      hit.istruth = (int)lfhit.truthflag;
      hit.score = lfhit.track_score;
      for(int x=0; x<3; x ++) hit.xyz[x] = lfhit[x];

      map1.insert(std::make_pair(ihit,hit));
      mmap1.insert(std::make_pair(hit.istruth,ihit));
      mmap2.insert(std::make_pair(std::make_pair(hit.tick,hit.Y),ihit));

    }
    
    //grab true triplets : istruth==1
    std::pair<Iterator1,Iterator1> query1 = mmap1.equal_range(1);
    int nhits_wtrueflow = std::distance(query1.first, query1.second);
    for(Iterator1 it = query1.first; it!= query1.second; it++){
      std::pair<int,int> coord = std::make_pair(map1.at(it->second).tick, map1.at(it->second).Y);
      hit_t truehit = map1.at(it->second);

      // get mcinfo if available
      Iterator3 query3 = pixm.find(coord);
      pixmeta_t trackmeta;
      float theta =-2; 
      float phi =-2;
      if(query3 == pixm.end()) continue;
      trackmeta = query3->second;
      theta = (trackmeta.end - trackmeta.start).Theta();
      phi =  (trackmeta.end - trackmeta.start).Phi();

      //loop to get best score
      float bestscore=0;
      int bestidx=0;
      std::pair<Iterator2,Iterator2> query2 = mmap2.equal_range(coord);
      for(Iterator2 it = query2.first; it!= query2.second; it++){	
	if(map1.at(it->second).score > bestscore){
	  bestscore = map1.at(it->second).score;
	  bestidx = it->second;
	}	

	//fill all scores
	TVector3 me(map1.at(it->second).xyz[0], map1.at(it->second).xyz[1], map1.at(it->second).xyz[2]);
	float dLine = DistFromLine(me, trackmeta.start, trackmeta.end);      
	float dx = sqrt(pow(map1.at(it->second).xyz[0] - truehit.xyz[0],2));
	float dy = sqrt(pow(map1.at(it->second).xyz[1] - truehit.xyz[1],2));
	float dz = sqrt(pow(map1.at(it->second).xyz[2] - truehit.xyz[2],2));
	float dist = sqrt(dx*dx + dy*dy + dz*dz);

	if(trackmeta.type==0){
	  hdist_v_theta[0]->Fill( theta, dist);
	  hdist_v_phi[0]->Fill( phi, dist);
	  
	  hdist_v_radius[0]->Fill( dist, dLine);
	  
	  hphi[0]->Fill(phi);
	  htheta[0]->Fill(theta);
	  
	  herrflow[0]->Fill( dist );
	  hprob_v_dist[0]->Fill( dist, map1.at(it->second).score);
	}
	
	if(trackmeta.type==1){
	  hdist_v_theta[1]->Fill( theta, dist);
	  hdist_v_phi[1]->Fill( phi, dist);
	  
	  hdist_v_radius[1]->Fill( dist, dLine);
	  
	  hphi[1]->Fill(phi);
	  htheta[1]->Fill(theta);
	  
	  herrflow[1]->Fill( dist );
	  hprob_v_dist[1]->Fill( dist, map1.at(it->second).score );	
      }
	hdist_v_theta[2]->Fill( theta, dist);
	hdist_v_phi[2]->Fill( phi, dist);
      
	hdist_v_radius[2]->Fill( dist, dLine);
      
	hphi[2]->Fill(phi);
	htheta[2]->Fill(theta);

	herrflow[2]->Fill( dist );
	hprob_v_dist[2]->Fill( dist, map1.at(it->second).score);

      }

      // fill bestmatch
      float dx = sqrt(pow(map1.at(bestidx).xyz[0] - truehit.xyz[0],2));
      float dy = sqrt(pow(map1.at(bestidx).xyz[1] - truehit.xyz[1],2));
      float dz = sqrt(pow(map1.at(bestidx).xyz[2] - truehit.xyz[2],2));
      float dist = sqrt(dx*dx + dy*dy + dz*dz);
      
      //float dU = map1.at(bestidx).U - truehit.U;
      //float dV = map1.at(bestidx).V - truehit.V;
      
      TVector3 me(map1.at(bestidx).xyz[0], map1.at(bestidx).xyz[1], map1.at(bestidx).xyz[2]);
      float dLine = DistFromLine(me, trackmeta.start, trackmeta.end);      

      if(trackmeta.type==0){
	hdist_v_theta_best[0]->Fill( theta, dist);
	hdist_v_phi_best[0]->Fill( phi, dist);

	hdist_v_radius_best[0]->Fill( dist, dLine);

	herrflow_best[0]->Fill( dist );
	hprob_v_dist_best[0]->Fill( dist, bestscore);
	hrad_best[0]->Fill( dLine );
	if(dist<=1.0) hrad_best_goodT[0]->Fill( dLine );
      }
      
      if(trackmeta.type==1){
	hdist_v_theta_best[1]->Fill( theta, dist);
	hdist_v_phi_best[1]->Fill( phi, dist);

	hdist_v_radius_best[1]->Fill( dist, dLine);

	herrflow_best[1]->Fill( dist );
	hprob_v_dist_best[1]->Fill( dist, bestscore);	
	hrad_best[1]->Fill( dLine );
	if(dist<=1.0) hrad_best_goodT[1]->Fill( dLine );

      }
      
      hdist_v_theta_best[2]->Fill( theta, dist);
      hdist_v_phi_best[2]->Fill( phi, dist);
      
      hdist_v_radius_best[2]->Fill( dist, dLine);
      
      herrflow_best[2]->Fill( dist );
      hprob_v_dist_best[2]->Fill( dist, bestscore);
      hrad_best[2]->Fill( dLine );
      if(dist<=1.0) hrad_best_goodT[2]->Fill( dLine );

    }// end of loop over points
    std::cout << "hits with true flow: " << nhits_wtrueflow << std::endl;
    std::cout << "total hits: " << lfhit_v->size() <<" "<< map1.size() << std::endl;
    
  }//event loop

  outfile->Write();
  outfile->Close();
  
  //io.finalize();
  llio.close();
  
  std::cout << "FIN" << std::endl;
  return 0;
};
