#include <iostream>
#include <string>

#include "TTree.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "ublarcvapp/MCTools/LArbysMC.h"
#include "larflow/KeyPoints/PrepKeypointData.h"

int main( int nargs, char** argv )
{

  // keypoint truth analysis
  // output:
  //  1) ttree with entries per truth keypoint, storing
  //     - x,y,z real + sce
  //     - distance to maximum keypoint score triplet
  //     - value of maximum keypoint score triplet
  //     - type of keypoint: track/shower/neutrino vertex


  // inputs
  // ------

  // 1) dlmerged file
  // 2) kps larmatch output

  std::string dlmerged_file = argv[1];
  std::string kps_file      = argv[2];

  // outputs
  // 1) ana tfile with ana ttree
  std::string outfilename   = argv[3];
  
  // Load inputs
  larcv::IOManager iocv( larcv::IOManager::kREAD, "larcv", larcv::IOManager::kTickBackward );
  iocv.add_in_file( dlmerged_file );
  iocv.reverse_all_products();
  iocv.initialize();

  larlite::storage_manager ioll( larlite::storage_manager::kREAD );
  ioll.add_in_filename( kps_file );
  ioll.add_in_filename( dlmerged_file );
  ioll.open();

  // load outputs
  TFile* out = new TFile(outfilename.c_str(),"new");

  // define output tree
  TTree* ana = new TTree("kpsana","Keypoint Truth Ana tree");
  float vtx_sce[3];
  float max_score;
  float max_score_dist;
  int is_nu_vtx;
  ana->Branch("vtx_sce",vtx_sce,"vtx_sce[3]/F");
  ana->Branch("max_score",&max_score,"max_score/F");
  ana->Branch("max_score_dist",&max_score_dist,"max_score_dist/F");
  ana->Branch("is_nu_vtx",&is_nu_vtx,"is_nu_vtx/I");

  const float cut_off_dist = 10.0; // cm
  const float cut_off_distsq = cut_off_dist*cut_off_dist;

  // Keypoint Truth Data Maker
  larflow::keypoints::PrepKeypointData kpdata;
  ublarcvapp::mctools::LArbysMC lmc;
  lmc.bindAnaVariables( ana );
  
  int nentries = iocv.get_n_entries();
  for (int ientry=0; ientry<nentries; ientry++) {
    iocv.read_entry(ientry);
    ioll.go_to(ientry);

    lmc.process( ioll );
    
    // larmatch hits
    larlite::event_larflow3dhit* ev_lmhit =
      (larlite::event_larflow3dhit*)ioll.get_data(larlite::data::kLArFlow3DHit,"larmatch");

    // truth keypoints
    kpdata.process( iocv, ioll );

    std::cout << "number of truth keypoints: " << kpdata.getKPdata().size() << std::endl;
    for ( auto const& kpd : kpdata.getKPdata() ) {
      // for each truth keypoint, we save max triplet within certain distance
      max_score_dist = 1.0e9;
      max_score = -1.0;
      float dist = 0.;
      for ( auto const& hit : *ev_lmhit ) {
        dist = 0.;
        for (int i=0; i<3; i++) {
          dist += (hit[i]-kpd.keypt[i])*(hit[i]-kpd.keypt[i]);
          vtx_sce[i] = kpd.keypt[i];
        }
        if ( dist<cut_off_distsq ) {
          if ( hit[13]>max_score ) {
            max_score = hit[13];
            max_score_dist = sqrt(dist);
          }
        }
      }


      // is neutrino vertex?
      float vtx_dist = 0.;
      float true_vtx[3] = { lmc._vtx_sce_x, lmc._vtx_sce_y, lmc._vtx_sce_z };
      for (int i=0; i<3; i++) {
        vtx_dist += (vtx_sce[i]-true_vtx[i])*(vtx_sce[i]-true_vtx[i]);
      }
      vtx_dist = sqrt(vtx_dist);
      std::cout << "true keypoint dist to true-neutrino vtx: " << vtx_dist << std::endl;
      if ( vtx_dist<1.0 ) is_nu_vtx = 1;
      else is_nu_vtx = 0;
      
      // save result
      ana->Fill();
    }
  }//end of event loop

  out->Write();

  return 0;
}
