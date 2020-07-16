#include <iostream>
#include <string>

#include "TTree.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventPGraph.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/larflow3dhit.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
#include "ublarcvapp/MCTools/LArbysMC.h"
#include "larflow/KeyPoints/PrepKeypointData.h"
#include "larflow/Reco/NuVertexCandidate.h"

int main( int nargs, char** argv )
{

  // keypoint reco analysis
  // output:
  //  1) ttree per event storing
  //     - mc event info
  //     - number of reco vertices near true neutrino vertex
  //     - number of reco vertices near true track keypoint
  //     - number of reco vertices that are false
  //  2) if wire-cell cosmic tagger info available, ttree per event storing
  //     - number of reco vertices near true neutrino vertex w/ wirecell mask
  //     - number of reco vertices near true track keypoint w/ wirecell mask
  //     - number of reco vertices that are false w/ wirecell mask

  // inputs
  // ------

  // 1) dlmerged file
  // 2) keypoint reco output

  std::string dlmerged_file = argv[1];
  std::string kpreco_file   = argv[2];

  // outputs
  // 1) ana tfile with ana ttree
  std::string outfilename   = argv[3];

  int ismc                  = std::atoi(argv[4]);
  
  // Load inputs
  larcv::IOManager iocv( larcv::IOManager::kREAD, "larcv", larcv::IOManager::kTickBackward );
  iocv.add_in_file( dlmerged_file );
  iocv.reverse_all_products();
  iocv.initialize();

  larlite::storage_manager ioll( larlite::storage_manager::kREAD );
  ioll.add_in_filename( dlmerged_file );
  ioll.set_data_to_read( larlite::data::kMCTruth,  "generator" );
  ioll.set_data_to_read( larlite::data::kMCTrack,  "mcreco" );
  ioll.set_data_to_read( larlite::data::kMCShower, "mcreco" );
  ioll.open();

  TFile kpreco_tfile(kpreco_file.c_str(),"open");
  TTree* kpreco_ttree = (TTree*)kpreco_tfile.Get("KPSRecoManagerTree");
  std::vector<larflow::reco::NuVertexCandidate>* ev_kpreco = 0;
  kpreco_ttree->SetBranchAddress( "nuvetoed_v", &ev_kpreco );

  // load outputs
  TFile* out = new TFile(outfilename.c_str(),"new");

  // define output trees

  // per-event analysis tree
  TTree* ev_ana = new TTree("nuvertexana_event","Nu Vertex Reco Ana Event tree");

  int run,subrun,event;
  
  int n_reco;               //< total number of candidates per event
  int n_near_true_vtx;      //< number near true vertex (within 5 cm)
  int n_false;              //< number of false vertices

  int n_reco_wct;           //< total vertices if we exclude those on the WC tagged pixels
  int n_near_true_vtx_wct;  //< num near true vertex (within 5 cm)
  int n_false_wct;          //< number false after wire cell filter

  int n_reco_dl;            //< number of DL vertices
  int n_near_true_vtx_dl;  //< num dl vertices near truth vertex  
  int n_false_dl;           //< number of false dl vertices

  float truth_vtx_qsum[3] = { 0., 0., 0. };  // truth of charge near the vertex
  float min_dist_to_vtx     = 0.;            // min distance to closest reco vtx
  float min_dist_to_vtx_wct = 0.;            // min distance to closest reco vtx not on tagged pixel
  float min_dist_to_vtx_dl  = 0.;            // min distance to closest reco DL vtx
  
  ev_ana->Branch("run",&run,"run/I");
  ev_ana->Branch("subrun",&subrun,"subrun/I");
  ev_ana->Branch("event",&event,"event/I");  

  ev_ana->Branch("n_reco",&n_reco,"n_reco/I");
  ev_ana->Branch("n_near_true_vtx",&n_near_true_vtx,"n_near_true_vtx/I");
  ev_ana->Branch("n_false",&n_false,"n_false/I");

  ev_ana->Branch("n_reco_wct",&n_reco_wct,"n_reco_wct/I");  
  ev_ana->Branch("n_near_true_vtx_wct",&n_near_true_vtx_wct,"n_near_true_vtx_wct/I");
  ev_ana->Branch("n_false_wct",&n_false_wct,"n_false_wct/I");

  ev_ana->Branch("n_reco_dl",&n_reco_dl,"n_reco_dl/I");  
  ev_ana->Branch("n_near_true_vtx_dl",&n_near_true_vtx_dl,"n_near_true_vtx_dl/I");
  ev_ana->Branch("n_false_dl",&n_false_dl,"n_false_dl/I");
  
  ev_ana->Branch("min_dist_to_vtx",     &min_dist_to_vtx,     "min_dist_to_vtx/F" );
  ev_ana->Branch("min_dist_to_vtx_dl",  &min_dist_to_vtx_dl,  "min_dist_to_vtx_dl/F" );
  ev_ana->Branch("min_dist_to_vtx_wct", &min_dist_to_vtx_wct, "min_dist_to_vtx_wct/F" );  
  ev_ana->Branch("true_vtx_qsum",truth_vtx_qsum, "true_vtx_qsum[3]/F" );

  // per reco vertex analysis tree
  TTree* kp_ana = new TTree("vertexana_recovertex","Ana per reco vertex");
  // int   nshower;
  // int   ntrack;
  // int   nshower_true;  
  // int   ntrack_true;
  float pos[3];
  float score;
  float dist_to_vertex;
  kp_ana->Branch("pos",pos,"pos[3]/F");
  kp_ana->Branch("score",&score,"score/F");
  kp_ana->Branch("dist_to_vertex",&dist_to_vertex,"dist_to_vertex/F");

  
  const float cut_off_dist = 10.0; // cm
  const float cut_off_distsq = cut_off_dist*cut_off_dist;

  // Keypoint Truth Data Maker
  //larflow::keypoints::PrepKeypointData kpdata; //truth key points
  ublarcvapp::mctools::LArbysMC lmc;
  lmc.bindAnaVariables( ev_ana );
  kp_ana->Branch("nlepton_35mev",&lmc._nlepton_35mev,"nlepton_35mev/I");
  kp_ana->Branch("nproton_60mev",&lmc._nproton_60mev,"nproton_60mev/I");
  kp_ana->Branch("nmeson_35mev",&lmc._nmeson_35mev,"nmeson_35mev/I");  
  kp_ana->Branch("nshower",&lmc._nshower,"nshower/I");    
                 
  int nentries = iocv.get_n_entries();
  for (int ientry=0; ientry<nentries; ientry++) {

    // load trees
    iocv.read_entry(ientry);
    ioll.go_to(ientry);
    kpreco_ttree->GetEntry(ientry);

    // Get neutrino interaction truth
    float true_vtx[3] = { 0 };
    if ( ismc ) {
      lmc.process( ioll );

      // true vertex after space charge correction      
      true_vtx[0] = lmc._vtx_sce_x;
      true_vtx[1] = lmc._vtx_sce_y;
      true_vtx[2] = lmc._vtx_sce_z;
    }

    // truth keypoints
    //kpdata.process( iocv, ioll );
    
    // wirecell cosmic pixel image
    larcv::EventImage2D* ev_thrumu =
      (larcv::EventImage2D*)iocv.get_data(larcv::kProductImage2D,"thrumu");
    auto const& thrumu_v = ev_thrumu->Image2DArray();

    // ADC images
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iocv.get_data(larcv::kProductImage2D,"wire");
    auto const& adc_v = ev_adc->Image2DArray();

    // DL Vertex
    larcv::EventPGraph* ev_pgraph =
      (larcv::EventPGraph*)iocv.get_data(larcv::kProductPGraph,"test");
    
    std::cout << "[ENTRY " << ientry << "]" << std::endl;
    //std::cout << "  number of truth keypoints: " << kpdata.getKPdata().size() << std::endl;
    std::cout << "  number of reco keypoints: " << ev_kpreco->size() << std::endl;
    std::cout << "  number of Wirecell images: " << thrumu_v.size() << std::endl;
    std::cout << "  number of DL reco vertices: " << ev_pgraph->PGraphArray().size() << std::endl;
    std::cout << "  true neutrino vertex: (" << true_vtx[0] << "," << true_vtx[1] << "," << true_vtx[2] << ")" << std::endl;
    if ( ismc ) {
      lmc.printInteractionInfo();
      //kpdata.printKeypoints();
    }

    // number of reco vertices
    n_reco = (int)ev_kpreco->size();
    
    // KP Reco loop
    n_near_true_vtx = 0;
    n_false = 0;

    n_reco_wct = 0;
    n_near_true_vtx_wct = 0;
    n_false_wct = 0;

    n_reco_dl = (int)ev_pgraph->PGraphArray().size();
    n_near_true_vtx_dl = 0;
    n_false_dl = 0;

    // characterize vertex
    min_dist_to_vtx     = 1110;
    min_dist_to_vtx_wct = 1110;    
    min_dist_to_vtx_dl  = 1110;

    // get vertex activity
    // sum charge on the three planes
    if ( ismc ) {
      for (int p=0; p<3; p++) truth_vtx_qsum[p] = 0.0;
      if ( lmc._vtx_tick>adc_v[0].meta().min_y() && lmc._vtx_tick<adc_v[0].meta().max_y() ) {
        int row = adc_v[0].meta().row( lmc._vtx_tick );
        
        for (int p=0; p<3; p++ ) {
          
          if ( lmc._vtx_wire[p]>=adc_v[p].meta().min_x() && lmc._vtx_wire[p]<(int)adc_v[p].meta().max_x() ) {
            int col = adc_v[p].meta().col( lmc._vtx_wire[p] );
            
            for (int dr=-3; dr<=3; dr++ ) {
              int r=row+dr;
              if ( r<0 || r>=(int)adc_v[p].meta().rows() ) continue;
              for (int dc=-3; dc<=3; dc++) {
                int c = col+dc;
                if (c<0 || c>=(int)adc_v[p].meta().cols() ) continue;
                if ( adc_v[p].pixel(r,c)>10.0 )
                  truth_vtx_qsum[p] += adc_v[p].pixel(r,c);
              }//end of col loop
            }//end of row loop
          }//end of if valid wire
        }//end of plane loop
      }//end of if valid tick
    }

    if ( ismc ) {
      // get dl vertex info
      for ( auto const& pgraph : ev_pgraph->PGraphArray() ) {
        for ( auto const& roi : pgraph.ParticleArray() ) {
          float dist = 0.;
          dist += ( roi.X()-true_vtx[0] )*( roi.X()-true_vtx[0] );
          dist += ( roi.Y()-true_vtx[1] )*( roi.Y()-true_vtx[1] );
          dist += ( roi.Z()-true_vtx[2] )*( roi.Z()-true_vtx[2] );
          dist = sqrt(dist);
          std::cout << "dl vtx: (" << roi.X() << "," << roi.Y() << "," << roi.Z() << ") dist2vtx=" << dist << " cm" << std::endl;
          if ( min_dist_to_vtx_dl>dist ) {
            min_dist_to_vtx_dl = dist;
          }
          if ( dist<5.0 )
            n_near_true_vtx_dl++;
        }
      }
    }
    
    // analysis for reco keypoints
    int ivtx = 0;
    for ( auto const& kpc : *ev_kpreco ) {

      for (int i=0; i<3; i++)
        pos[i] = kpc.pos[i];

      score = kpc.score;
      
      
      // determine if on Wirecell-tagged cosmic
      // requirement: be on tagged pixel for 2/3 planes
      bool iscosmic = false;

      float tick = 3200 + kpc.pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
      if ( tick<=thrumu_v[0].meta().min_y() || tick>=thrumu_v[0].meta().max_y() ) {
        iscosmic = true;
      }
      else {
        int row = thrumu_v[0].meta().row( tick );        
        int nplanes_on_wctpixel = 0;
        for ( auto const& img : thrumu_v ) {
          // out of image?  we remove it implicitly by adding to counter
          std::vector<double> dpos = { (double)kpc.pos[0],
                                       (double)kpc.pos[1],
                                       (double)kpc.pos[2] };
          int nearestwire = larutil::Geometry::GetME()->NearestWire( dpos, img.meta().plane() );
          int col = img.meta().col((float)nearestwire);

          bool onpixel = false;
          for (int dr=-1; dr<=1; dr++) {
            int r = row+dr;
            if ( r<0 || r>=(int)img.meta().rows() ) continue;
            for (int dc=-1; dc<=1; dc++) {
              int c = col+dc;
              if ( c<0 || c>=(int)img.meta().cols() ) continue;
              if ( img.pixel(r,c)>10.0 ) onpixel = true;
              if ( onpixel ) break;
            }
            if ( onpixel ) break;            
          }

          if ( onpixel ) nplanes_on_wctpixel++;
        }
        if ( nplanes_on_wctpixel>=3 ) iscosmic = true;
      }

      // dist to vtx
      dist_to_vertex = 0;
      if ( ismc ) {
        for (int i=0; i<3; i++) {
          dist_to_vertex += (kpc.pos[i]-true_vtx[i])*(kpc.pos[i]-true_vtx[i]);
        }
        dist_to_vertex = sqrt(dist_to_vertex);

        if ( dist_to_vertex < min_dist_to_vtx )
          min_dist_to_vtx = dist_to_vertex;

        if ( !iscosmic && dist_to_vertex<min_dist_to_vtx_wct ) {
          min_dist_to_vtx_wct = dist_to_vertex;
        }

        if ( dist_to_vertex>5.0 )
          n_false++;
        else
          n_near_true_vtx++;

        if ( !iscosmic ) {
          if ( dist_to_vertex>5.0 )
            n_false_wct++;
          else
            n_near_true_vtx_wct++;
        }
      }
      else {
        dist_to_vertex = 0;
      }
      std::cout << "RecoVtx[" << ivtx << "] score=" << score << " dist-to-vertex=" << dist_to_vertex  << std::endl;
      
      kp_ana->Fill();
      
    }//end of reco vertex loop


    
    // for ( auto const& kpd : kpdata.getKPdata() ) {
    //   // for each truth keypoint, we save max triplet within certain distance
    //   max_score_dist = 1.0e9;
    //   max_score = -1.0;
    //   float dist = 0.;
    //   for ( auto const& hit : *ev_lmhit ) {
    //     dist = 0.;
    //     for (int i=0; i<3; i++) {
    //       dist += (hit[i]-kpd.keypt[i])*(hit[i]-kpd.keypt[i]);
    //       vtx_sce[i] = kpd.keypt[i];
    //     }
    //     if ( dist<cut_off_distsq ) {
    //       if ( hit[13]>max_score ) {
    //         max_score = hit[13];
    //         max_score_dist = sqrt(dist);
    //       }
    //     }
    //   }

    ev_ana->Fill();
      

  }//end of event loop

  out->Write();

  return 0;
}
