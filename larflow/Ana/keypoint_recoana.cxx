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
#include "larflow/Reco/KPCluster.h"

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
  
  // Load inputs
  larcv::IOManager iocv( larcv::IOManager::kREAD, "larcv", larcv::IOManager::kTickBackward );
  iocv.add_in_file( dlmerged_file );
  iocv.reverse_all_products();
  iocv.initialize();

  larlite::storage_manager ioll( larlite::storage_manager::kREAD );
  ioll.add_in_filename( dlmerged_file );
  ioll.open();

  TFile kpreco_tfile(kpreco_file.c_str(),"open");
  TTree* kpreco_ttree = (TTree*)kpreco_tfile.Get("larflow_keypointreco");
  std::vector<larflow::reco::KPCluster>* ev_kpreco = 0;
  kpreco_ttree->SetBranchAddress( "kpcluster_v", &ev_kpreco );

  // load outputs
  TFile* out = new TFile(outfilename.c_str(),"new");

  // define output tree
  TTree* ev_ana = new TTree("kprecoana_event","Keypoint Reco Ana Event tree");

  int run,subrun,event;
  
  int n_reco_kp;
  int n_near_true_vtx;
  int n_near_true_trackend;
  int n_near_true_showerstart;
  int n_false_keypoint;

  int n_reco_kp_wct;  
  int n_near_true_vtx_wct;
  int n_near_true_trackend_wct;
  int n_near_true_showerstart_wct;
  int n_false_keypoint_wct;

  float vtx_qsum[3] = { 0., 0., 0. };
  float min_dist_to_vtx = 0.;
  float min_dist_to_vtx_dl = 0.;
  int has_good_dl_vertex = 0;
  
  ev_ana->Branch("run",&run,"run/I");
  ev_ana->Branch("subrun",&subrun,"subrun/I");
  ev_ana->Branch("event",&event,"event/I");  

  ev_ana->Branch("n_reco_kp",&n_reco_kp,"n_reco_kp/I");
  ev_ana->Branch("n_near_true_vtx",&n_near_true_vtx,"n_near_true_vtx/I");
  ev_ana->Branch("n_near_true_trackend",&n_near_true_trackend,"n_near_true_trackend/I");
  ev_ana->Branch("n_near_true_showerstart",&n_near_true_showerstart,"n_near_true_showerstart/I");
  ev_ana->Branch("n_false_keypoint",&n_false_keypoint,"n_false_keypoint/I");

  ev_ana->Branch("n_reco_kp_wct",&n_reco_kp_wct,"n_reco_kp_wct/I");  
  ev_ana->Branch("n_near_true_vtx_wct",&n_near_true_vtx_wct,"n_near_true_vtx_wct/I");
  ev_ana->Branch("n_near_true_trackend_wct",&n_near_true_trackend_wct,"n_near_true_trackend_wct/I");
  ev_ana->Branch("n_near_true_showerstart_wct",&n_near_true_showerstart_wct,"n_near_true_showerstart_wct/I");
  ev_ana->Branch("n_false_keypoint_wct",&n_false_keypoint_wct,"n_false_keypoint_wct/I");

  ev_ana->Branch("min_dist_to_vtx", &min_dist_to_vtx, "min_dist_to_vtx/F" );
  ev_ana->Branch("min_dist_to_vtx_dl", &min_dist_to_vtx_dl, "min_dist_to_vtx_dl/F" );
  ev_ana->Branch("has_good_dl_vertex", &has_good_dl_vertex, "has_good_dl_vertex/I" );
  ev_ana->Branch("vtx_qsum",vtx_qsum, "vtx_qsum[3]/F" );

  TTree* kp_ana = new TTree("kprecoana_keypt","Keypoint Reco Ana per True Keypoint tree");  
  float vtx_sce[3];
  float max_score;
  float max_score_dist;
  int is_nu_vtx;
  int n_nearby_5cm;
  kp_ana->Branch("vtx_sce",vtx_sce,"vtx_sce[3]/F");
  kp_ana->Branch("max_score",&max_score,"max_score/F");
  kp_ana->Branch("max_score_dist",&max_score_dist,"max_score_dist/F");
  kp_ana->Branch("n_nearby_5cm",&n_nearby_5cm,"n_nearby_5cm/I");
  kp_ana->Branch("is_nu_vtx",&is_nu_vtx,"is_nu_vtx/I");

  
  const float cut_off_dist = 10.0; // cm
  const float cut_off_distsq = cut_off_dist*cut_off_dist;

  // Keypoint Truth Data Maker
  larflow::keypoints::PrepKeypointData kpdata;
  ublarcvapp::mctools::LArbysMC lmc;
  lmc.bindAnaVariables( ev_ana );
  
  int nentries = iocv.get_n_entries();
  for (int ientry=0; ientry<nentries; ientry++) {

    // load trees
    iocv.read_entry(ientry);
    ioll.go_to(ientry);
    kpreco_ttree->GetEntry(ientry);

    // Get neutrino interaction truth
    lmc.process( ioll );
    float true_vtx[3] = { lmc._vtx_sce_x, lmc._vtx_sce_y, lmc._vtx_sce_z };

    // truth keypoints
    kpdata.process( iocv, ioll );
    
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
    std::cout << "  number of truth keypoints: " << kpdata.getKPdata().size() << std::endl;
    std::cout << "  number of reco keypoints: " << ev_kpreco->size() << std::endl;
    std::cout << "  number of Wirecell images: " << thrumu_v.size() << std::endl;
    std::cout << "  true neutrino vertex: (" << true_vtx[0] << "," << true_vtx[1] << "," << true_vtx[2] << ")" << std::endl;
    std::cout << "  number of DL reco vertices: " << ev_pgraph->PGraphArray().size() << std::endl;
    lmc.printInteractionInfo();
    kpdata.printKeypoints();
    
    n_reco_kp = (int)ev_kpreco->size();
    
    // KP Reco loop
    n_near_true_vtx = 0;
    n_near_true_trackend = 0;
    n_near_true_showerstart = 0;
    n_false_keypoint = 0;

    n_reco_kp_wct = 0;
    n_near_true_vtx_wct = 0;
    n_near_true_trackend_wct = 0;
    n_near_true_showerstart_wct = 0;
    n_false_keypoint_wct = 0;

    // characterize vertex
    min_dist_to_vtx = 1.0e9;

    // get vertex activity
    for (int p=0; p<3; p++) vtx_qsum[p] = 0.0;
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
                vtx_qsum[p] += adc_v[p].pixel(r,c);
            }//end of col loop
          }//end of row loop
        }//end of if valid wire
      }//end of plane loop
    }//end of if valid tick

    // get dl vertex info
    has_good_dl_vertex = 0;
    min_dist_to_vtx_dl = 1.0e9;
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
      }
    }
    if ( min_dist_to_vtx_dl<5.0 )
      has_good_dl_vertex = 1;
    
    for ( auto const& kpc : *ev_kpreco ) {

      // determine if on Wirecell-tagged cosmic
      // requirement: be on tagged pixel for 2/3 planes
      bool iscosmic = false;

      float tick = 3200 + kpc.max_pt_v[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
      if ( tick<=thrumu_v[0].meta().min_y() || tick>=thrumu_v[0].meta().max_y() ) {
        iscosmic = true;
      }
      else {
        int row = thrumu_v[0].meta().row( tick );        
        int nplanes_on_wctpixel = 0;
        for ( auto const& img : thrumu_v ) {
          // out of image?  we remove it implicitly by adding to counter
          std::vector<double> dpos = { (double)kpc.max_pt_v[0],
                                       (double)kpc.max_pt_v[1],
                                       (double)kpc.max_pt_v[2] };
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
      float reco_vtx_dist = 0.;
      for (int i=0; i<3; i++) {
        reco_vtx_dist += (kpc.max_pt_v[i]-true_vtx[i])*(kpc.max_pt_v[i]-true_vtx[i]);
      }
      reco_vtx_dist = sqrt(reco_vtx_dist);
      if ( reco_vtx_dist < min_dist_to_vtx )
        min_dist_to_vtx = reco_vtx_dist;

      
      // loop over truth keypoints and determine if near by
      int num_nearby = 0;
      float min_dist = 1.0e9;
      for ( auto const& kpd : kpdata.getKPdata() ) {
        
        float dist = 0.;
        for (int i=0; i<3; i++) {
          dist += (kpc.max_pt_v[i]-kpd.keypt[i])*(kpc.max_pt_v[i]-kpd.keypt[i]);
        }
        dist = sqrt(dist);

        if ( min_dist>dist )
          min_dist = dist;

        if ( dist>5.0 ) {
          continue;
        }

        // is nearby
        num_nearby++;

        // is neutrino vertex?
        float vtx_dist = 0.;
        for (int i=0; i<3; i++) {
          vtx_dist += (kpd.keypt[i]-true_vtx[i])*(kpd.keypt[i]-true_vtx[i]);
        }
        vtx_dist = sqrt(vtx_dist);
        if ( vtx_dist<1.0 ) {
          n_near_true_vtx++;
          if ( !iscosmic ) n_near_true_vtx_wct++;
          continue;
        }

        // is shower start?
        if ( kpd.is_shower==1 ) {
          n_near_true_showerstart++;
          if ( !iscosmic ) n_near_true_showerstart++;
          continue;
        }

        // is near track end
        n_near_true_trackend++;
        if ( !iscosmic ) n_near_true_trackend_wct++;

        
      }//end of loop over true keypoints

      //std::cout << "[reco keypoint]  closest true keypoint: " << min_dist << " cm" << std::endl;
      
      if ( num_nearby==0 ) {
        n_false_keypoint++;
        if ( !iscosmic ) n_false_keypoint_wct++;
      }
      
    }//end of loop over reco keypoints

    std::cout << "  nearby true nu vtx: " << n_near_true_vtx << " w/ wire-cell tagger: " << n_near_true_vtx_wct << std::endl;
    std::cout << "  nearby true shower start: " << n_near_true_showerstart << " w/ wire-cell tagger: " << n_near_true_showerstart_wct << std::endl;
    std::cout << "  nearby true track end: " << n_near_true_trackend << " w/ wire-cell tagger: " << n_near_true_trackend_wct << std::endl;
    std::cout << "  false keypoint: " << n_false_keypoint << " w/ wire-cell tagger: " << n_false_keypoint_wct << std::endl;
    
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
