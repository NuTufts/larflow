#include <iostream>

#include "TFile.h"
#include "TTree.h"

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/storage_manager.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/larflow3dhit.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"

#include "larflow/Reco/geofuncs.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"

int main( int nargs, char** argv )
{

  std::string dlmerged_input_file = "../../../testdata/mcc9_v29e_wc_bnb_overlay_run3g/merged_dlreco_497be540-00f9-49a8-9f80-7846143c4fce.root";  
  std::string larmatch_input_file = "../../../testdata/mcc9_v29e_wc_bnb_overlay_run3g/larmatch_kps_497be540-00f9-49a8-9f80-7846143c4fce_larlite.root";

  larlite::storage_manager io( larlite::storage_manager::kREAD );
  io.add_in_filename( dlmerged_input_file );
  io.add_in_filename( larmatch_input_file );
  io.set_data_to_read( larlite::data::kLArFlow3DHit, "larmatch" );
  io.set_data_to_read( larlite::data::kMCTrack,  "mcreco" );
  io.set_data_to_read( larlite::data::kMCShower, "mcreco" );
  io.set_data_to_read( larlite::data::kMCTruth,  "generator" );  
  io.open();

  larcv::IOManager iolcv( larcv::IOManager::kREAD, "IOManager", larcv::IOManager::kTickBackward );
  iolcv.add_in_file( dlmerged_input_file );
  iolcv.specify_data_read( larcv::kProductImage2D, "wire" );
  iolcv.reverse_all_products();
  iolcv.initialize();

  int nentries = io.get_entries();  

  larutil::SpaceChargeMicroBooNE sce;
  const std::vector<Double_t> orthy = larutil::Geometry::GetME()->GetOrthVectorsY();
  const std::vector<Double_t> orthz = larutil::Geometry::GetME()->GetOrthVectorsZ();
 
  TFile* out = new TFile("anaout_true_dedx.root","recreate");
  TTree* ana = new TTree("anadedx", "Analysis of dQdx using truth tracks");

  int pid;
  float res;
  float rad;
  float pixval;
  float dqdx;
  float lm;

  ana->Branch("pid",&pid,"pid/I");
  ana->Branch("res",&res,"res/F");
  ana->Branch("rad",&rad,"rad/F");
  ana->Branch("pixval",&pixval,"pixval/F");
  ana->Branch("dqdx",&dqdx,"dqdx/F");
  ana->Branch("lm",&lm,"lm/F");    

  std::cout << "NUM ENTRIES: " << nentries << std::endl;
  for (int ientry=0; ientry<nentries; ientry++ ) {

    std::cout << "===[ ENTRY " << ientry << " ]===" << std::endl;
    
    io.go_to(ientry);
    iolcv.read_entry(ientry);

    // Get wire image
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector<larcv::Image2D>& adc_v = ev_adc->as_vector();
    
    // Get truth tracks
    std::vector< const larlite::mctrack* > primary_muons_v;
    std::vector< const larlite::mctrack* > primary_protons_v;

    larlite::event_mctrack* ev_mctrack = (larlite::event_mctrack*)io.get_data( larlite::data::kMCTrack, "mcreco" );
    for (auto const& track : *ev_mctrack ) {
      if ( track.Origin()==larlite::simb::kBeamNeutrino ) {

        if ( track.TrackID()==track.MotherTrackID() ) {

          if ( abs(track.PdgCode())==13 )
            primary_muons_v.push_back( &track );
          if ( abs(track.PdgCode())==2212 )
            primary_protons_v.push_back( &track );
          
        }
        
      }
    }

    for ( auto const& ptrack : primary_muons_v ) {
      std::cout << "Muon Track: start=(" << ptrack->Start().X() << "," << ptrack->Start().Y() << "," << ptrack->Start().Z() << ")"
                << " end=(" << ptrack->End().X() << "," << ptrack->End().Y() << "," << ptrack->End().Z() << ")"
                << std::endl;
    }
    for ( auto const& ptrack : primary_protons_v ) {
      std::cout << "Proton Track: start=(" << ptrack->Start().X() << "," << ptrack->Start().Y() << "," << ptrack->Start().Z() << ")"
                << " end=(" << ptrack->End().X() << "," << ptrack->End().Y() << "," << ptrack->End().Z() << ")"
                << std::endl;
    }

    // GET LARMATCH POINTS
    larlite::event_larflow3dhit* ev_larmatch =
      (larlite::event_larflow3dhit*)io.get_data(larlite::data::kLArFlow3DHit, "larmatch" );
    

    std::vector< std::vector< const larlite::mctrack* >* > track_list_v;
    track_list_v.push_back( &primary_muons_v );
    track_list_v.push_back( &primary_protons_v );
    std::vector<int> track_list_pid_v = { 13, 2212 };

    int itracklist = -1;
    for ( auto const& ptracklist : track_list_v ) {
      itracklist++;
      pid = track_list_pid_v[itracklist];
    
      // loop over muon tracks, get larmatch points within some distance along trajectory
      // assign charge to larmatch points, assign residual range distance, plot
      for ( auto const& ptrack : *ptracklist ) {

        std::vector<int> hit_index_v;
        hit_index_v.reserve( ev_larmatch->size() );
        
        // set 3D bounds and get step points in apparent detector coordinates
        std::vector< std::vector<float> > bounds(3);
        for (int dim=0; dim<3; dim++) {
          bounds[dim].resize(2,0);
          bounds[dim][0] = 1e9;
          bounds[dim][1] = -1e9;
        }
        
        
        std::vector< std::vector<float> > detpath;
        std::vector< std::vector<float> > detdir;        
        
        for ( auto const& step : *ptrack ) {
          
          std::vector<float> fpt = { step.T(), step.X(), step.Y(), step.Z() };
          
          float tick = ublarcvapp::mctools::CrossingPointsAnaMethods::getTick( fpt, 4050.0, &sce );
          std::vector<double> offsets = sce.GetPosOffsets( fpt[1], fpt[2], fpt[3] );
          
          std::vector<float> pathpt(3,0);
          pathpt[0] = (tick-3200.0)*larutil::LArProperties::GetME()->DriftVelocity()*0.5;
          pathpt[1] = step.Y() + offsets[1];
          pathpt[2] = step.Z() + offsets[2];
          
          for (int dim=0; dim<3; dim++) {
            if (bounds[dim][0]>pathpt[dim])
              bounds[dim][0] = pathpt[dim];
            if (bounds[dim][1]<pathpt[dim])
              bounds[dim][1] = pathpt[dim];
          }
          
          detpath.push_back( pathpt );
        }
        
        std::cout << " track bounds x:[" << bounds[0][0] << "," << bounds[0][1] << "] "
                  << "y:[" << bounds[1][0] << "," << bounds[1][1] << "] "
                  << "z:[" << bounds[2][0] << "," << bounds[2][1] << "] "
                  << std::endl;
        
        // narrow down the search set
        std::vector<int> search_index_v;
        std::vector< std::vector<float> > point_v;
        std::vector< std::vector<int> > imgcoord_v;
        
        for (int ihit=0; ihit<ev_larmatch->size(); ihit++) {
          
          auto const& hit = (*ev_larmatch)[ihit];
          if ( hit[0]<bounds[0][0]-1.0 || hit[0]>bounds[0][1]+1.0
               || hit[1]<bounds[1][0]-1.0 || hit[1]>bounds[1][1]+1.0
               || hit[2]<bounds[2][0]-1.0 || hit[2]>bounds[2][1]+1.0 )
            continue;
          
          search_index_v.push_back( ihit );
          std::vector<float> pt = { hit[0], hit[1], hit[2] };
          point_v.push_back( pt );
          std::vector<int> imgcoord = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
          imgcoord_v.push_back( imgcoord );
        }

        std::cout << "number of hits inside bounding box: " << point_v.size() << std::endl;
        
        // now collect hits along track
        struct TrackPt_t {
          int hitidx;
          float s;
          float r;
          float q;
          float dqdx;
          float lm;
          std::vector<float> pt;
          bool operator<( const TrackPt_t& rhs ) const
          {
            if ( s>rhs.s) return true;
            return false;
          };
        };
        
        float current_len = 0.;
        std::vector< TrackPt_t > trackpt_v;
        
        for ( int istep=0; istep<(int)detpath.size()-1; istep++ ) {
          std::vector<float>& start = detpath[istep];
          std::vector<float>& end   = detpath[istep+1];
          std::vector<float> dir(3,0);
          std::vector<float> truedir(3,0);          
          float len = 0.;
          float truelen = 0.;
          for (int dim=0; dim<3; dim++) {
            dir[dim] += end[dim]-start[dim];
            len += dir[dim]*dir[dim];

            truedir[dim] = ptrack->at(istep+1).Position()[dim]-ptrack->at(istep).Position()[dim];
            truelen += truedir[dim]*truedir[dim];
          }
          len = sqrt(len);
          truelen = sqrt(truelen);
          
          if (len<=3.0 )
            continue;
          
          for (int i=0; i<3; i++ ) {
            dir[i] /= len;
            truedir[i] /= truelen;
          }
          
          for (int ii=0; ii<(int)point_v.size(); ii++) {
            auto const& pt = point_v[ii];
            auto const& imgcoord = imgcoord_v[ii];
            float r = larflow::reco::pointLineDistance3f( start, end, pt );
            float s = larflow::reco::pointRayProjection3f( start, dir, pt );
            //std::cout << "  point: r=" << r << " s=" << s << std::endl;
            
            if ( r>5.0 || s<0 || s>len ) {
              continue;
            }

            // on segment
            TrackPt_t trkpt;
            trkpt.pt     = pt;
            trkpt.hitidx = search_index_v[ii];
            trkpt.r = r;
            trkpt.s = s+current_len;
            trkpt.q = 0.;            
            trkpt.dqdx = 0.;
            trkpt.lm = ev_larmatch->at(trkpt.hitidx).track_score;

            // get the median charge inside the image
            int row = adc_v.front().meta().row( imgcoord[3] );

            std::vector<float> dqdx_v(3,0);
            std::vector<float> pixval_v(3,0);            
            for ( int p=0; p<3; p++) {

              float pixsum = 0.;
              int npix = 0;
              for (int dr=-2; dr<=2; dr++ ) {
                int r = row+dr;
                if ( r<0 || r>=(int)adc_v.front().meta().rows() )
                  continue;
                pixsum += adc_v[p].pixel( r, imgcoord[p] );
                npix++;
              }
              if ( npix>0 )
                pixval_v[p] = pixsum/float(npix);
              else
                pixval_v[p] = 0;
              
              float dcos_yz = fabs(truedir[1]*orthy[p] + truedir[2]*orthz[p]);
              float dcos_x  = fabs(truedir[0]);
              float dx = 3.0;
              if ( dcos_yz>0.785 )
                dx = 3.0/dcos_yz;
              else
                dx = 3.0/dcos_x;
              dqdx_v[p] = pixval_v[p]/dx;
            }
            // median value
            //std::sort( pixval_v.begin(), pixval_v.end() );
            //std::sort( dqdx_v.begin(), dqdx_v.end() ); 
            //trkpt.q = pixval_v[1];
            //trkpt.dqdx = dqdx_v[1];

            // y-plane only
            trkpt.q = pixval_v[2];
            trkpt.dqdx = dqdx_v[2];
            
            trackpt_v.push_back( trkpt );
          }//end of point loop
          
          
          current_len += len;
        }//end of loop over detpath steps
        
        std::cout << "Number of hits assigned to track: " << trackpt_v.size() << std::endl;
        std::cout << "Total length of track: " << current_len << " cm" << std::endl;
        std::sort( trackpt_v.begin(), trackpt_v.end() );

        for ( auto& trkpt : trackpt_v ) {
          res = current_len - trkpt.s;
          pixval = trkpt.q;
          dqdx = trkpt.dqdx;
          rad = trkpt.r;
          lm = trkpt.lm;
          ana->Fill();
        }
        
        
      }//end of loop over tracks
    }//end of track list
    
    
  }

  out->Write();
  
  io.close();
  iolcv.finalize();
  
  return 0;
}
