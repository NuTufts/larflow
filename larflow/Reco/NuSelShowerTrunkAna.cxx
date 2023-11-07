#include "NuSelShowerTrunkAna.h"

#include <sstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "TrackdQdx.h"
#include "geofuncs.h"

namespace larflow {
namespace reco {

  void NuSelShowerTrunkAna::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                     larflow::reco::NuSelectionVariables& output,
                                     larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();

    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );
    bool has_mc = false;
    if ( ev_mcshower != NULL && ev_mcshower->size()>0 )
      has_mc = true;

    //dqdx_algo.set_verbosity(larcv::msg::kINFO);
    dqdx_algo.set_verbosity( logger().level() );    
    
    // reco variables
    //  - dq/dx of first 3 cm
    //  - distance until below 1.5MIP
    // truth variables [maybe leave this to katie's truth matching]
    //  - distance between start and closest truth start
    //  - angle between shower prof dir and trunk of closest shower

    auto const& vtxpos = nuvtx.pos;
    TVector3 tvtx( vtxpos[0], vtxpos[1], vtxpos[2] );

    clear();

    _shower_dqdx_v.reserve( nuvtx.shower_v.size() );
    _shower_gapdist_v.reserve( nuvtx.shower_v.size() );    
    _shower_avedqdx_v.resize(4);
    _shower_ll_v.resize(4);
    for (int p=0; p<4; p++) {
      _shower_avedqdx_v[p].clear();
      _shower_ll_v[p].clear();
    }
    

    _shower_true_match_index_v.reserve( nuvtx.shower_v.size() );
    _shower_true_match_cos_v.reserve( nuvtx.shower_v.size() );
    _shower_true_match_vtx_err_dist_v.reserve( nuvtx.shower_v.size() );    
    
    for (size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++) {

      LARCV_DEBUG() << "start shower[" << ishower << "]" << std::endl;
      
      auto& shower       = nuvtx.shower_v[ishower];
      auto& shower_trunk = nuvtx.shower_trunk_v[ishower];
      auto& shower_pca   = nuvtx.shower_pcaxis_v[ishower];

      int num_trunk_points = shower_trunk.NumberTrajectoryPoints();
      LARCV_DEBUG() << "num trunk points: " << num_trunk_points << std::endl;

      std::vector<float> fend(3,0);
      std::vector<float> fstart(3,0);
      std::vector<float> fdir(3,0); 
      float gapdist = 0.;

      bool trunkok = refineShowerTrunk( shower_trunk, shower_pca, tvtx,
                                        fstart, fend, fdir, gapdist );

      if ( !trunkok ) {

        LARCV_DEBUG() << "Shower trunk is bad (too short), store sentinal values" << std::endl;
        storeShowerRecoSentinelValues();
        storeShowerTruthSentinelValues();
        continue;
      }

      dqdx_algo.clear();      
      dqdx_algo.processShower( shower, shower_trunk, shower_pca, adc_v, nuvtx );

      std::vector<float> plane_dqdx = dqdx_algo._pixsum_dqdx_v;
      std::vector<float> ave_dqdx(4,0);
      for (int p=0; p<3; p++) {
        ave_dqdx[p] = dqdx_algo._pixsum_dqdx_v[p];
      }
      ave_dqdx[3] = dqdx_algo._best_pixsum_dqdx;

      // fill data containers
      larlite::track shower_dqdx;
      float ll_e = 0.;
      float ll_g = 0.;
      float ll_tot = 0.0;
      if ( dqdx_algo._best_pixsum_plane>=0 ) {
        shower_dqdx = dqdx_algo.makeLarliteTrackdqdx(dqdx_algo._best_pixsum_plane);
        auto const& seg_v = dqdx_algo._plane_seg_dedx_v.at(dqdx_algo._best_pixsum_plane);
        for (int ipt=0; ipt<(int)seg_v.size(); ipt++) {
          auto const& seg = seg_v[ipt];
          TVector3 pos3d;
          for (int i=0; i<3; i++)
            pos3d[i] = 0.5*( seg.endpt[0][i]+seg.endpt[1][i] );
          ll_e = (seg.dqdx-250.0)/25.0;
          ll_g = (seg.dqdx-500.0)/50.0;
          ll_tot += ll_e*ll_e + ll_g*ll_g;
        }
        if ( seg_v.size()>0 )
          ll_tot /= float(seg_v.size());
        else
          ll_tot = -100.0;
      }
      
      _shower_dqdx_v.emplace_back( std::move(shower_dqdx) );
      for (int p=0; p<4; p++) {
        _shower_avedqdx_v[p].push_back( ave_dqdx[p] );
        _shower_ll_v[p].push_back( ll_tot );
      }
      _shower_gapdist_v.push_back( gapdist );


      // Truth variables
      if ( has_mc ) {
        try {
          dqdx_algo.calcGoodShowerTaggingVariables(  shower, shower_trunk, shower_pca, adc_v, *ev_mcshower );
          _shower_true_match_index_v.push_back( dqdx_algo._true_min_index );
          _shower_true_match_pdg_v.push_back( dqdx_algo._true_match_pdg );
          _shower_true_match_cos_v.push_back( dqdx_algo._true_dir_cos );
          _shower_true_match_vtx_err_dist_v.push_back( dqdx_algo._true_vertex_err_dist );
        }
        catch(...) {
          LARCV_WARNING() << "Error with RECO-TRUTH matching function" << std::endl;
          storeShowerTruthSentinelValues();          
        }
      }
      
    }//end of shower loop

    // store into output class
    output._shower_var_v.clear();
    output._shower_var_v.reserve(nuvtx.shower_v.size());
    
    int max_nhits = 0;
    float min_gap_dist = 1e6;
    output.largest_shower_ll = -100.0;
    output.closest_shower_ll = -100.0;
    output.largest_shower_avedqdx = -100;
    output.closest_shower_avedqdx = -100;
    
    for ( size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++) {

      // save shower variables
      larflow::reco::NuSelectionVariables::ShowerVar_t shrvar;
      shrvar.dqdx_ave = _shower_avedqdx_v[3][ishower];
      shrvar.llshower = _shower_ll_v[3][ishower];
      output._shower_var_v.emplace_back( std::move(shrvar) );

      // determine summary variables above
      int nhits = nuvtx.shower_v[ishower].size();

      // but only use showers above hit threshold
      if ( nhits<300 ) // hmmmmm
        continue;
      
      if ( nhits>max_nhits ) {
        max_nhits = nhits;
        output.largest_shower_ll = _shower_ll_v[3][ishower];
        output.largest_shower_avedqdx = _shower_avedqdx_v[3][ishower];
      }

      float dist = _shower_gapdist_v[ishower];
      if ( dist<min_gap_dist ) {
        min_gap_dist = dist;
        output.closest_shower_ll = _shower_ll_v[3][ishower];
        output.closest_shower_avedqdx = _shower_avedqdx_v[3][ishower];
      }
      
    }//loop over shower instances to make output variables
    
  }

  bool NuSelShowerTrunkAna::refineShowerTrunk( const larlite::track& shower_trunk,
                                               const larlite::pcaxis& shower_pca,
                                               const TVector3& tvtx,
                                               std::vector<float>& fstart,
                                               std::vector<float>& fend,
                                               std::vector<float>& fdir,
                                               float& gapdist )
  {
    
    fend.resize(3,0);
    fstart.resize(3,0);
    fdir.resize(3,0);
    gapdist = 0.;
    
    int closest_end = 0;
    int other_end = 1;
    float end_dist[2] = {0};
    end_dist[0] = (shower_trunk.LocationAtPoint(0)-tvtx).Mag();
    end_dist[1] = (shower_trunk.LocationAtPoint(1)-tvtx).Mag();
    if ( end_dist[1]<end_dist[0] ) {
      closest_end = 1;
      other_end = 0;
      gapdist = end_dist[1];
    }
    else {
      gapdist = end_dist[0];
    }
      
    TVector3 shower_dir = (shower_trunk.LocationAtPoint(other_end)-shower_trunk.LocationAtPoint(closest_end));
    float shower_trunk_len = shower_dir.Mag();
    if ( shower_trunk_len>0) {
      for (int i=0; i<3; i++) {
        shower_dir[i] /= shower_trunk_len;
      }
    }      
    LARCV_DEBUG() << "shower trunk length = " << shower_trunk_len << std::endl;

    if ( shower_trunk_len<0.5 ) {
      // trunk too short, try pca?
      LARCV_DEBUG() << "original trunk too short: " << shower_trunk_len << ", try using pca instead" << std::endl;
      float pcalen = 0.;
      float dx = 0.;
      try {
        for (int i=0; i<3; i++) {
          dx = (shower_pca.getEigenVectors()[3][i]-shower_pca.getEigenVectors()[4][i]);          
          pcalen += dx*dx;
        }
      }
      catch (...) {
        pcalen = 0.;
      }
        
      if ( pcalen<0.5 ) {
        // admit defeat for this shower
        LARCV_DEBUG() << "PCA also too shower: " << pcalen << ", stopped." << std::endl;
        return false;
      }//end of if pca is too short
      else {
        // good enough, find closest end
        LARCV_DEBUG() << "PCA len, " << pcalen << ", is acceptable" << std::endl;
        end_dist[0] = 0.;
        end_dist[1] = 0.;
        for (int i=0; i<3; i++) {
          end_dist[0] += ( shower_pca.getEigenVectors()[3][i]-tvtx[i] )*( shower_pca.getEigenVectors()[3][i]-tvtx[i] );
          end_dist[1] += ( shower_pca.getEigenVectors()[4][i]-tvtx[i] )*( shower_pca.getEigenVectors()[4][i]-tvtx[i] );            
        }
        shower_trunk_len = pcalen;
        if ( end_dist[0]<end_dist[1] ) {
          for (int i=0; i<3; i++) {
            shower_dir[i] = ( shower_pca.getEigenVectors()[4][i]-shower_pca.getEigenVectors()[3][i] )/pcalen;
            fstart[i] = shower_pca.getEigenVectors()[3][i];
            fend[i]   = shower_pca.getEigenVectors()[4][i];
          }
        }
        else {
          for (int i=0; i<3; i++) {
            shower_dir[i] = ( shower_pca.getEigenVectors()[3][i]-shower_pca.getEigenVectors()[4][i] )/pcalen;
            fstart[i] = shower_pca.getEigenVectors()[4][i];
            fend[i]   = shower_pca.getEigenVectors()[3][i];              
          }            
        }
      }//end of else pca len is long enough

    }// if original trunk is bad
    else {
      // current trunk is fine
      for (int i=0; i<3; i++) {
        fstart[i] = shower_trunk.LocationAtPoint(0)[i];
        fend[i]   = shower_trunk.LocationAtPoint(1)[i];
        fdir[i]   = shower_dir[i];
      }
    }

    return true;
  }

  // old dq/dx code  
  //   // gather points around the trunk      
  //   larlite::larflowcluster trunk_hits;
  //   for ( auto& hit : shower ) {
  //     std::vector<float> fhit = { hit[0], hit[1], hit[2] };
  //     float r = larflow::reco::pointLineDistance3f(  fstart, fend, fhit );
  //     float s = larflow::reco::pointRayProjection3f( fstart, fdir, fhit );
  //     if ( r<1.5 && s>0.0 && s<10.0 ) {
  //       trunk_hits.push_back( hit );
  //     }
  //   }
  
  //   LARCV_DEBUG() << "calc dqdx with " << trunk_hits.size() << " trunk hits" << std::endl;
  
  
  //   int npts = (int)shower_dqdx.NumberTrajectoryPoints();
  //   LARCV_DEBUG() << "return num points: " << npts << std::endl;
  
  //   if ( npts<2 ) {
  //     _shower_dqdx_v.emplace_back( std::move(shower_dqdx) );
  //     for (int p=0; p<4; p++) {
  //       _shower_avedqdx_v[p].push_back(-100);
  //       _shower_ll_v[p].push_back( 100 );
  //     }
  //     continue;
  //   }
  
  //   float enddist[2] = {0};
  //   for (int i=0; i<3; i++) {
  //     enddist[0] += ( shower_dqdx.LocationAtPoint(0)[i] - vtxpos[i] )*( shower_dqdx.LocationAtPoint(0)[i] - vtxpos[i] );
  //     enddist[1] += ( shower_dqdx.LocationAtPoint(npts-1)[i] - vtxpos[i] )*( shower_dqdx.LocationAtPoint(npts-1)[i] - vtxpos[i] );
  //   }
  
  //   closest_end = ( enddist[0]<=enddist[1] ) ? 0 : 1;
  //   other_end   = (closest_end==0) ? 1 : 0;
  //   int index_dir   = (closest_end==0) ? 1 : -1;
  //   int start_index = (closest_end==0) ? 0 : npts-1;
  //   int end_index   = (closest_end==0) ? npts-1 : 0;
  
  //   LARCV_DEBUG() << "closest end: " << closest_end << std::endl;
  
  //   std::vector<float> llshower(4,0);
  //   std::vector<float> ave_dqdx(4,0);
  //   int npoints = 0;
  
  //   for (int idx=start_index; idx!=end_index; idx += index_dir ) {
  
  //     //LARCV_DEBUG() << " pt[" << idx << "]" << std::endl;
  
  //     auto const& dqdxpt = shower_dqdx.LocationAtPoint(idx);
  //     float vtxdist = 0.;
  //     for (int i=0; i<3; i++) {
  //       vtxdist += ( vtxpos[i] - dqdxpt[i] )*( vtxpos[i] - dqdxpt[i] );
  //     }
  //     vtxdist = sqrt(vtxdist);
  //     if ( vtxdist<3.5 ) {
  
  //       std::vector<float> el_v(4,0);
  //       std::vector<float> ga_v(4,0);
  
  //       for (int p=0; p<4; p++) {
  //         el_v[p] = fabs(shower_dqdx.DQdxAtPoint( idx, (larlite::geo::View_t)p ) - 75.0)/25.0;
  //         ga_v[p] = fabs(shower_dqdx.DQdxAtPoint( idx, (larlite::geo::View_t)p ) - 150.0)/25.0;
  //         ave_dqdx[p] += shower_dqdx.DQdxAtPoint( idx, (larlite::geo::View_t)p );
  //         llshower[p] += el_v[p]*el_v[p] - ga_v[p]*ga_v[p];
  //       }
  
  //       npoints++;          
  //     }
  //   }//end of point loop
  
  //   if (npoints>0) {
  //     for (int p=0; p<4; p++) {
  //       llshower[p] /= (float)npoints;
  //       ave_dqdx[p] /= (float)npoints;
  //     }
  //   }
  
  //   _shower_dqdx_v.emplace_back( std::move(shower_dqdx) );
  //   std::stringstream ss_dqdx;
  //   ss_dqdx << "dQdx results: ";
  //   std::stringstream ss_ll;
  //   ss_ll << "LL results: ";      
  //   for (int p=0; p<4; p++) {
  //     _shower_avedqdx_v[p].push_back( ave_dqdx[p] );
  //     _shower_ll_v[p].push_back( llshower[p] );
  //     ss_dqdx << " [" << p << "] " << ave_dqdx[p];
  //     ss_ll << " [" << p << "] " << llshower[p];
  //   }
  //   ss_dqdx;
  //   ss_ll;
  //   LARCV_DEBUG() << ss_dqdx.str() << std::endl;
  //   LARCV_DEBUG() << ss_ll.str() << std::endl;
  // }//end of shower loop

  void NuSelShowerTrunkAna::clear()
  {
    _shower_dqdx_v.clear();
    _shower_avedqdx_v.clear();
    _shower_ll_v.clear();
    _shower_gapdist_v.clear();

    _shower_true_match_index_v.clear();
    _shower_true_match_cos_v.clear();
    _shower_true_match_vtx_err_dist_v.clear();

    _shower_avedqdx_v.clear();
    _shower_ll_v.clear();
    
  }
  
  
  void NuSelShowerTrunkAna::storeShowerRecoSentinelValues()
  {
    larlite::track shower_dqdx_empty;
    _shower_dqdx_v.emplace_back( std::move(shower_dqdx_empty) );
    _shower_gapdist_v.push_back( 100.0 );
    for (int p=0; p<4; p++) {
      _shower_avedqdx_v[p].push_back(-100);
      _shower_ll_v[p].push_back( 100 );
    }
  }

  void NuSelShowerTrunkAna::storeShowerTruthSentinelValues()
  {
    _shower_true_match_index_v.push_back( -1 );
    _shower_true_match_cos_v.push_back( -2 );
    _shower_true_match_vtx_err_dist_v.push_back( 1000 );
    _shower_true_match_pdg_v.push_back( 0 );            
  }
  
  
}
}
