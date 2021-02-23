#include "NuSelShowerTrunkAna.h"

#include <sstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "TrackdQdx.h"
#include "geofuncs.h"

namespace larflow {
namespace reco {

  void NuSelShowerTrunkAna::analyze( larflow::reco::NuVertexCandidate& nuvtx,
                                     larflow::reco::NuSelectionVariables& output,
                                     larcv::IOManager& iolcv )
  {

    larcv::EventImage2D* ev_adc
      = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_adc->as_vector();
    
    // reco variables
    //  - dq/dx of first 3 cm
    //  - distance until below 1.5MIP
    // truth variables [maybe leave this to katie's truth matching]
    //  - distance between start and closest truth start
    //  - angle between shower prof dir and trunk of closest shower

    auto const& vtxpos = nuvtx.pos;
    TVector3 tvtx( vtxpos[0], vtxpos[1], vtxpos[2] );
    larflow::reco::TrackdQdx algo;
    _shower_dqdx_v.clear();
    _shower_dqdx_v.reserve( nuvtx.shower_v.size() );

    _shower_avedqdx_v.clear();
    _shower_avedqdx_v.resize(4);
    
    _shower_ll_v.clear();
    _shower_ll_v.resize(4);

    _shower_gapdist_v.clear();
    _shower_gapdist_v.reserve( nuvtx.shower_v.size() );
    
    for (int p=0; p<4; p++) {
      _shower_avedqdx_v[p].clear();
      _shower_ll_v[p].clear();
    }
    
    for (size_t ishower=0; ishower<nuvtx.shower_v.size(); ishower++) {

      LARCV_DEBUG() << "start shower[" << ishower << "]" << std::endl;
      
      auto& shower       = nuvtx.shower_v[ishower];
      auto& shower_trunk = nuvtx.shower_trunk_v[ishower];
      auto& shower_pca   = nuvtx.shower_pcaxis_v[ishower];

      int num_trunk_points = shower_trunk.NumberTrajectoryPoints();
      LARCV_DEBUG() << "num trunk points: " << num_trunk_points << std::endl;

      int closest_end = 0;
      int other_end = 1;
      float end_dist[2] = {0};
      float gapdist = 0.;
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
      _shower_gapdist_v.push_back( gapdist );
      
      TVector3 shower_dir = (shower_trunk.LocationAtPoint(other_end)-shower_trunk.LocationAtPoint(closest_end));
      float shower_trunk_len = shower_dir.Mag();
      if ( shower_trunk_len>0) {
        for (int i=0; i<3; i++) {
          shower_dir[i] /= shower_trunk_len;
        }
      }      
      LARCV_DEBUG() << "shower trunk length = " << shower_trunk_len << std::endl;

      // gather points around the trunk
      std::vector<float> fend(3,0);
      std::vector<float> fstart(3,0);
      std::vector<float> fdir(3,0);
      for (int i=0; i<3; i++) {
        fstart[i] = shower_trunk.LocationAtPoint(0)[i];
        fend[i]   = shower_trunk.LocationAtPoint(1)[i];
        fdir[i]   = shower_dir[i];
      }

      larlite::larflowcluster trunk_hits;
      for ( auto& hit : shower ) {
        std::vector<float> fhit = { hit[0], hit[1], hit[2] };
        float r = larflow::reco::pointLineDistance3f(  fstart, fend, fhit );
        float s = larflow::reco::pointRayProjection3f( fstart, fdir, fhit );
        if ( r<1.5 && s>0.0 && s<10.0 ) {
          trunk_hits.push_back( hit );
        }
      }
      
      LARCV_DEBUG() << "calc dqdx with " << trunk_hits.size() << " trunk hits" << std::endl;
      larlite::track shower_dqdx = algo.calculatedQdx( shower_trunk, trunk_hits, adc_v );

      int npts = (int)shower_dqdx.NumberTrajectoryPoints();
      LARCV_DEBUG() << "return num points: " << npts << std::endl;

      if ( npts<2 ) {
        _shower_dqdx_v.emplace_back( std::move(shower_dqdx) );
        for (int p=0; p<4; p++) {
          _shower_avedqdx_v[p].push_back(-100);
          _shower_ll_v[p].push_back( 100 );
        }
        continue;
      }
      
      float enddist[2] = {0};
      for (int i=0; i<3; i++) {
        enddist[0] += ( shower_dqdx.LocationAtPoint(0)[i] - vtxpos[i] )*( shower_dqdx.LocationAtPoint(0)[i] - vtxpos[i] );
        enddist[1] += ( shower_dqdx.LocationAtPoint(npts-1)[i] - vtxpos[i] )*( shower_dqdx.LocationAtPoint(npts-1)[i] - vtxpos[i] );
      }

      closest_end = ( enddist[0]<=enddist[1] ) ? 0 : 1;
      other_end   = (closest_end==0) ? 1 : 0;
      int index_dir   = (closest_end==0) ? 1 : -1;
      int start_index = (closest_end==0) ? 0 : npts-1;
      int end_index   = (closest_end==0) ? npts-1 : 0;

      LARCV_DEBUG() << "closest end: " << closest_end << std::endl;

      std::vector<float> llshower(4,0);
      std::vector<float> ave_dqdx(4,0);
      int npoints = 0;
      
      for (int idx=start_index; idx!=end_index; idx += index_dir ) {

        //LARCV_DEBUG() << " pt[" << idx << "]" << std::endl;
        
        auto const& dqdxpt = shower_dqdx.LocationAtPoint(idx);
        float vtxdist = 0.;
        for (int i=0; i<3; i++) {
          vtxdist += ( vtxpos[i] - dqdxpt[i] )*( vtxpos[i] - dqdxpt[i] );
        }
        vtxdist = sqrt(vtxdist);
        if ( vtxdist<3.5 ) {

          std::vector<float> el_v(4,0);
          std::vector<float> ga_v(4,0);

          for (int p=0; p<4; p++) {
            el_v[p] = fabs(shower_dqdx.DQdxAtPoint( idx, (larlite::geo::View_t)p ) - 75.0)/25.0;
            ga_v[p] = fabs(shower_dqdx.DQdxAtPoint( idx, (larlite::geo::View_t)p ) - 150.0)/25.0;
            ave_dqdx[p] += shower_dqdx.DQdxAtPoint( idx, (larlite::geo::View_t)p );
            llshower[p] += el_v[p]*el_v[p] - ga_v[p]*ga_v[p];
          }
          
          npoints++;          
        }
      }//end of point loop

      if (npoints>0) {
        for (int p=0; p<4; p++) {
          llshower[p] /= (float)npoints;
          ave_dqdx[p] /= (float)npoints;
        }
      }
      
      _shower_dqdx_v.emplace_back( std::move(shower_dqdx) );
      std::stringstream ss_dqdx;
      ss_dqdx << "dQdx results: ";
      std::stringstream ss_ll;
      ss_ll << "LL results: ";      
      for (int p=0; p<4; p++) {
        _shower_avedqdx_v[p].push_back( ave_dqdx[p] );
        _shower_ll_v[p].push_back( llshower[p] );
        ss_dqdx << " [" << p << "] " << ave_dqdx[p];
        ss_ll << " [" << p << "] " << llshower[p];
      }
      ss_dqdx;
      ss_ll;
      LARCV_DEBUG() << ss_dqdx.str() << std::endl;
      LARCV_DEBUG() << ss_ll.str() << std::endl;
    }//end of shower loop


    // store into output class
    output._shower_var_v.clear();
    output._shower_var_v.reserve(nuvtx.shower_v.size());

    int max_nhits = 0;
    float min_gap_dist = 1e6;
    output.largest_shower_ll = 100.0;
    output.closest_shower_ll = 100.0;
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
      if ( nhits<500 ) // hmmmmm
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
  
  
}
}
