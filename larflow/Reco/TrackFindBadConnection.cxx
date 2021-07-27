#include "TrackFindBadConnection.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ClusterImageMask.h"

namespace larflow {
namespace reco {
  
  std::vector<larlite::track >
  TrackFindBadConnection::splitBadTrack( const larlite::track& track,
                                         const std::vector<larcv::Image2D>& adc_v,
                                         float  mingap )
  {

    const int dpix = 0;
    
    std::vector<larlite::track> split_v;

    larflow::reco::ClusterImageMask masker;
    masker.set_verbosity( larcv::msg::kDEBUG );
    
    int npts = track.NumberTrajectoryPoints();
    if ( npts==1 )
      return split_v;
    
    for (int ipt=0; ipt<npts-1; ipt++) {
      auto const& start = track.LocationAtPoint(ipt);
      auto const& end   = track.LocationAtPoint(ipt+1);

      double len = (end-start).Mag();
      TVector3 dir = end-start;
      for (int i=0; i<3; i++)
        dir[i] /= len;

      if ( len>mingap ) {
        // check this gap which was probably made by merging two track segments
        // does it go over a bunch of charge?
        // if, so, we should not have merged it, since that is for jumping dead regions only.
        LARCV_DEBUG() << "Test gap: size=" << len << std::endl;

        larlite::track gap;
        gap.reserve(2);
        gap.add_vertex( start );
        gap.add_direction( dir );
        gap.add_vertex( end );
        gap.add_direction( dir );
        
        std::vector<larcv::Image2D> mask_v;
        for (int p=0; p<(int)adc_v.size(); p++) {
          larcv::Image2D blank(adc_v[p].meta());
          mask_v.emplace_back( std::move(blank) );
        }

        masker.maskTrack( gap, adc_v, mask_v, 10.0, dpix, dpix );
        std::vector<int> ncharged_v(adc_v.size(),0);
        for (int p=0; p<(int)adc_v.size(); p++) {
          for ( auto& pix : mask_v[p].as_vector() ) {
            ncharged_v[p]  += pix;
          }
        }
        masker.maskTrack( gap, adc_v, mask_v, -1.0, dpix, dpix );
        std::vector<int> nall_v(adc_v.size(),0);
        std::vector<float> frac_v(adc_v.size(),0);
        int nplanes_w_charge = 0;
        for (int p=0; p<(int)adc_v.size(); p++) {
          for ( auto& pix : mask_v[p].as_vector() ) {
            nall_v[p]  += pix;
          }
          if ( nall_v[p]>0 )
            frac_v[p] = float(ncharged_v[p])/float(nall_v[p]);
          if (frac_v[p]>0.5)
            nplanes_w_charge++;
          LARCV_DEBUG()  << "plane[" << p << "] masked=" << ncharged_v[p] << " nall=" << nall_v[p] << " frac=" << frac_v[p] << std::endl;          
        }
        LARCV_DEBUG() << "planes with charge: " << nplanes_w_charge << std::endl;

        if ( nplanes_w_charge==(int)adc_v.size() ) {
          // split the track
          larlite::track first;
          larlite::track second;
          first.reserve( ipt+1 );
          second.reserve( npts-(ipt+1) );
          for (int ii=0; ii<=ipt; ii++) {
            first.add_vertex( track.LocationAtPoint(ii) );
            first.add_direction( track.DirectionAtPoint(ii) );
          }
          for (int ii=ipt+1; ii<npts; ii++) {
            second.add_vertex( track.LocationAtPoint(ii) );
            second.add_direction( track.DirectionAtPoint(ii) );
          }

          split_v.emplace_back( std::move(first) );
          split_v.emplace_back( std::move(second) );
          return split_v;
        }
      }
      
    }
    
  }

  int TrackFindBadConnection::processNuVertexTracks( larflow::reco::NuVertexCandidate& nuvtx,
                                                     larcv::IOManager& iolcv )
  {
    larcv::EventImage2D* ev_adc =
      (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    
    const std::vector<larcv::Image2D>& adc_v = ev_adc->as_vector();
    
    int ntracks = nuvtx.track_v.size();
    int nsplit = 0;
    for (int i=0; i<ntracks; i++) {
      auto& track = nuvtx.track_v[i];
      std::vector<larlite::track> split_v = splitBadTrack( track, adc_v, 5.0 );
      if ( split_v.size()>0 && split_v.at(0).NumberTrajectoryPoints()>3 ) {
        std::swap( nuvtx.track_v[i], split_v[0] );
        nsplit++;
      }
    }
    
    LARCV_INFO() << "Number of tracks split: " << nsplit << std::endl;
  }
}
}
