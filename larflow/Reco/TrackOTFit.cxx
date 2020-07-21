#include "TrackOTFit.h"

#include <cmath>
#include <stdexcept>

namespace larflow {
namespace reco {

  void TrackOTFit::fit( std::vector< std::vector<float> >& initial_track,
                        std::vector< std::vector<float> >& track_pts_w_feat_v )
  {
    
  }

  // void TrackOTFit::fitByStepper( std::vector< std::vector<float> >& initial_track,
  //                                std::vector< std::vector<float> >& track_pts_w_feat_v )
  // {
  //   // go segment by segment, set max distance
  // }
  
  float TrackOTFit::d2_segment_point( const std::vector<float>& seg_start,
                                      const std::vector<float>& seg_end,
                                      const std::vector<float>& testpt )
  {

    std::vector<float> a(3,0);
    std::vector<float> b(3,0);
    float a2 = 0.;
    float b2 = 0.;
    float c2 = 0.;
    float ab = 0.;
    for (int i=0; i<3; i++) {
      a[i] = seg_start[i]-testpt[i];
      b[i] = seg_end[i]-seg_start[i];
      a2 += a[i]*a[i];
      b2 += b[i]*b[i];
      ab += a[i]*b[i];
      c2 += ( seg_end[i]-testpt[i] )*( seg_end[i]-testpt[i] );
    }

    if (b2<1e-9) {
      throw std::runtime_error( "[TrackOTFit::d2_segment_point] segment too short" );
    }

    float lenb = sqrt(b2);
    float s  = -ab/lenb;  // projection onto segment

    if ( s>lenb+0.5 || s<-0.5 ) {
      // past the segment end, return distance to end
      return c2;
    }

    float d2 = (a2*b2-ab*ab)*b2; // distance from point to line


    return d2;
    
  }

  std::vector<float> TrackOTFit::grad_d2_wrt_segend( const std::vector<float>& seg_start,
                                                     const std::vector<float>& seg_end,
                                                     const std::vector<float>& testpt )
  {

    std::vector<float> a(3,0);
    std::vector<float> b(3,0);
    float a2 = 0.;
    float b2 = 0.;
    float ab = 0.;
    float c2 = 0.;    
    for (int i=0; i<3; i++) {
      a[i] = seg_start[i]-testpt[i];
      b[i] = seg_end[i]-seg_start[i];
      a2 += a[i]*a[i];
      b2 += b[i]*b[i];
      ab += a[i]*b[i];
      c2 += ( seg_end[i]-testpt[i] )*( seg_end[i]-testpt[i] );      
    }

    if (b2<1e-9) {
      throw std::runtime_error( "[TrackOTFit::grad_d2_wrt_segend] segment too short" );
    }


    std::vector<float> grad_d2(3,0);

    
    float lenb = sqrt(b2);
    float s  = -ab/lenb;  // projection onto segment

    if ( s>lenb+0.5 || s<-0.5 ) {
      // past the segment end, return distance to end, so we return grad
      for (int i=0; i<3; i++ ) {
        grad_d2[i] = 2.0*( seg_end[i] - testpt[i] );
      }
      return grad_d2;
    }

    
    std::vector<float> db2(3,0); // partials of |b|^2
    std::vector<float> dab(3,0); // partials of a.b
    for (int i=0; i<3; i++ ) {
      db2[i] = 2*(seg_end[i]-seg_start[i]);
      dab[i] = (seg_start[i]-testpt[i]);
    }

    float c = a2*b2-ab*ab; // numerator of d2 formula
    for (int i=0; i<3; i++) {
      grad_d2[i] = (b2*(a2*db2[i]-2*ab*dab[i]) - c*db2[i])/(b2*b2);
    }

    return grad_d2;
    
  }

  void TrackOTFit::getLossAndGradient(  const std::vector< std::vector<float> >& initial_track,
                                        const std::vector< std::vector<float> >& track_pts_w_feat_v,
                                        float& loss,
                                        std::vector<float>& grad )
  {
    grad.resize(3,0);
    for (int i=0; i<3; i++)
      grad[i] = 0.;
    loss = 0.;

    const std::vector<float>& start = initial_track[0];
    const std::vector<float>& end   = initial_track[1];
    
    int ndatapts = track_pts_w_feat_v.size();
    for ( int ipt=0; ipt<ndatapts; ipt++ ) {
      const std::vector<float>& testpt = track_pts_w_feat_v[ipt];
      loss += d2_segment_point( start, end, testpt );
      std::vector<float> ptgrad = grad_d2_wrt_segend( start, end, testpt );

      for (int i=0; i<3; i++) {
        grad[i] += ptgrad[i]/float(ndatapts);
      }
      
    }
    
    loss /= float(ndatapts);

  }
  
}
}
