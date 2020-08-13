#include "TrackOTFit.h"

#include <cmath>
#include <stdexcept>
#include <iostream>

namespace larflow {
namespace reco {

  /**
   * @brief fit points to a line segment. we vary only the last end point
   * 
   * @param[out] initial_track Initial track points, consisting of a start and end point.
   *             expects the outer vector to have length 2. 
   *             The first inner vector is the start 3d position of the line segment.
   *             The second inner vector is the end 3d position of the line segment.
   *             The second 'end' position is updated by the routine.
   * @param[in]  track_pts_w_feat_v vector of space points (vector<float>) to fit to
   * @param[in]  _maxiters_ Maximum number of fitting iterations. Default 100.
   * @param[in]  lr learning rate. default 0.1.
   */
  void TrackOTFit::fit_segment( std::vector< std::vector<float> >& initial_segment,
                                std::vector< std::vector<float> >& track_pts_w_feat_v,
                                const int _maxiters_,
                                const float lr )
  {

    int iter=0;
    // const int _maxiters_ = 1000;
    // float lr = 0.1;

    float first_loss = -1;
    float current_loss = -1;

    std::vector< std::vector<float> > current_segment;
    current_segment.push_back( initial_segment[0] );
    current_segment.push_back( initial_segment[1] );
    
    while ( iter<_maxiters_ ) {

      float iter_loss = 0;
      float iter_weight = 0.;        
      std::vector<float> iter_grad(3,0);

      larflow::reco::TrackOTFit::getWeightedLossAndGradient( current_segment,
                                                             track_pts_w_feat_v,
                                                             iter_loss,
                                                             iter_weight,
                                                             iter_grad );

      if ( first_loss<0 )
        first_loss = iter_loss;
      
      // update
      float gradlen = 0.;
      for (int i=0; i<3; i++ ) {
        current_segment[1][i] += -lr*iter_grad[i];
        gradlen += iter_grad[i]*iter_grad[i];
      }
      current_loss = iter_loss;

      std::cout << "[TrackOTFit::fit_segment] iter[" << iter << "] "
                << " grad=(" << iter_grad[0] << "," << iter_grad[1] << "," << iter_grad[2] << ")"
                << " len=" << sqrt(gradlen)
                << " currentvtx=(" << current_segment[1][0] << "," << current_segment[1][1] << "," << current_segment[1][2] << ")"
                << " loss=" << current_loss
                << std::endl;
      
      if ( sqrt(gradlen)<1.0e-3 )
        break;
      iter++;
    }

    std::cout << "[TrackOTFit::fit_segment] FIT RESULTS -----------------" << std::endl;
    std::cout << "  num iterations: " << iter << std::endl;
    std::cout << "  original vertex: (" << current_segment[1][0] << "," << current_segment[1][1] << "," << current_segment[1][2] << ")" << std::endl;
    std::cout << "  final vertex: (" << current_segment[1][0] << "," << current_segment[1][1] << "," << current_segment[1][2] << ")" << std::endl;
    std::cout << "  original loss: " << first_loss << std::endl;
    std::cout << "  current loss: " << current_loss << std::endl;    
    std::cout << "-----------------------------------------------------------" << std::endl;

    initial_segment[1] = current_segment[1];
    
  }

  /**
   * @brief calculate the squared-distance from a line segment
   *
   * Use line-distance formala if test point projects onto the segment.
   * Use point-point distance if projects outside the segment
   *
   * @param[in] seg_start Start spacepoint defining segment
   * @param[in] seg_end   End spacepoint defining segment
   * @param[in] testpt    Test spacepoint
   */
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

  /**
   * @brief calculate gradient of squared distance with respect to end-point position
   *
   * @param[in] seg_start Segment start point
   * @param[in] seg_end   Segment end point
   * @param[in] testpt    Test point
   *
   */
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

  /**
   * @brief calculate loss and gradient without weighting points
   *
   * @param[in] initial_track Initial track points, consisting of a start and end point
   * @param[in] track_pts_w_feat_v vector of space points (vector<float>) to fit to
   * @param[out] loss The final total loss which is the average squared-distance
   * @param[out] grad The average gradient for all points
   */  
  void TrackOTFit::getLossAndGradient(  const std::vector< std::vector<float> >& initial_track,
                                        const std::vector< std::vector<float> >& track_pts_w_feat_v,
                                        float& loss,
                                        std::vector<float>& grad )
  {
    float weight = 0;
    getWeightedLossAndGradient( initial_track, track_pts_w_feat_v, loss, weight, grad );
  }

  /**
   * @brief calculate loss and gradient with weights
   *
   * For the vectors in track_pts_w_feat_v, the first 3 entries are (x,y,z).
   * The weight for a given spacepoint is calculated by multiplying
   * all vector values for entries [3:] and above.
   *
   * @param[in] initial_track Initial track points, consisting of a start and end point
   * @param[in] track_pts_w_feat_v vector of space points (vector<float>) to fit to
   * @param[out] loss The final total loss which is the average squared-distance
   * @param[out] tot_weight The total weight
   * @param[out] grad The weighted average gradient for all points
   */  
  void TrackOTFit::getWeightedLossAndGradient(  const std::vector< std::vector<float> >& initial_track,
                                                const std::vector< std::vector<float> >& track_pts_w_feat_v,
                                                float& loss,
                                                float& tot_weight,
                                                std::vector<float>& grad )
  {
    grad.resize(3,0);
    for (int i=0; i<3; i++)
      grad[i] = 0.;
    loss = 0.;

    tot_weight = 0.;

    const std::vector<float>& start = initial_track[0];
    const std::vector<float>& end   = initial_track[1];
    
    int ndatapts = track_pts_w_feat_v.size();
    for ( int ipt=0; ipt<ndatapts; ipt++ ) {
      const std::vector<float>& testpt = track_pts_w_feat_v[ipt];

      // calculate the weight
      float w = 1.0;
      for (int i=3;i<(int)testpt.size(); i++)
        w *= testpt[i];
      w = fabs(w);
      
      loss += w*d2_segment_point( start, end, testpt );
      std::vector<float> ptgrad = grad_d2_wrt_segend( start, end, testpt );

      for (int i=0; i<3; i++) {
        grad[i] += ptgrad[i]*w;
      }
      tot_weight += w;
    }

    if ( tot_weight>0 ) {
      loss /= tot_weight;
      for (int i=0; i<3; i++)
        grad[i] /= tot_weight;
    }

  }
  
}
}
