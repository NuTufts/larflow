#include "ProjPixFitter.h"

#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"


namespace larflow {
namespace reco {

  /**
   * @brief calculate pixel distance from projected point in plane and grad w.r.t. 3d pos
   *
   * @param[in] pos Test Position in the detector, assumed to be NOT t0-corrected positions (so can be outside the detector)
   * @param[in] tick Tick of pixel in the image (we are fitting to)
   * @param[in] wire Wire of pixel in the image (we are fitting to)
   * @param[in] plane Wire plane the pixel is in
   * @param[out] d2 The distance squared in cm in image distance
   * @param[out] grad The gradient w.r.t. the test position
   */  
  void ProjPixFitter::grad_and_d2_pixel( const std::vector<float>& pos,
                                         const float tick,
                                         const float wire,
                                         const int plane,
                                         float& d2,
                                         std::vector<float>& grad )
  {

    const std::vector<Double_t> orthy = larutil::Geometry::GetME()->GetOrthVectorsY();
    const std::vector<Double_t> orthz = larutil::Geometry::GetME()->GetOrthVectorsZ();
    const std::vector<Double_t> firstwire = larutil::Geometry::GetME()->GetFirstWireProj();
    const double driftv = larutil::LArProperties::GetME()->DriftVelocity();

    float proj_wirecoord = pos[1]*orthy[plane]
      + pos[2]*orthz[plane]
      - firstwire[plane];

    float pix_x = (tick-3200)*0.5*driftv;


    float dwire = 0.3*(wire-proj_wirecoord);
    float dtick = pix_x-pos[0];

    d2 = dwire*dwire + dtick*dtick;    
    
    grad.resize(3,0);
    grad[0] = -2*dtick;
    grad[1] = -2*dwire*0.3*orthy[plane];
    grad[2] = -2*dwire*0.3*orthz[plane];

  }


}
}
    
