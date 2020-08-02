#include "ProjPixFitter.h"

#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"


namespace larflow {
namespace reco {

  /**
   * @brief calculate pixel distance from projected point in plane and grad w.r.t. 3d pos
   *
   * @param[in] pt1 3D segment start point
   * @param[in] pt2 3D segment end point. The point we're calculate the gradient for.
   * @param[in] tick Tick of pixel in the image (we are fitting to)
   * @param[in] wire Wire of pixel in the image (we are fitting to)
   * @param[in] plane Wire plane the pixel is in
   * @param[out] d2 The distance squared in cm in image distance
   * @param[out] grad The gradient w.r.t. the test position
   */  
  void ProjPixFitter::grad_and_d2_pixel( const std::vector<float>& pt1, 
                                         const std::vector<float>& pt2, 
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

    float x1 = pt1[1]*orthy[plane]
      + pt1[2]*orthz[plane]
      - firstwire[plane];
    float x2 = pt2[1]*orthy[plane]
      + pt2[2]*orthz[plane]
      - firstwire[plane];

    x1 *= 0.3;
    x2 *= 0.3;
    float x0 = 0.3*wire;

    float y0 = (tick-3200)*0.5*driftv;
    float y1 = pt1[0];
    float y2 = pt2[0];

    float numer = (x2-x1)*(y1-y0) - (x1-x0)*(y2-y1);
    float l2 = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1);
    d2 = numer*numer/l2;
    
    float dLdx2 = (l2*(y1-y0)*2*numer  - (numer*numer)*2*(x2-x1) )/(l2*l2);
    float dLdy2 = (-l2*(x1-x0)*2*numer - (numer*numer)*2*(y2-y1) )/(l2*l2);

    //std::cout << " (dL/dy2) " << dLdy2 << " y0-y2=" << y0-y2 << " y0=" << y0 << " y2=" << y2 << " y2-y1=" << y2-y1 << std::endl;

    float dx2dy = orthy[plane];
    float dx2dz = orthz[plane];
    float dy2dx = 1.0;

    grad.resize(3,0);
    grad[0] = dLdy2*dy2dx;
    grad[1] = dLdx2*dx2dy;
    grad[2] = dLdx2*dx2dz;

  }


}
}
    
