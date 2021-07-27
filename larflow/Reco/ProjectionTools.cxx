#include "ProjectionTools.h"

namespace larflow {
namespace reco {

  float ProjectionTools::wirecoordinate_and_grad( const std::vector<float>& pos,
                                                  const int plane,
                                                  std::vector<float>& grad )
  {

    if ( plane<0 || plane>=(int)larutil::Geometry::GetME()->Nplanes() )
      throw std::runtime_error("ProjectionTools::wirecoordinate_and_grad invalid plane number");
    
    // microboone only
    double firstwireproj = larutil::Geometry::GetME()->GetFirstWireProj(); 
    std::vector<double> orthovect(3) = { 0,
                                         larutil::Geometry::GetME()->GetOrthVectorsY().at(plane),
                                         larutil::Geometry::GetME()->GetOrthVectorsZ().at(plane) };

    // from larlite Geometry::WireCoordinate(...)
    float wirecoord = pos[1]*orthovect[1] + pos[2]*orthovect[2] - firstwireproj;

    grad.resize(3,0);
    grad[0] = 0.0;
    grad[1] = orthovect[1];
    grad[2] = orthovect[2];
    return wirecoord;
  }
  
}
}
