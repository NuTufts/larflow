#include "LArFlowConstants.h"

#include <stdexcept>

namespace larflow {

  const FlowDir_t LArFlowConstants::FlowPlaneMatrix[3][3] = { {kNumFlows, kU2V, kU2Y},
                                                              {kV2U, kNumFlows, kV2Y},
                                                              {kY2U, kY2V, kNumFlows} };
    
  std::string LArFlowConstants::getFlowName( FlowDir_t dir ) {
    
    switch (dir) {
    case kU2V:
      return "u2v";
      break;
    case kU2Y:
      return "u2y";
      break;
    case kV2U:
      return "v2u";
      break;
    case kV2Y:
      return "v2y";
      break;
    case kY2U:
      return "y2u";
      break;
    case kY2V:
      return "y2v";
      break;
    case kNumFlows:
      return "allflows";
      break;
    default:
      throw std::runtime_error("invalid flow direction given");
      break;
    }
    return "nevergethere";
  }

  
  int LArFlowConstants::getOtherPlane( int sourceplane, int targetplane ) {

    switch ( sourceplane ) {
    case 0:
      return ( targetplane==1 ) ? 2 : 1;
      break;
    case 1:
      return ( targetplane==0 ) ? 2 : 0;
      break;
    case 2:
      return ( targetplane==0 ) ? 1 : 0;
      break;
    default:
      char msg[100];
      sprintf( msg, "Unrecognized combination of sourceplane[%d] and targetplane[%d] planes", sourceplane, targetplane );
      throw std::runtime_error( msg );
      break;
    }

    char msg[100];
    sprintf( msg, "Unrecognized combination of sourceplane[%d] and targetplane[%d] planes", sourceplane, targetplane );
    throw std::runtime_error( msg );
    
  }

  FlowDir_t LArFlowConstants::getFlowDirection( int sourceplane, int targetplane ) {

    if ( sourceplane==targetplane ) {
      throw std::runtime_error("[LArFlowConstants.cxx:getFlowDirection] no flow direction when source and target plane the same");
    }
    if ( sourceplane<0 || sourceplane>=3 ) {
      throw std::runtime_error("[LArFlowConstants.cxx:getFlowDirection] invalid source plane index");
    }
    if ( targetplane<0 || targetplane>=3 ) {
      throw std::runtime_error("[LArFlowConstants.cxx:getFlowDirection] invalid target plane index");
    }

    return FlowPlaneMatrix[sourceplane][targetplane];
  }

  void LArFlowConstants::getFlowPlanes( FlowDir_t dir, int& sourceplane, int& targetplane ) {
    switch( dir ) {
    case kU2V:
      sourceplane = 0;
      targetplane = 1;
      break;
    case kU2Y:
      sourceplane = 0;
      targetplane = 2;
      break;
    case kV2U:
      sourceplane = 1;
      targetplane = 0;
      break;
    case kV2Y:
      sourceplane = 1;
      targetplane = 2;
      break;
    case kY2U:
      sourceplane = 2;
      targetplane = 0;
      break;
    case kY2V:
      sourceplane = 2;
      targetplane = 1;
      break;
    default:
      throw std::runtime_error("[LArFlowConstants.cxx:getFlowPlanes] invalid flow direction");
      break;
    }
    
  }
  
}
    
