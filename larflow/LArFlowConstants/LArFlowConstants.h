#ifndef __LARFLOW_BASE_LARFLOW_CONSTANTS_H__
#define __LARFLOW_BASE_LARFLOW_CONSTANTS_H__

#include <string>

namespace larflow {

  typedef enum { kU2V=0, kU2Y, kV2U, kV2Y, kY2U, kY2V, kNumFlows } FlowDir_t;
    
  class LArFlowConstants {

  protected:
    LArFlowConstants() {};
    virtual ~LArFlowConstants() {};
    
  public:
   
    static const FlowDir_t FlowPlaneMatrix[3][3];

    static std::string getFlowName( FlowDir_t dir );  
    static FlowDir_t   getFlowDirection( int sourceplane, int targetplane );
    static int         getOtherPlane( int sourceplane, int targetplane );
    static void        getFlowPlanes( FlowDir_t dir, int& sourceplane, int& targetplane );

  };
  
}

#endif
