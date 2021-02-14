#ifndef __LARFLOW_BASE_LARFLOW_CONSTANTS_H__
#define __LARFLOW_BASE_LARFLOW_CONSTANTS_H__


#include <string>

namespace larflow {

  /**
   * Symbol for Plane Flows
   */
  typedef enum { kU2V=0, kU2Y, kV2U, kV2Y, kY2U, kY2V, kNumFlows }   FlowDir_t;

  /**
   * Type of Keypoint Labels
   */
  typedef enum { kNuVertex=0, kTrackEnds, kShowerStart, kVertexActivity, kNumKeyPoints } KeyPoint_t;

  /**
   * @ingroup LArFlowConstants
   * @class LArFlowConstants
   * @brief Constants and utility functions
   *
   * @author Taritree Wongjirad (taritree.wongjirad@tufts.edu)
   * @date $Date: 2020/07/2 16:35 $
   * 
   * Contact: taritree.wongjirad@tufts.edu
   *
   * Revision History:
   * 2020/07/22: writing documentation
   * 
   */  
  class LArFlowConstants {

  protected:
    LArFlowConstants() {};
    virtual ~LArFlowConstants() {};
    
  public:
   
    static const FlowDir_t FlowPlaneMatrix[3][3];  ///< Map from plane ID numbers to FlowDir labels

    static std::string getFlowName( FlowDir_t dir );  ///< Get string with name of flow direction
    static FlowDir_t   getFlowDirection( int sourceplane, int targetplane ); ///< get FlowDir for pair of planes
    static int         getOtherPlane( int sourceplane, int targetplane );    ///< get the other plane index given the source and target plane
    static void        getFlowPlanes( FlowDir_t dir, int& sourceplane, int& targetplane ); ///< Given FlowDir_t, get source and target plane indices

  };
  
}

#endif
