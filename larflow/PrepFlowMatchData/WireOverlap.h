#ifndef __LARFLOW_WIREOVERLAP_H__
#define __LARFLOW_WIREOVERLAP_H__


#include <vector>
#include <map>

namespace larflow {
namespace prep { 

  /**
   *
   * @class WireOverlap
   *
   * @ingroup PrepFlowMatchData
   *
   * @brief calculates and stores possible wire intersections
   *
   * @author Taritree Wongjirad
   *
   * @date $Date: 2020/07/22 16:53 $
   * 
   * contact: taritree.wongjirad@tufts.edu
   *
   */  
  class WireOverlap {

  protected:
    
    WireOverlap();
    virtual ~WireOverlap() {};

  public:

    static std::vector< std::vector<int> > getOverlappingWires( int sourceplane, int targetplane, int source_wire );
    
  protected:
    
    static void _build();
    static bool _isbuilt; ///< flag indicating information in class has been build
    static std::vector< std::vector<int> >   _wire_targetoverlap[6];  ///< [6] refers to flow direction, outer vector is source plane, inner vector is list of target wire numbers
    static std::vector< std::vector<int> >   _wire_otheroverlap[6];   ///< [6] refers to flow direction, outer vector is source plane, outer vector is list of target wire numbers
    static std::map< std::pair<int,int>, int > _planeflow2mapindex;   ///< map from (source,target) plane to flow index for _wire_targetoverlap[X] and _wire_otheroverlap[X]
    
  };

}
}

#endif
