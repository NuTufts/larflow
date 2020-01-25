#ifndef __LARFLOW_WIREOVERLAP_H__
#define __LARFLOW_WIREOVERLAP_H__

#include <vector>
#include <map>

namespace larflow {
  
  class WireOverlap {

  protected:
    WireOverlap();
    virtual ~WireOverlap() {};

  public:

    static std::vector< std::vector<int> > getOverlappingWires( int sourceplane, int targetplane, int source_wire );
    
  protected:
    
    static void _build();
    static bool _isbuilt;
    static std::vector< std::vector<int> >   _wire_targetoverlap[6];
    static std::vector< std::vector<int> >   _wire_otheroverlap[6];
    static std::map< std::pair<int,int>, int > _planeflow2mapindex;
    
  };

}

#endif
