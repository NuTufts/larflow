#ifndef __MCPIXELPGRAPH_H__
#define __MCPIXELPGRAPH_H__

#include <vector>
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctrack.h"

/**
 * Determine particle graph. Collect pixels for each particle.
 * In particular, handle the showers.
 *
 */

namespace larflow {
namespace reco {

  class MCPixelPGraph {
  public:

    MCPixelPGraph() {};
    virtual ~MCPixelPGraph() {};

    void buildgraph( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    
    void buildgraph( const std::vector<larcv::Image2D>& adc_v,
                     const std::vector<larcv::Image2D>& segment_v,
                     const std::vector<larcv::Image2D>& instance_v,
                     const std::vector<larcv::Image2D>& ancestor_v,
                     const larlite::event_mcshower& shower_v,
                     const larlite::event_mctrack&  track_v );


    struct Node_t {
      int nodeidx;    // position in node_v
      int type;       // track=0, shower=1
      int vidx;       // position in mcshower or mctrack vector
      int tid;        // geant4 track-ID
      int pid;        // particle ID
      Node_t* mother; // pointer to Mother Node_t
      int mid;        // mother nodeidx      
      std::vector<int>      daughter_idx_v; // daughter node indices in node_v
      std::vector<Node_t*>  daughter_v;     // pointer to daughters 
      std::vector< std::vector<int> > pix_vv; // pixels in each plane. pixels stored in (row,col)

      Node_t()
      : nodeidx(-1),
        type(-1),
        tid(-1),
        vidx(-1),
        pid(-1),
        mother(nullptr),
        mid(-1)
      {};
        
      Node_t(int _nodeidx, int _type, int _tid, int _vidx, int _pid, Node_t* _mother=nullptr, int _mid=-1 )
      : nodeidx(_nodeidx),
        type(_type),
        tid(_tid),
        vidx(_vidx),
        pid(_pid),
        mother(_mother),
        mid(_mid) {};

      bool operator<( const Node_t& rhs ) const {
        if ( tid < rhs.tid ) return true;
        return false;
      };
    };


    std::vector< Node_t > node_v;

    Node_t* findTrackID( int trackid );    
    void printAllNodeInfo();
    void printNodeInfo( const Node_t& node );
    std::string strNodeInfo( const Node_t& node );
    void _recursivePrintGraph( Node_t* node, int& depth );
    void printGraph();
    
  };
  
}
}

#endif
