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
                     const larlite::event_mctrack&  track_v,
                     const larlite::event_mctruth&  mctruth_v );


    struct Node_t {
      int nodeidx;    // position in node_v
      int type;       // track=0, shower=1
      int vidx;       // position in mcshower or mctrack vector
      int tid;        // geant4 track-ID
      int pid;        // particle ID
      Node_t* mother; // pointer to Mother Node_t
      int mid;        // mother nodeidx
      float E_MeV;    // energy
      std::vector<int>      daughter_idx_v; // daughter node indices in node_v
      std::vector<Node_t*>  daughter_v;     // pointer to daughters 
      std::vector< std::vector<int> > pix_vv; // pixels in each plane. pixels stored in (row,col)
      std::vector<float> start;

      Node_t()
      : nodeidx(-1),
        type(-1),
        tid(-1),
        vidx(-1),
        pid(-1),
        mother(nullptr),
        mid(-1),
        E_MeV(-1.0),
        start({0,0,0})
      {};
        
      Node_t(int _nodeidx, int _type, int _tid, int _vidx, int _pid, Node_t* _mother=nullptr, int _mid=-1, float _energy=-1.0 )
      : nodeidx(_nodeidx),
        type(_type),
        tid(_tid),
        vidx(_vidx),
        pid(_pid),
        mother(_mother),
        mid(_mid),
        E_MeV(_energy),
        start({0,0,0})
      {};

      bool operator<( const Node_t& rhs ) const {
        if ( tid < rhs.tid ) return true;
        return false;
      };
    };

    // list of nodes
    std::vector< Node_t > node_v;
    std::vector< std::vector<int> > _unassigned_pixels_vv;

    // number of planes
    size_t _nplanes; // set when _scanPixelData is run

    // search methods
    Node_t* findTrackID( int trackid );

    // print info methods
    void printAllNodeInfo();
    void printNodeInfo( const Node_t& node );
    std::string strNodeInfo( const Node_t& node );
    void printGraph( Node_t* start_node=nullptr );

    // get pixels
    std::vector< std::vector<int> > getPixelsFromParticleAndDaughters( int trackid );

    // graph traversal
    std::vector<Node_t*> getNodeAndDescendentsFromTrackID( const int& trackid );
    void recursiveGetNodeAndDescendents( Node_t* node, std::vector<Node_t*>& nodelist );

    // get primary list
    std::vector<Node_t*> getPrimaryParticles( bool exclude_neutrons=true );

  protected:
    
    void _recursivePrintGraph( Node_t* node, int& depth );
    void _scanPixelData( const std::vector<larcv::Image2D>& adc_v,
                         const std::vector<larcv::Image2D>& segment_v,
                         const std::vector<larcv::Image2D>& instance_v,
                         const std::vector<larcv::Image2D>& ancestor_v,
                         const std::vector<float> threshold_v );
    
  };
  
}
}

#endif