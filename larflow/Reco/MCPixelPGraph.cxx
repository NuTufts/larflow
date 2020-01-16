#include "MCPixelPGraph.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"

namespace larflow {
namespace reco {

  void MCPixelPGraph::buildgraph( larcv::IOManager& iolcv,
                                  larlite::storage_manager& ioll ) {
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    larcv::EventImage2D* ev_seg = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "segment" );
    larcv::EventImage2D* ev_ins = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "instance" );
    larcv::EventImage2D* ev_anc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "ancestor" );

    larlite::event_mctrack*  ev_mctrack  = (larlite::event_mctrack*) ioll.get_data( larlite::data::kMCTrack,  "mcreco" );
    larlite::event_mcshower* ev_mcshower = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );

    buildgraph( ev_adc->Image2DArray(),
                ev_seg->Image2DArray(),
                ev_ins->Image2DArray(),
                ev_anc->Image2DArray(),
                *ev_mcshower, *ev_mctrack );
  }
  
  void MCPixelPGraph::buildgraph( const std::vector<larcv::Image2D>& adc_v,
                                  const std::vector<larcv::Image2D>& segment_v,
                                  const std::vector<larcv::Image2D>& instance_v,
                                  const std::vector<larcv::Image2D>& ancestor_v,
                                  const larlite::event_mcshower& shower_v,
                                  const larlite::event_mctrack&  track_v ) {

    // how do we build this graph?
    // we want to order N

    // (0) create root node
    // (1) loop through track and shower, creating node objects
    // (2) loop through node objects connecting daughters to mothers
    // (3) (optional) get depth of each node by doing breath-first traversal
    // (4) sort vector pointers by depth (necessary?)

    node_v.clear();
    node_v.reserve( shower_v.size()+track_v.size() );

    // Create ROOT node
    Node_t neutrino ( node_v.size(), -1, 0, 0, -1 );
    node_v.emplace_back( std::move(neutrino) );

    for (int vidx=0; vidx<(int)track_v.size(); vidx++ ) {
      const larlite::mctrack& mct = track_v[vidx];
      // std::cout << "track[" << vidx << "] origin=" << mct.Origin()
      //           << " tid=" << mct.TrackID()
      //           << " mid=" << mct.MotherTrackID()
      //           << " aid=" << mct.AncestorTrackID()
      //           << " pid=" << mct.PdgCode()
      //           << std::endl;

      // toss out neutrons? (sigh...)
      
      if ( mct.Origin()==1 ) {
        // neutrino origin
        Node_t tracknode( node_v.size(), 0, mct.TrackID(), vidx, mct.PdgCode() );
        node_v.emplace_back( std::move(tracknode) );
      }
    }

    for (int vidx=0; vidx<(int)shower_v.size(); vidx++ ) {
      const larlite::mcshower& mcsh = shower_v[vidx];

      // std::cout << "shower[" << vidx << "] origin=" << mcsh.Origin()
      //           << " tid=" << mcsh.TrackID()
      //           << " mid=" << mcsh.MotherTrackID()
      //           << " aid=" << mcsh.AncestorTrackID()
      //           << " pid=" << mcsh.PdgCode()
      //           << std::endl;
      
      if ( mcsh.Origin()==1 ) {
        // neutrino origin
        Node_t showernode( node_v.size(), 1, mcsh.TrackID(), vidx, mcsh.PdgCode() );
        node_v.emplace_back( std::move(showernode) );
      }
    }

    // sort Node_t object by geant track ID, relabel node IDs
    std::sort( node_v.begin(), node_v.end() );
    for ( size_t nid=0; nid<node_v.size(); nid++ ) {
      node_v[nid].nodeidx = nid;
    }

    // now connect mothers and daughters. we use the mother and ancestor ID to do this.
    for ( auto& node : node_v ) {
      if ( node.tid<0 ) continue; // the root node

      // find the mother node
      Node_t* mothernode = nullptr;
      
      if ( node.type==0 ) {
        // track nodes
        const larlite::mctrack& track = track_v[ node.vidx ];
        if ( track.TrackID()==track.MotherTrackID() ) {
          // primary
          mothernode = &node_v[0];
        }
        else {
          // secondary
          mothernode = findTrackID( track.MotherTrackID() );
          if( mothernode==nullptr ) {
            // try ancestor ID
            mothernode = findTrackID( track.AncestorTrackID() );
          }
        }
      }
      else if (node.type==1) {
        //shower nodes
        const larlite::mcshower& shower = shower_v[node.vidx];
        if (shower.TrackID()==shower.MotherTrackID() ) {
          //primary
          mothernode = &node_v[0];
        }
        else {
          //secondary
          mothernode = findTrackID( shower.MotherTrackID() );
          if( mothernode==nullptr ) {
            // try ancestor ID
            mothernode = findTrackID( shower.AncestorTrackID() );
          }
        }
      }

      if (mothernode) {
        // found mother, connect
        //std::cout << "found mother: " << strNodeInfo(*mothernode) << std::endl;
        node.mother = mothernode;
        node.mid    = mothernode->nodeidx;
        mothernode->daughter_v.push_back( &node );
        mothernode->daughter_idx_v.push_back( node.nodeidx );
      }
      
    }//end of node loop

    //printAllNodeInfo();
    //printGraph();
  }

  MCPixelPGraph::Node_t* MCPixelPGraph::findTrackID( int trackid ) {
    Node_t dummy;
    dummy.tid = trackid;
    auto it = std::lower_bound( node_v.begin(), node_v.end(), dummy );
    if ( it==node_v.end() ) {
      
      return nullptr;
    }
    //std::cout << "find trackid=" << trackid << ": " << strNodeInfo( *it ) << std::endl;    
    return &*(it+0);
  }
  
  void MCPixelPGraph::printAllNodeInfo()  {
    for ( auto const& node : node_v ) {
      printNodeInfo(node);
    }
  }

  std::string MCPixelPGraph::strNodeInfo( const Node_t& node ) {
    std::stringstream ss;
    ss << "node[" << node.nodeidx << "," << &node << "] "
       << " (type,vidx)=(" << node.type << "," << node.vidx << ") "
       << " trackid=" << node.tid
       << " pdg=" << node.pid
       << " (mid,mother)=(" << node.mid << "," << node.mother << ") "
       << " ndaughters=" << node.daughter_v.size()
       << " npixels=(";
    for ( size_t i=0; i<node.pix_vv.size(); i++ ) {
      ss << node.pix_vv[i].size();
      if ( i+1<node.pix_vv.size() ) ss << ", ";
    }
    ss << ")";

    return ss.str();
  }
  
  void MCPixelPGraph::printNodeInfo( const Node_t& node ) {
    std::cout << strNodeInfo(node) << std::endl;
  }

  void MCPixelPGraph::printGraph() {
    //  here we go!
    std::cout << "=======[ MCPixelPGraph::printGraph ]==============" << std::endl;
    int depth = 0;
    _recursivePrintGraph( &node_v.front(), depth );
  }
  
  void MCPixelPGraph::_recursivePrintGraph( Node_t* node, int& depth ) {
    if ( depth<0 ) return; // we're done (error?)
    
    // depth first printing of nodes   
    std::string info = strNodeInfo( *node );
    std::string branch = "";
    for ( int i=0; i<depth; i++ )
      branch += " |";
    if ( depth>0 ) 
      branch += "-- ";
    std::cout << branch << info << std::endl;

    // we loop through our daughters
    for ( auto& daughter : node->daughter_v ) {
      ++depth;
      _recursivePrintGraph( daughter, depth );
    }
    --depth;
    return;
  }
      
}
}
