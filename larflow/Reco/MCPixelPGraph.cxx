#include "MCPixelPGraph.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/DataFormatTypes.h"
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

      std::cout << "shower[" << vidx << "] origin=" << mcsh.Origin()
                << " tid=" << mcsh.TrackID()
                << " mid=" << mcsh.MotherTrackID()
                << " aid=" << mcsh.AncestorTrackID()
                << " pid=" << mcsh.PdgCode()
                << std::endl;
      
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

    std::vector<float> threshold_v(adc_v.size(),10.0);
    _scanPixelData( adc_v, segment_v, instance_v, ancestor_v, threshold_v );
    
    //printAllNodeInfo();
    //printGraph();
  }

  /**
   * locate Node_t in node_v using trackid (from geant4)
   *
   * @return The node if found, nullptr if not found
   *
   */
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

  /**
   * print node info for all nodes stored in node_v
   *
   */
  void MCPixelPGraph::printAllNodeInfo()  {
    for ( auto const& node : node_v ) {
      printNodeInfo(node);
    }
  }

  /**
   * create string with info from a given Node_t
   *
   * @param[in] node Note_t object to make info for.
   *
   */
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
      ss << node.pix_vv[i].size()/2;
      if ( i+1<node.pix_vv.size() ) ss << ", ";
    }
    ss << ")";

    return ss.str();
  }

  /**
   * print Node_t info to standard out
   *
   * @param[in] node Node_t object to print info for.
   *
   */
  void MCPixelPGraph::printNodeInfo( const Node_t& node ) {
    std::cout << strNodeInfo(node) << std::endl;
  }

  /**
   * dump graph to standard out
   *
   */
  void MCPixelPGraph::printGraph( Node_t* rootnode ) {
    //  here we go!
    std::cout << "=======[ MCPixelPGraph::printGraph ]==============" << std::endl;
    int depth = 0;
    if (rootnode==nullptr )
      rootnode = &node_v.front();
    _recursivePrintGraph( rootnode, depth );
  }

  /*
   * internal recursive function that prints node info
   *
   */
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

  /**
   * internal method to scan the adc and truth images and fill pixel locations in the Node_t objects
   *
   */
  void MCPixelPGraph::_scanPixelData( const std::vector<larcv::Image2D>& adc_v,
                                      const std::vector<larcv::Image2D>& segment_v,
                                      const std::vector<larcv::Image2D>& instance_v,
                                      const std::vector<larcv::Image2D>& ancestor_v,
                                      const std::vector<float> threshold_v ) {

    _nplanes = adc_v.size();

    // need to check that images have same meta (they should!)
    
    // loop through nodes and setup pixel arrays    
    for (auto& node : node_v ) {
      node.pix_vv.resize(_nplanes);
      for ( size_t p=0; p<_nplanes; p++ ) {
        node.pix_vv[p].clear();
      }
    }

    // track how efficient we were in assigning owner to pixel
    std::vector<int> nabove_thresh(_nplanes,0); // all pixels (some will be cosmic with no truth of course)
    std::vector<int> nabove_thresh_withlabel(_nplanes,0); // pixels with labels
    std::vector<int> nassigned(_nplanes,0);               // assignable to node in node_v
    _unassigned_pixels_vv.clear();
    _unassigned_pixels_vv.resize(_nplanes);

    std::set<int> shower_ancestor_ids;
    
    // loop through images, store pixels into graph nodes
    for ( size_t p=0; p<_nplanes; p++ ) {

      _unassigned_pixels_vv[p].clear();

      auto const& meta = adc_v[p].meta();
      const float threshold = threshold_v[p];
      
      for (size_t r=0; r<meta.rows(); r++) {
        int tick = (int)meta.pos_y(r);
        for (size_t c=0; c<meta.cols(); c++ ) {
          int wire = (int)meta.pos_x(c);

          float adc = adc_v[p].pixel(r,c);
          if ( adc<threshold )
            continue;

          nabove_thresh[p]++;
          
          // above threshold, now lets find instance or ancestor
          int tid = instance_v[p].pixel(r,c);
          int aid = ancestor_v[p].pixel(r,c);
          int seg = segment_v[p].pixel(r,c);

          if ( tid>0 || aid>0 )
            nabove_thresh_withlabel[p]++;

          if ( seg==(int)larcv::kROIEminus || seg==(int)larcv::kROIGamma ) {
            shower_ancestor_ids.insert( aid );
          }

          Node_t* node = nullptr;

          if ( tid>0 ) {
            // first we use the instance ID          
            node = findTrackID( tid );
          }

          // use ancestor if we could not find the node
          if ( !node && aid>0) {
            node = findTrackID( aid );
          }

          if ( node ) {
            nassigned[p]++;
            node->pix_vv[p].push_back( tick );
            node->pix_vv[p].push_back( wire );
          }
          else {
            _unassigned_pixels_vv[p].push_back( tick );
            _unassigned_pixels_vv[p].push_back( wire );
          }
          
        }//end of loop over columns
      }//end of loop over rows
    }//end of loop over planes

    std::cout << "[MCPixelPGraph::_scanPixelData]" << std::endl;
    for (size_t p=0; p<_nplanes; p++ ) {
      std::cout << " plane[" << p << "]"
                << " num above threshold=" << nabove_thresh[p]
                << " and with label=" << nabove_thresh_withlabel[p]
                << " num assigned=" << nassigned[p];
      if ( nabove_thresh_withlabel[p]>0 )
        std::cout << " fraction=" << float(nassigned[p])/float(nabove_thresh_withlabel[p]);
      std::cout << std::endl;
    }
    std::cout << "  ancestor list from all shower pixels: [";
    for ( auto& aid : shower_ancestor_ids )
      std::cout << aid << " ";
    std::cout << "]" << std::endl;
    
  }

  /**
   * get pixels associated with node and its descendents
   * 
   */
  std::vector< std::vector<int> > MCPixelPGraph::getPixelsFromParticleAndDaughters( int trackid ) {
    std::vector< std::vector<int> > pixels_vv(_nplanes);

    std::vector<MCPixelPGraph::Node_t*> nodelist = getNodeAndDescendentsFromTrackID( trackid );
    for ( auto const& pnode : nodelist ) {
      for ( size_t p=0; p<3; p++ ) {
        if ( pnode->pix_vv[p].size()>0 ) {
          pixels_vv[p].insert( pixels_vv[p].end(), pnode->pix_vv[p].begin(), pnode->pix_vv[p].end() );
        }
      }
    }

    return pixels_vv;    
  }

  /**
   * get list of Nodes_t that are decendents of the given trackID
   *
   *
   */
  std::vector<MCPixelPGraph::Node_t*> MCPixelPGraph::getNodeAndDescendentsFromTrackID( const int& trackid ) {
    std::vector<MCPixelPGraph::Node_t*> nodelist;

    Node_t* rootnode = findTrackID( trackid );
    if ( rootnode==nullptr )
      return nodelist;

    nodelist.push_back( rootnode );
    recursiveGetNodeAndDescendents( rootnode, nodelist );
    return nodelist;
  }

  /**
   * recursively get list of Nodes_t that are descendents of the given Node_t*
   *
   * follows depth-first traversal
   */
  void MCPixelPGraph::recursiveGetNodeAndDescendents( Node_t* node, std::vector<Node_t*>& nodelist ) {
    if ( node==nullptr ) return;
    for ( auto& pdaughter : node->daughter_v ) {
      nodelist.push_back( pdaughter );
      recursiveGetNodeAndDescendents( pdaughter, nodelist );
    }
    return;
  }

  /**
   * get list of primary particles
   *
   * by default, neutrons are excluded
   *
   */
  std::vector<MCPixelPGraph::Node_t*> MCPixelPGraph::getPrimaryParticles( bool exclude_neutrons ) {
    std::vector<Node_t*> nodelist;
    Node_t* rootnode = &node_v[0];
    for ( auto& node : node_v ) {
      if ( node.mother==rootnode ) {
        // primary
        if ( !exclude_neutrons || node.pid!=2212 ) {
          nodelist.push_back( &node );
        }
      }
    }
    return nodelist;      
  }
  

}
}
