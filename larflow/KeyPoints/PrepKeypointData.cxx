#include "PrepKeypointData.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <sstream>
#include <algorithm>
#include <queue>

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/LArProperties.h"
#include "DataFormat/storage_manager.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/mcshower.h"
#include "DataFormat/mctruth.h"

namespace larflow {
namespace keypoints {

  bool compare_x( const bvhnode_t* lhs, const bvhnode_t* rhs ) {
    if ( lhs->bounds[0][0] < rhs->bounds[0][0] )
      return true;
    return false;
  }
  
  bool compare_y( const bvhnode_t* lhs, const bvhnode_t* rhs ) {
    if ( lhs->bounds[1][0] < rhs->bounds[1][0] )
        return true;
    return false;
  }
  
  bool compare_z( const bvhnode_t* lhs, const bvhnode_t* rhs ) {
    if ( lhs->bounds[2][0] < rhs->bounds[2][0] )
        return true;
    return false;
  }

  std::string strnode( const bvhnode_t* node ) {
    std::stringstream ss;
    if ( node->children.size()>0 ) {
      ss << "node: x[" << node->bounds[0][0] << "," << node->bounds[0][1] << "] "
         << "y[" << node->bounds[1][0] << "," << node->bounds[1][1] << "] "
         << "z[" << node->bounds[2][0] << "," << node->bounds[2][1] << "] ";
    }
    else {
      ss << "LEAF: "
         << "x[" << node->bounds[0][0] << "] "
         << "y[" << node->bounds[1][0] << "] " 
         << "z[" << node->bounds[2][0] << "] "
         << "kpdata-index=" << node->kpdidx;
    }
    return ss.str();
  }

  void print_graph( const bvhnode_t* node ) {
    int depth=0;
    _recurse_printgraph( node, depth );
  }
  
  void _recurse_printgraph( const bvhnode_t* node, int& depth ) {
    std::string info =  strnode(node);
    std::string branch = "";
    for (int i=0; i<depth; i++)
      branch += " |";
    if ( depth>0 ) 
      branch += "-- ";

    std::cout << branch << info << std::endl;

    // we loop through our daughters
    for ( auto& child : node->children ) {
      ++depth;
      _recurse_printgraph( child, depth );
    }
    --depth;
  }
  
  bool PrepKeypointData::_setup_numpy = false;

  
  /**
   * process one event, given io managers
   *
   */
  void PrepKeypointData::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    auto ev_adc      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wiremc");
    auto ev_segment  = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");
    auto ev_instance = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"instance");
    auto ev_ancestor = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"ancestor");

    auto ev_mctrack  = (larlite::event_mctrack*)ioll.get_data(  larlite::data::kMCTrack, "mcreco" );
    auto ev_mcshower = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );
    auto ev_mctruth  = (larlite::event_mctruth*)ioll.get_data(  larlite::data::kMCTruth, "generator" );
    
    std::vector<larcv::Image2D> badch_v;
    for ( auto const& img : ev_adc->Image2DArray() ) {
      larcv::Image2D blank(img.meta());
      blank.paint(0.0);
      badch_v.emplace_back( std::move(blank) );
    }

    std::cout << "[PrepKeypointData Inputs]" << std::endl;
    std::cout << "  adc images: "      << ev_adc->Image2DArray().size() << std::endl;
    std::cout << "  badch images: "    << badch_v.size() << std::endl;    
    std::cout << "  segment images: "  << ev_segment->Image2DArray().size() << std::endl;
    std::cout << "  instance images: " << ev_instance->Image2DArray().size() << std::endl;
    std::cout << "  ancestor images: " << ev_ancestor->Image2DArray().size() << std::endl;
    std::cout << "  mctracks: " << ev_mctrack->size() << std::endl;
    std::cout << "  mcshowers: " << ev_mcshower->size() << std::endl;
    std::cout << "  mctruths: " << ev_mctruth->size() << std::endl;
                                                         
    
    process( ev_adc->Image2DArray(),
             badch_v,
             ev_segment->Image2DArray(),
             ev_instance->Image2DArray(),
             ev_ancestor->Image2DArray(),
             *ev_mctrack,
             *ev_mcshower,
             *ev_mctruth );
  }

  /**
   * process one event, directly given input containers
   *
   */
  void PrepKeypointData::process( const std::vector<larcv::Image2D>&    adc_v,
                                  const std::vector<larcv::Image2D>&    badch_v,
                                  const std::vector<larcv::Image2D>&    segment_v,
                                  const std::vector<larcv::Image2D>&    instance_v,
                                  const std::vector<larcv::Image2D>&    ancestor_v,                                  
                                  const larlite::event_mctrack&  mctrack_v,
                                  const larlite::event_mcshower& mcshower_v,
                                  const larlite::event_mctruth&  mctruth_v ) {

    // allocate space charge class
    larutil::SpaceChargeMicroBooNE sce;

    // make particle graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraph( adc_v, segment_v, instance_v, ancestor_v,
                     mcshower_v, mctrack_v, mctruth_v );


    // build key-points
    _kpd_v.clear();
    
    // build crossing points for muon track primaries
    std::vector<KPdata> track_kpd
      = getMuonEndpoints( mcpg, adc_v, mctrack_v, &sce );

    std::cout << "[Track Endpoint Results]" << std::endl;
    for ( auto const& kpd : track_kpd ) {
      std::cout << "  " << str(kpd) << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    // add points for shower starts
    std::vector<KPdata> shower_kpd
      = getShowerStarts( mcpg, adc_v, mcshower_v, &sce );
    std::cout << "[Shower Endpoint Results]" << std::endl;
    for ( auto const& kpd : shower_kpd ) {
      std::cout << "  " << str(kpd) << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );      
    }

    makeBVH();
    printBVH();
  }

  
  std::vector<KPdata>
  PrepKeypointData::getMuonEndpoints( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      const larlite::event_mctrack& mctrack_v,
                                      larutil::SpaceChargeMicroBooNE* psce )
  {

    bool verbose = false;
    
    // get list of primaries
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> primaries
      = mcpg.getPrimaryParticles();

    // output vector of keypoint data
    std::vector<KPdata> kpd_v;

    for ( auto const& pnode : primaries ) {

      if ( abs(pnode->pid)!=13
           && abs(pnode->pid)!=2212
           && abs(pnode->pid)!=211 )
        continue;

      auto const& mctrk = mctrack_v.at( pnode->vidx );

      int crossingtype =
        ublarcvapp::mctools::CrossingPointsAnaMethods::
        doesTrackCrossImageBoundary( mctrk,
                                     adc_v.front().meta(),
                                     4050.0,
                                     psce );

      if ( crossingtype>=0 ) {
        
        if ( crossingtype>=0 && crossingtype<=2) {
          KPdata kpd;
          kpd.crossingtype = crossingtype;
          kpd.trackid = pnode->tid;
          kpd.pid     = pnode->pid;
          kpd.vid     = pnode->vidx;
          kpd.is_shower = 0;
          kpd.origin  = pnode->origin;
          kpd.imgcoord = 
            ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                       4050.0, true, 0.3, 0.1,
                                                                                       kpd.keypt, psce, verbose );
          if ( kpd.imgcoord.size()>0 ) {
            kpd_v.emplace_back( std::move(kpd) );            
          }
          
        }

        if ( crossingtype>=0 && crossingtype<=2 ) {
          KPdata kpd;
          kpd.crossingtype = crossingtype;
          kpd.trackid = pnode->tid;
          kpd.pid     = pnode->pid;
          kpd.vid     = pnode->vidx;
          kpd.is_shower = 0;
          kpd.origin  = pnode->origin;
          kpd.imgcoord = 
            ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                       4050.0, false, 0.3, 0.1,
                                                                                       kpd.keypt, psce, verbose );
          if ( kpd.imgcoord.size()>0 ) {
            kpd_v.emplace_back( std::move(kpd) );
          }
        }


      }//if track in image

    }//end of primary loop

    return kpd_v;
  }

  std::vector<KPdata>
  PrepKeypointData::getShowerStarts( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                     const std::vector<larcv::Image2D>& adc_v,
                                     const larlite::event_mcshower& mcshower_v,
                                     larutil::SpaceChargeMicroBooNE* psce )
  {

    // get list of primaries
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> primaries
      = mcpg.getPrimaryParticles();

    // output vector of keypoint data
    std::vector<KPdata> kpd_v;

    for ( auto const& pnode : primaries ) {

      if ( abs(pnode->pid)!=11
           && abs(pnode->pid)!=22 )
        continue;

      auto const& mcshr = mcshower_v.at( pnode->vidx );

      //we convert shower to track trajectory
      larlite::mctrack mct;
      mct.push_back( mcshr.Start() );
      mct.push_back( mcshr.End() );

      int crossingtype =
        ublarcvapp::mctools::CrossingPointsAnaMethods::
        doesTrackCrossImageBoundary( mct,
                                     adc_v.front().meta(),
                                     4050.0,
                                     psce );

      if ( crossingtype>=0 ) {

        KPdata kpd;
        kpd.crossingtype = crossingtype;
        kpd.trackid = pnode->tid;
        kpd.pid     = pnode->pid;
        kpd.vid     = pnode->vidx;
        kpd.origin  = pnode->origin;
        kpd.is_shower = 1;
        
        kpd.imgcoord = 
          ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mct, adc_v.front().meta(),
                                                                                     4050.0, true, 0.3, 0.1,
                                                                                     kpd.keypt, psce, false );
        if ( kpd.imgcoord.size()>0 ) {
          kpd_v.emplace_back( std::move(kpd) );
        }
      }

    }//end of primary loop
    
    return kpd_v;
  }

  std::string PrepKeypointData::str( const KPdata& kpd )
  {
    std::stringstream ss;
    ss << "[type=" << kpd.crossingtype << " pid=" << kpd.pid
       << " vid=" << kpd.vid
       << " isshower=" << kpd.is_shower
       << " origin=" << kpd.origin << "] ";

    if ( kpd.imgcoord.size()>0 )
      ss << " imgstart=(" << kpd.imgcoord[0] << ","
         << kpd.imgcoord[1] << ","
         << kpd.imgcoord[2] << ","
         << kpd.imgcoord[3] << ") ";

    if ( kpd.keypt.size()>0 )
      ss << " keypt=(" << kpd.keypt[0] << "," << kpd.keypt[1] << "," << kpd.keypt[2] << ") ";
            
    return ss.str();
  }


  /**
   * return an array with keypoints
   * array columns [tick,wire-U,wire-V,wire-Y,x,y,z,isshower,origin,pid]
   *
   */
  PyObject* PrepKeypointData::get_keypoint_array() const
  {

    if ( !PrepKeypointData::_setup_numpy ) {
      import_array1(0);
      PrepKeypointData::_setup_numpy = true;
    }
    
    // first count the number of unique points
    std::set< std::vector<int> >    unique_coords;
    std::vector< std::vector<int> > kpd_index;
    int npts = 0;
    for ( size_t ikpd=0; ikpd<_kpd_v.size(); ikpd++ ) {
      auto const& kpd = _kpd_v[ikpd];

      if (kpd.imgcoord.size()>0) {
        if ( unique_coords.find( kpd.imgcoord )==unique_coords.end() ) {
          kpd_index.push_back( std::vector<int>{(int)ikpd,0} );
          unique_coords.insert( kpd.imgcoord );
          npts++;
        }
      }      
    }
    
    int nd = 2;
    npy_intp dims[] = { npts, 10 };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

    size_t ipt = 0;
    for ( auto& kpdidx : kpd_index ) {
      
      auto const& kpd = _kpd_v[kpdidx[0]];

      if ( kpdidx[1]==0 ) {
        // start img coordinates
        for ( size_t i=0; i<4; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,i)) = (float)kpd.imgcoord[i];
        // 3D point
        for ( size_t i=0; i<3; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,4+i)) = (float)kpd.keypt[i];
        // is shower
        *((float*)PyArray_GETPTR2(array,ipt,7)) = (float)kpd.is_shower;
        // origin
        *((float*)PyArray_GETPTR2(array,ipt,8)) = (float)kpd.origin;
        // PID
        *((float*)PyArray_GETPTR2(array,ipt,9)) = (float)kpd.pid;
        ipt++;
      }
    }// end of loop over keypointdata structs

    return (PyObject*)array;
  }

  /**
   * given a set of match proposals, we make labels for each
   * label columns:
   *  [0]:   has true end-point with X cm
   *  [1-3]: shift in 3D points from point to closest end-point
   *  [4-9]: shift in 2D pixels from image points to closest end-point
   */
  void PrepKeypointData::make_proposal_labels( const larflow::PrepMatchTriplets& match_proposals )
  {

    _match_proposal_labels_v.resize(match_proposals._triplet_v.size());

    for (int imatch=0; imatch<match_proposals._triplet_v.size(); imatch++ ) {
      const std::vector<int>& triplet = match_proposals._triplet_v[imatch]; 
      const std::vector<float>& pos   = match_proposals._pos_v[imatch];

      // dumb assignment, loops over all truth keypoints
      // smarter one uses a BVH structure (does being in a leaf volume guarentee that point is the closest?)
      // O(200k) proposals x 40 keypoints
      // brute forces is O(8M) while BVH is O(737K), a factor of 10 speed-up
    }
      
  }

  /**
   * we build a boundary volume hierarchy tree, using a top-down method
   *
   */
  void PrepKeypointData::makeBVH() {

    std::cout << "=========================" << std::endl;
    std::cout << " makeBVH" << std::endl;
    std::cout << "=========================" << std::endl;    
    
    clearBVH();
    
    // xlimits derived from bound of image
    float xmin_det = (2400-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
    float xmax_det = (2400+1008*6-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();

    // make the root node    
    _bvhroot = new bvhnode_t(xmin_det,xmax_det,-118,118,0,1036);

    // make a node for all points
    std::vector< bvhnode_t* > leafs;
    int ikpd = 0;
    for ( auto const& kpd : _kpd_v ) {
      bvhnode_t* node = new bvhnode_t( kpd.keypt[0], kpd.keypt[0],
                                       kpd.keypt[1], kpd.keypt[1],
                                       kpd.keypt[2], kpd.keypt[2] );
      node->kpdidx = ikpd;
      ikpd++;
      leafs.push_back( node );
      node->mother = _bvhroot;
      _bvhroot->children.push_back( node );
    }

    std::queue< bvhnode_t* > q;
    q.push( _bvhroot );

    while ( q.size()>0 ) {
      std::cout << "process node on the queue" << std::endl;
      bvhnode_t* node = q.front();
      q.pop();

      // get longest dim of box
      float dimlen = 0;
      int longdim = 0;
      for (int i=0; i<3; i++ ) {
        float len = node->bounds[i][1]-node->bounds[i][0];
        if (len>dimlen) {
          longdim = i;
          dimlen  = len;
        }
      }
      // set this longest dim as the splitting dimension of this node
      node->splitdim = longdim;

      // sort children by the dim
      if      ( longdim==0 ) std::sort( node->children.begin(), node->children.end(), compare_x );
      else if ( longdim==1 ) std::sort( node->children.begin(), node->children.end(), compare_y );
      else if ( longdim==2 ) std::sort( node->children.begin(), node->children.end(), compare_z );

      // get the center point to make a box
      int lowidx  = 0;
      int highidx = 0;
      if ( node->children.size()%2==0 ) {
        // even
        highidx = node->children.size()/2; // 4/2 = 2
        lowidx = highidx-1;                // 1
      }
      else {
        // odd
        lowidx = node->children.size()/2; // 3/2 = 1
        highidx = lowidx + 1;             // 2
      }

      // calculate midpoint using leaf nodes
      float midpt = 0.5*(node->children[lowidx]->bounds[longdim][0] + node->children[highidx]->bounds[longdim][0] );

      // we define two new boundary volume, using the midpoint of low and high
      // to split the previous volume.
      // we split the children into them.
      // if the node only has one, then its a leaf node
      float lo_bounds[3][2];
      float hi_bounds[3][2];
      for (int i=0; i<3; i++ ) {
        for (int j=0; j<2; j++ ) {
          lo_bounds[i][j] = node->bounds[i][j];
          hi_bounds[i][j] = node->bounds[i][j];
        }
      }
      lo_bounds[longdim][1] = midpt;
      hi_bounds[longdim][0] = midpt;
      bvhnode_t* lo_node = new bvhnode_t( lo_bounds[0][0], lo_bounds[0][1],
                                          lo_bounds[1][0], lo_bounds[1][1],
                                          lo_bounds[2][0], lo_bounds[2][1] );
      bvhnode_t* hi_node = new bvhnode_t( hi_bounds[0][0], hi_bounds[0][1],
                                          hi_bounds[1][0], hi_bounds[1][1],
                                          hi_bounds[2][0], hi_bounds[2][1] );

      // now we divide the node's children into the new nodes
      for (int i=0; i<=lowidx; i++ ) {
        node->children[i]->mother = lo_node;
        lo_node->children.push_back( node->children[i] );
      }
      for (int i=highidx; i<node->children.size(); i++ ) {
        node->children[i]->mother = hi_node;        
        hi_node->children.push_back( node->children[i] );
      }

      // clear the node's children and replace it with the new nodes
      node->children.clear();
      node->children.push_back( lo_node );
      node->children.push_back( hi_node );


      //set them into the queue if they have more than one child
      if (lo_node->children.size()>1 )
        q.push( lo_node );
      if (hi_node->children.size()>1 )
        q.push( hi_node );

      std::cout << "finished processing node. left in queue: " << q.size() << std::endl;
    }//end of loop over queue
    
  }

  void PrepKeypointData::clearBVH() {
    for ( auto& pnode : _bvhnodes_v )
      delete pnode;
    _bvhnodes_v.clear();
    if ( _bvhroot )
      delete _bvhroot;
    _bvhroot = nullptr;
  }

  void PrepKeypointData::printBVH() {
    print_graph( _bvhroot );
  }
  
}
}
