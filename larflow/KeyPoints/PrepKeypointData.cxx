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


  /**
   * static variable to track if numpy environment has been setup
   */
  bool PrepKeypointData::_setup_numpy = false;

  /**
   * constructor
   */
  PrepKeypointData::PrepKeypointData()
    : _adc_image_treename("wire"),
      _bvhroot(nullptr),
      _use_bvh(false),
      _label_tree(nullptr)
  {
    _nclose = 0;
    _nfar   = 0;
    
    hdist[0] = new TH1F("hdist_x","",2002,-500,500.0);
    hdist[1] = new TH1F("hdist_y","",2002,-500,500.0);
    hdist[2] = new TH1F("hdist_z","",2002,-500,500.0);
    
    hdpix[0] = new TH1F("hdpix_dt","",1001,-500,500);
    hdpix[1] = new TH1F("hdpix_du","",1001,-500,500);
    hdpix[2] = new TH1F("hdpix_dv","",1001,-500,500);
    hdpix[3] = new TH1F("hdpix_dy","",1001,-500,500);        
  }

  /**
   * deconstructor
   */
  PrepKeypointData::~PrepKeypointData()
  {
    for (int v=0; v<3; v++ )
      if ( hdist[v] ) delete hdist[v];
    for (int v=0; v<4; v++ )
      if ( hdpix[v] ) delete hdpix[v];
    if ( _label_tree ) delete _label_tree;      
  }
  
  
  /**
   * process one event, given io managers
   *
   * @param[in] iolcv LArCV IOManager containing event data
   * @param[in] ioll  LArLite storage_manager containing event data
   */
  void PrepKeypointData::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {
    auto ev_adc      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,_adc_image_treename);
    auto ev_segment  = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");
    auto ev_instance = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"instance");
    auto ev_ancestor = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"ancestor");

    auto ev_mctrack  = (larlite::event_mctrack*)ioll.get_data(  larlite::data::kMCTrack,  "mcreco" );
    auto ev_mcshower = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );
    auto ev_mctruth  = (larlite::event_mctruth*)ioll.get_data(  larlite::data::kMCTruth,  "generator" );
    
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

    _run    = iolcv.event_id().run();
    _subrun = iolcv.event_id().subrun();
    _event  = iolcv.event_id().event();    
    
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
    _kppos_v.clear();
    
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

    // filter duplicates
    filter_duplicates();

    // copy positions of keypoints into flat vector for storage
    for ( auto const& kpd : _kpd_v ) {
      _kppos_v.push_back( kpd.keypt );
    }

    // make BVH tree (to help truth point search speed)
    // deprecated
    //makeBVH();
    //printBVH();
  }


  /**
   * make list of end-points for track-like particles
   *
   */
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

  /**
   * make list of end-points for shower-like particles
   *
   */  
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

  /**
   * make info string for KPdata
   *
   */    
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
   * filter out duplicates
   *
   */
  void PrepKeypointData::filter_duplicates()
  {

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

    std::vector<KPdata> kpd_v;
    for ( auto const& kpdidx : kpd_index ) {
      kpd_v.emplace_back( std::move( _kpd_v[kpdidx[0]] ) );
    }
    std::swap(kpd_v,_kpd_v);

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
   * we build a boundary volume hierarchy tree, using a top-down method
   *
   */
  void PrepKeypointData::makeBVH() {

    // std::cout << "=========================" << std::endl;
    // std::cout << " makeBVH" << std::endl;
    // std::cout << "=========================" << std::endl;    
    
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
      _bvhnodes_v.push_back( node );
      node->mother = _bvhroot;
      _bvhroot->children.push_back( node );
    }

    std::queue< bvhnode_t* > q;
    q.push( _bvhroot );

    while ( q.size()>0 ) {
      //std::cout << "process node on the queue" << std::endl;
      bvhnode_t* node = q.front();
      q.pop();

      // make copies of child pointers in order to sort by dimension
      std::vector< bvhnode_t* > x_v[3];
      for (int v=0; v<3; v++) {
        x_v[v].resize(node->children.size(),nullptr);
        for (size_t i=0; i<node->children.size(); i++)
          x_v[v][i] = node->children[i];
      }
      std::sort( x_v[0].begin(), x_v[0].end(), compare_x );
      std::sort( x_v[1].begin(), x_v[1].end(), compare_y );
      std::sort( x_v[2].begin(), x_v[2].end(), compare_z );
      int nchild = (int)x_v[0].size();

      // we split by the largest gap
      float dimlen = -1;
      int longdim = 0;
      float midpt = 0.0;
      int lowidx  = 0;
      int highidx = 0;
      for (int i=0; i<3; i++ ) {

        // get the center point to make a box
        int _lowidx  = 0;
        int _highidx = 0;
        if ( nchild%2==0 ) {
          // even
          _highidx = nchild/2; // 4/2 = 2
          _lowidx  = _highidx-1;                // 1
        }
        else {
          // odd
          _lowidx  = nchild/2; // 3/2 = 1
          _highidx = _lowidx + 1;             // 2
        }

        // calculate gap dist
        float gapdim = 
          fabs(x_v[i][_lowidx]->bounds[i][0] - x_v[i][_highidx]->bounds[i][0]);

        if ( gapdim>=dimlen ) {
          longdim = i;        
          // calculate midpoint using leaf nodes
          midpt   = 0.5*(x_v[i][_lowidx]->bounds[i][0] + x_v[i][_highidx]->bounds[i][0] );
          lowidx  = _lowidx;
          highidx = _highidx;
          dimlen  = gapdim;
        }
      }// end of loop over dimensions
      
      // set this longest dim as the splitting dimension of this node
      node->splitdim = longdim;

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
      // add them into node list, so we know to destroy them
      _bvhnodes_v.push_back( lo_node );
      _bvhnodes_v.push_back( hi_node );

      // now we divide the node's children into the new nodes
      for (int i=0; i<=lowidx; i++ ) {
        x_v[longdim][i]->mother = lo_node;
        lo_node->children.push_back( x_v[longdim][i] );
      }
      for (int i=highidx; i<nchild; i++ ) {
        x_v[longdim][i]->mother = hi_node;        
        hi_node->children.push_back( x_v[longdim][i] );
      }

      // clear the node's children and replace it with the new nodes
      node->children.clear();
      node->children.push_back( lo_node );
      node->children.push_back( hi_node );

      // std::cout << "[split " << strnode(node) << " nchild=" << nchild << "]" << std::endl;
      // std::cout << "  splitdim=" << longdim << " gapsize=" << dimlen << std::endl;
      // std::cout << "  lo: " << strnode(lo_node) << " nchild=" << lo_node->children.size() << std::endl;
      // std::cout << "  hi: " << strnode(hi_node) << " nchild=" << hi_node->children.size() << std::endl;      

      //set them into the queue if they have more than one child
      if (lo_node->children.size()>1 )
        q.push( lo_node );
      if (hi_node->children.size()>1 )
        q.push( hi_node );

      //std::cout << "finished processing node. left in queue: " << q.size() << std::endl;
      //std::cin.get();
    }//end of loop over queue
    
  }

  /**
   * clear past BVH nodes
   *
   * clears _bvhnode_v container and _bvhroot node.
   *
   */
  void PrepKeypointData::clearBVH() {
    for ( auto& pnode : _bvhnodes_v )
      delete pnode;
    _bvhnodes_v.clear();
    if ( _bvhroot )
      delete _bvhroot;
    _bvhroot = nullptr;
  }

  /**
   * print current BVH tree
   */
  void PrepKeypointData::printBVH() {
    print_graph( _bvhroot );
  }

  void PrepKeypointData::findClosestKeypoint( const std::vector<float>& testpt,
                                              int& kpindex, float& dist ) {
    
  }

  /**
   * given a set of match proposals, we make labels
   *
   * label columns:
   *  [0]:   has true end-point with X cm
   *  [1-3]: shift in 3D points from point to closest end-point
   *  [4-7]: shift in 2D pixels from image points to closest end-point: drow, dU, dV, dY
   *
   * 
   */
  void PrepKeypointData::make_proposal_labels( const larflow::PrepMatchTriplets& match_proposals )
  {

    _match_proposal_labels_v.clear();
    _match_proposal_labels_v.reserve(match_proposals._triplet_v.size());


    for (int imatch=0; imatch<match_proposals._triplet_v.size(); imatch++ ) {
      const std::vector<int>& triplet = match_proposals._triplet_v[imatch]; 
      const std::vector<float>& pos   = match_proposals._pos_v[imatch];
      // std::cout << "[match " << imatch << "] "
      //           << "testpt=(" << pos[0] << "," << pos[1] << "," << pos[2] << ") "
      //           << std::endl;

      std::vector<float> label_v(10,0);
      float dist = 1.0e9;
      
      // dumb assignment, loops over all truth keypoints
      // smarter one uses a BVH structure (does being in a leaf volume guarentee that point is the closest?)
      // O(200k) proposals x 40 keypoints
      // brute forces is O(8M) while BVH is O(737K), a factor of 10 speed-up
      const bvhnode_t* leaf = nullptr;
      const KPdata* kpd = nullptr;
      std::vector<float> leafpos(3,0);
      
      if ( _use_bvh ) {
        throw std::runtime_error("BVH method should not be used");
        leaf = recurse_findleaf( pos, _bvhroot );
        // std::cout << "  leaf-node[kpdindex: " << leaf->kpdidx << "] keypt="
        //           << "(" << leaf->bounds[0][0] << ","
        //           << leaf->bounds[1][0] << ","
        //           << leaf->bounds[2][0] << ")"
        //           << std::endl;
        kpd = &(_kpd_v[ leaf->kpdidx ]);
     
        for (int i=0; i<3; i++ ) {
          leafpos[i] = leaf->bounds[i][0];
          dist += (pos[i]-leafpos[i])*(pos[i]-leafpos[i]);
        }
        dist = sqrt(dist);
      }
      else {
        // brute force
        for (auto const& testkpd : _kpd_v ) {
          float testdist = 0.;
          for ( int v=0; v<3; v++ )
            testdist += (testkpd.keypt[v]-pos[v])*(testkpd.keypt[v]-pos[v]);
          testdist = sqrt(testdist);

          if ( dist>testdist ) {
            dist = testdist;
            for (int v=0; v<3; v++ )
              leafpos[v] = testkpd.keypt[v];
            kpd = &testkpd;
          }
        }
      }

      // make label vector

      // within 50 pixels/15 cm
      if ( dist<0.3*50 ) {
        label_v[0] = 1.0;
        _nclose++;
      }
      else {
        label_v[0] = 0.0;
        _nfar++;
      }

      // shift in 3D
      for (int i=0; i<3; i++ ) {
        label_v[1+i] = leafpos[i]-pos[i];
        hdist[i]->Fill(label_v[1+i]);
      }

      // shift in imgcoords
      std::vector<int> imgcoords(4,0);
      imgcoords[0] = match_proposals._sparseimg_vv[0][triplet[0]].row;
      for (int p=0; p<3; p++ ) {
        imgcoords[1+p] = match_proposals._sparseimg_vv[p][triplet[p]].col;
      }
      for (int i=0; i<4; i++) {
        label_v[4+i] = imgcoords[i]-kpd->imgcoord[i];
        hdpix[i]->Fill( label_v[4+i] );
      }

      _match_proposal_labels_v.push_back(label_v);
    }//end of match proposal loop
      
  }

  /** 
   * write tracking histograms
   *
   */
  void PrepKeypointData::writeHists()
  {
    for (int i=0; i<3; i++ ) {
      hdist[i]->Write();
    }
    for (int i=0; i<4; i++ ) {
      hdpix[i]->Write();
    }
  }

  /**
   * define ROOT tree where we save the labels
   *
   */
  void PrepKeypointData::defineAnaTree()
  {
    _label_tree = new TTree("keypointlabels","Key point Training Labels");
    _label_tree->Branch("run",&_run,"run/I");
    _label_tree->Branch("subrun",&_subrun,"subrun/I");
    _label_tree->Branch("event",&_event,"event/I");
    _label_tree->Branch("kplabel",&_match_proposal_labels_v);
    _label_tree->Branch("kppos",&_kppos_v);
  }

  void PrepKeypointData::writeAnaTree()
  {
    if (_label_tree)
      _label_tree->Write();
  }

  /**
   * dump keypoint locations to standard out
   *
   */
  void PrepKeypointData::printKeypoints() const
  {

    std::cout << "[PrepKeypointData::printKeypoints] -----------------" << std::endl;
    for ( size_t i=0; i<_kpd_v.size(); i++ ) {
      auto const& kpd = _kpd_v[i];
      std::cout << "  [" << i << "] "
                << "(" << kpd.keypt[0] << "," << kpd.keypt[1] << "," << kpd.keypt[2] << ")"
                << std::endl;
    }
    std::cout << "----------------------------------------------------" << std::endl;    
    
  }
  
  
}
}
