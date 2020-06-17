#include "PrepKeypointData.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <sstream>
#include <algorithm>
#include <queue>

#include "TH2D.h"

#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"
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
   * expect the following inputs within the event data containers
   * in IOManager:
   * - charge image with tree name _adc_image_treename, specifiable using `setADCimageTreeName`
   * - "segment" images indicating particle type at a given pixel
   * - "instance" images indicating MC ID of particle making charge at pixel
   * - "ancestor" images indicating ancestor track ID for particle making charge at pixel
   *
   * in larlite:
   * - "mcreco" MCTrack tree, holding truth info for track-like particles
   * - "mcreco" MCShower tree, holding truth info for shower-like particles
   * - "generator" MCTruth tree, holding truth about neutrino interaction in image (if exists)
   *
   * important class data members produced by this method:
   * - _kpd_v vector of KPdata class which holds info on ground truth keypoints
   * - _match_proposal_labels_v[3] For each keypoint type (total 3), hold a vector of floats for every 3D spacepoint proposal
   *                               There will be a vector<float> for each triplet in PrepMatchTriplet::_triplet_v
   * - _kppos_v[3] For each keypoint type (total 3), store 3D position for each ground truth keypoint identified
   *
   * 
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
    //mcpg.printGraph();

    // build key-points
    _kpd_v.clear();
    for (int i=0; i<(int)larflow::kNumKeyPoints; i++)
      _kppos_v[i].clear();
    
    // build crossing points for muon track primaries
    std::vector<KPdata> track_kpd
      = getMuonEndpoints( mcpg, adc_v, mctrack_v, &sce );

    std::cout << "[Track Endpoint Results]" << std::endl;
    for ( auto const& kpd : track_kpd ) {
      std::cout << "  " << kpd.str() << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    // add points for shower starts
    std::vector<KPdata> shower_kpd
      = getShowerStarts( mcpg, adc_v, mcshower_v, &sce );
    std::cout << "[Shower Endpoint Results]" << std::endl;
    for ( auto const& kpd : shower_kpd ) {
      std::cout << "  " << kpd.str() << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    // we change the kptype to neutrino vertex for those on it
    _label_nu_keypoints( mctruth_v, adc_v, &sce, _kpd_v );
    
    // filter duplicates
    //filter_duplicates();

    // copy positions of keypoints into flat vector for storage
    for ( auto const& kpd : _kpd_v ) {
      if ( kpd.kptype>=0 && kpd.kptype<larflow::kNumKeyPoints ) {
        _kppos_v[ kpd.kptype ].push_back( kpd.keypt );
      }
      else {
        throw std::runtime_error("unrecognized keypoint type");
      }          
    }

  }


  /**
   * make list of end-points for track-like particles
   *
   * @param[in] mcpg Instance of MCPixelPGraph, which organizes information 
   *                 true particle information into graph, while also associating
   *                 to each truth particle, the pixels in the image (if any)
   * @param[in] adc_v Vector of wire charge image, one for each plane
   * @param[in] mctrack_v Event container (vector) of mctrack objects, containing truth
   *                      information of track-like particles in the event
   * @param[in] psce Pointer to SpaceChargeMicroBooNE class. For converting true
   *                 3D trajectory information into the observed trajectory due to
   *                 space charge effects
   * @return Vector of KPdata instances, one for each ground truth track end
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
          kpd.kptype  = larflow::kTrackEnds;
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
          kpd.kptype  = larflow::kTrackEnds;          
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
   * @param[in] mcpg Instance of MCPixelPGraph, which organizes information 
   *                 true particle information into graph, while also associating
   *                 to each truth particle, the pixels in the image (if any).
   *                 We get a list of showers using this graph, and only consider
   *                 those who have at least 10 visible pixels in one of the planes.
   * @param[in] adc_v Vector of wire charge image, one for each plane
   * @param[in] mcshower_v Event container (vector) of mcshower objects, containing truth
   *                       information of shower-like particles in the event
   * @param[in] psce Pointer to SpaceChargeMicroBooNE class. For converting true
   *                 3D trajectory information into the observed trajectory due to
   *                 space charge effects
   * @return Vector of KPdata instances, one for each ground truth shower start
   * 
   */  
  std::vector<KPdata>
  PrepKeypointData::getShowerStarts( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                     const std::vector<larcv::Image2D>& adc_v,
                                     const larlite::event_mcshower& mcshower_v,
                                     larutil::SpaceChargeMicroBooNE* psce )
  {

    // output vector of keypoint data
    std::vector<KPdata> kpd_v;

    // loop over nodes, look for electron/gamma pixels
    for ( auto const& pnode : mcpg.node_v ) {

      if ( abs(pnode.pid)!=11
           && abs(pnode.pid)!=22 )
        continue;


      int max_plane_pixels = 0;
      for (auto const& pix_v : pnode.pix_vv ) {
        if ( max_plane_pixels<pix_v.size() )
          max_plane_pixels = pix_v.size();
      }

      if ( max_plane_pixels<10 )
        continue;

      
      // start: pnode.start; //should be in apparent position already
      KPdata kpd;
      kpd.crossingtype = 2;
      kpd.trackid = pnode.tid;
      kpd.pid     = pnode.pid;
      kpd.vid     = pnode.vidx;
      kpd.origin  = pnode.origin;
      kpd.is_shower = 1;
      kpd.kptype  = larflow::kShowerStart;        
      kpd.keypt.resize(3,0);
      for (int i=0; i<3; i++)
        kpd.keypt[i]   = pnode.imgpos4[i];

      std::vector<double> dpos(3,0);
      for (int i=0; i<3; i++ ) dpos[i] = pnode.imgpos4[i];

      kpd.imgcoord.resize(4,0);      
      try {
        for (int p=0; p<3; p++)
          kpd.imgcoord[1+p] = (int)larutil::Geometry::GetME()->NearestWire( dpos, p );
      }
      catch (...) {
        continue;
      }
      float tick = pnode.imgpos4[3];
      if ( tick>adc_v[0].meta().min_y() && tick<adc_v[0].meta().max_y() ) {
        kpd.imgcoord[0] = adc_v[0].meta().row( tick );
      }
      else {
        continue;
      }
      kpd_v.emplace_back( std::move(kpd) );

    }//end of node loop
    
    return kpd_v;
  }
  
  /**
   * loop through existing keypoints and change type to neutrino
   * if close to neutrino vertex.
   *
   * @param[in]     mctruth_v Truth information about the neutrino interaction.
   * @param[in]     img_v     Wire Images, just for the meta
   * @param[in]     psce      Pointer to space-charge microboone instance
   * @param[in/out] kpdata_v  Keypoint elements to potentially change
   *
   */
  void PrepKeypointData::_label_nu_keypoints( const larlite::event_mctruth& mctruth_v,
                                              const std::vector<larcv::Image2D>& img_v,
                                              larutil::SpaceChargeMicroBooNE* psce,
                                              std::vector<KPdata>& kpdata_v  )
  {

    // loop over all interactions
    int inu = -1;
    for ( auto const& mct : mctruth_v ) {
      inu++;
      auto const& nu = mct.GetNeutrino();

      auto const& nutraj = nu.Nu().Trajectory();

      if (nutraj.size()>0) {
      
        // get the space-charge corrected neutrino vertex
        std::vector<double> nupos(3,0);
        for (int i=0; i<3; i++ )
          nupos[i] = nutraj.front().Position()[i];

        std::vector<double> offsets = psce->GetPosOffsets( nupos[0], nupos[1], nupos[2] );
        nupos[0] = nupos[0] - offsets[0] + 0.7;
        nupos[1] += offsets[1];
        nupos[2] += offsets[2];

        // make a neutrino keypoint
        KPdata kpd;
        kpd.crossingtype = 0;
        kpd.trackid = 0;
        kpd.pid     = 12;
        kpd.vid     = inu;
        kpd.is_shower = 0;
        kpd.origin  = 1;
        kpd.kptype  = larflow::kNuVertex;
        kpd.keypt.resize(3,0);
        for (int i=0; i<3; i++) kpd.keypt[i] = nupos[i];
        kpd.imgcoord.resize(4,0);

        try {
          for (int p=0; p<3; p++)
            kpd.imgcoord[1+p] = (int)larutil::Geometry::GetME()->NearestWire( nupos, p );
        }
        catch (...) {
          continue;
        }
        float tick = 3200 + nupos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
        if ( tick>img_v[0].meta().min_y() && tick<img_v[0].meta().max_y() ) {
          kpd.imgcoord[0] = img_v[0].meta().row( tick );
        }
        else {
          continue;
        }

        kpdata_v.push_back( kpd );
        
      }//end of if neutrino truth object has trajectory point for vertex
    }//end of loop over mctruth elements
    
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
   * 
   * return a numpy array with keypoints for a given class
   *
   * @param[in] iclass larflow::KeyPoint_t value, indicating keypoint type
   * @return Numpy array with columns [tick,wire-U,wire-V,wire-Y,x,y,z,isshower,origin,pid]
   *
   */
  PyObject* PrepKeypointData::get_keypoint_array( int iclass ) const
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
      if ( kpd.kptype!=(larflow::KeyPoint_t)iclass ) continue;
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
   * return a numpy array with keypoint class scores
   *
   * The score is calculated for each proposed spacepoint using a gaussian where the mean
   *  is the 3d position of the closest ground truth keypoint for the given class.
   * For any point 50*0.3 cm away from a ground truth keypoint, the score is set to zero.
   *
   * Assumes that `process` has already been run.
   *
   * @param[in] sig The sigma used in Gaussian to calculate keypoint class score
   * @return Numpy array with shape [num space points, larflow::kNumKeyPoints classes ]
   *
   */
  PyObject* PrepKeypointData::get_triplet_score_array( float sig ) const
  {

    if ( !PrepKeypointData::_setup_numpy ) {
      import_array1(0);
      PrepKeypointData::_setup_numpy = true;
    }

    // get label info for each triplet proposal
    int npts = (int)_match_proposal_labels_v[0].size();
    for (size_t iclass=0; iclass<larflow::kNumKeyPoints; iclass++) {
      if ( _match_proposal_labels_v[iclass].size()!=npts ) {
        throw std::runtime_error("number of triplet labels/scores for each class does not match!");
      }
    }
    
    int nd = 2;
    npy_intp dims[] = { npts, larflow::kNumKeyPoints };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

    size_t ipt = 0;
    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      for (size_t iclass=0; iclass<larflow::kNumKeyPoints; iclass++) {
        
        auto const& label_v = _match_proposal_labels_v[iclass][ipt];
        
        if ( label_v[0]==0.0 ) {
          *((float*)PyArray_GETPTR2(array,ipt,iclass)) = 0.0;
        }
        else {
          float dist = 0.;
          for (int i=0; i<3; i++) dist += label_v[1+i]*label_v[1+i];
          *((float*)PyArray_GETPTR2(array,ipt,iclass)) = exp( -0.5*dist/(sig*sig) );
        }
      }
    }// end of loop over keypointdata structs
    
    return (PyObject*)array;
  }
  
  /**
   * given a set of space point (i.e. triplet match) proposals, 
   *  we define a vector<float> which we use to we make ground truth information
   *
   * vector elements:
   *  [0]:   1.0 if has true end-point with 0.3*50 cm, 0.0 if not.
   *  [1-3]: shift in 3D points from point to closest end-point
   *  [4-7]: shift in 2D pixels from image points to closest end-point: drow, dU, dV, dY
   * 
   */
  void PrepKeypointData::make_proposal_labels( const larflow::PrepMatchTriplets& match_proposals )
  {

    for (int i=0; i<larflow::kNumKeyPoints; i++) {
      _match_proposal_labels_v[i].clear();
      _match_proposal_labels_v[i].reserve(match_proposals._triplet_v.size());
    }

    for (int imatch=0; imatch<match_proposals._triplet_v.size(); imatch++ ) {

      // triplet index (index of the sparse matrix coordinates)
      const std::vector<int>& triplet = match_proposals._triplet_v[imatch];

      // 3D position formed by intersection of the wires
      const std::vector<float>& pos   = match_proposals._pos_v[imatch];
      // std::cout << "[match " << imatch << "] "
      //           << "testpt=(" << pos[0] << "," << pos[1] << "," << pos[2] << ") "
      //           << std::endl;

      // make a score for each class
      for (int ikpclass=0; ikpclass<(int)larflow::kNumKeyPoints; ikpclass++) {

        // store label values
        // [0]: has a match to a true keypoint
        // [1,2,3]: (dx,dy,dz) to closest keypoint
        // [4,5,6,7]: (dr,du,dv,dy) shift in row and columns
        std::vector<float> label_v(10,0);
        float dist = 1.0e9;
      
        // dumb assignment, loops over all truth keypoints
        // seems fast enough
        const KPdata* kpd = nullptr;
        std::vector<float> leafpos(3,0);
      
        for (auto const& testkpd : _kpd_v ) {
          // ignore those not in class
          if ( testkpd.kptype!=(larflow::KeyPoint_t)ikpclass )
            continue;
          
          float testdist = 0.;
          for ( int v=0; v<3; v++ )
            testdist += (testkpd.keypt[v]-pos[v])*(testkpd.keypt[v]-pos[v]);
          testdist = sqrt(testdist);
        
          if ( dist>testdist ) {
            // update the leadpos and kpd pointer
            dist = testdist;
            for (int v=0; v<3; v++ )
              leafpos[v] = testkpd.keypt[v];
            kpd = &testkpd;
          }
        }//end of loop over true points

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

        // make shift in 3D label
        if ( dist<0.3*50 ) {
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
        }
        _match_proposal_labels_v[ikpclass].push_back(label_v);
      }//end of keypoint class loop
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
   * this TTree is intended to be used to save ground truth information
   * we can load during  training
   *
   */
  void PrepKeypointData::defineAnaTree()
  {
    _label_tree = new TTree("keypointlabels","Key point Training Labels");
    _label_tree->Branch("run",&_run,"run/I");
    _label_tree->Branch("subrun",&_subrun,"subrun/I");
    _label_tree->Branch("event",&_event,"event/I");
    _label_tree->Branch("kplabel_nuvertex",    &_match_proposal_labels_v[0]);
    _label_tree->Branch("kplabel_trackends",   &_match_proposal_labels_v[1]);
    _label_tree->Branch("kplabel_showerstart", &_match_proposal_labels_v[2]);
    _label_tree->Branch("kppos_nuvertex",    &_kppos_v[0]);
    _label_tree->Branch("kppos_trackends",   &_kppos_v[1]);
    _label_tree->Branch("kppos_showerstart", &_kppos_v[2]);
  }

  /**
   * call the ana tree's Write method
   *
   */
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

  /**
   * 
   * dump out th2d of scores, for visualization and debugging
   *
   * @param[in] ikpclass   KeyPoint_t type
   * @param[in] sigma      Width of score Gaussian in cm
   * @param[in] histname   Stem of name to use for TH2D
   * @param[in] tripmaker  Instance of PrepMatchTriplet with prepared triplets
   * @param[in] adc_v      vector of Image2D images, for meta
   * @return  vector of TH2D, one for each 
   * 
   */
  std::vector<TH2D> PrepKeypointData::makeScoreImage( const int ikpclass, const float sigma,
                                                      const std::string histname,
                                                      const larflow::PrepMatchTriplets& tripmaker,
                                                      const std::vector<larcv::Image2D>& adc_v ) const
  {

    std::vector<TH2D> hist_v;
    for ( size_t p=0; p<adc_v.size(); p++ ) {
      std::stringstream ss;
      ss << histname << "_p" << (int)p;
      TH2D hist( ss.str().c_str(), ss.str().c_str(),
                 adc_v[p].meta().cols(), adc_v[p].meta().min_x(), adc_v[p].meta().max_x(),
                 adc_v[p].meta().rows(), adc_v[p].meta().min_y(), adc_v[p].meta().max_y() );

      for (size_t ipt=0; ipt<tripmaker._triplet_v.size(); ipt++) {
        int r = tripmaker._sparseimg_vv[p][ tripmaker._triplet_v[ipt][p] ].row;
        int c = tripmaker._sparseimg_vv[p][ tripmaker._triplet_v[ipt][p] ].col;

        auto const& label_v = _match_proposal_labels_v[ikpclass][ipt];
        
        if ( label_v[0]==0.0 && hist.GetBinContent(c+1,r+1)<0.01 ) {
          hist.SetBinContent( c+1, r+1, 0.01 );
        }
        else if (label_v[0]>0.0) {
          float dist = 0.;
          for (int i=0; i<3; i++) dist += label_v[1+i]*label_v[1+i];
          float score = exp( -0.5*dist/(sigma*sigma) );
          if ( hist.GetBinContent(c+1,r+1)<score )
            hist.SetBinContent( c+1, r+1, score );
        }
      }
      hist_v.emplace_back( std::move(hist) );
    }
    
    return hist_v;
  }
  
}
}
