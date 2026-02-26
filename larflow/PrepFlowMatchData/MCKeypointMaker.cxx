#include "MCKeypointMaker.h"

#include <sstream>
#include <algorithm>
#include <queue>

#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"

// larlite
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "larlite/LArUtil/LArProperties.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/mctrack.h"
#include "larlite/DataFormat/mcshower.h"
#include "larlite/DataFormat/mctruth.h"

#include "ublarcvapp/MCTools/MCPGNode.h"
#include "ublarcvapp/MCTools/MCParticleGraph.h"
#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"
#include "ublarcvapp/MCTools/MCPixelLabelMaker.h"
#include "ublarcvapp/MCTools/EventMCPixelLabels.h"

#include "larflow/RecoUtils/cluster_functions.h"

#include <highfive/H5Easy.hpp>

namespace larflow {
namespace prep {


  /**
   * constructor
   */
  MCKeypointMaker::MCKeypointMaker()
    : larcv::larcv_base("MCKeypointMaker"),
    _adc_image_treename("wire"),
    _mcpg(nullptr),
    _psce(nullptr),
    _ioll(nullptr),
    _iolcv(nullptr)
  {
    _nclose = 0;
    _nfar   = 0;

    // for (int i=0; i<3; i++) hdist[i] = nullptr;
    // for (int i=0; i<4; i++) hdpix[i] = nullptr;
    for (int i=0; i<6; i++) {
      _match_proposal_labels_v[i].clear();
      _kppos_v[i].clear();
      _kp_pdg_trackid_v[i].clear();  
      _kp_startpos_v[i].clear();    
    }

    // so dumb that this is hard-coded.
    tpc_bounds[0][0] = 0.0;
    tpc_bounds[0][1] = 255.0;
    tpc_bounds[1][0] = -116.5;
    tpc_bounds[1][1] =  116.5;
    tpc_bounds[2][0] = 0.5;
    tpc_bounds[2][1] = 1035.5;

  }

  /**
   * deconstructor
   */
  MCKeypointMaker::~MCKeypointMaker()
  {
    // for (int v=0; v<3; v++ )
    //   if ( hdist[v] ) delete hdist[v];
    // for (int v=0; v<4; v++ )
    //   if ( hdpix[v] ) delete hdpix[v];
    // if ( _label_tree ) delete _label_tree;      
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
   * - _kpd_v vector of MCKeypoint class which holds info on ground truth keypoints
   * - _match_proposal_labels_v[3] For each keypoint type (total 3), hold a vector of floats for every 3D spacepoint proposal
   *                               There will be a vector<float> for each triplet in PrepMatchTriplet::_triplet_v
   * - _kppos_v[6] For each keypoint type (total 3), store 3D position for each ground truth keypoint identified
   *
   * 
   *
   * @param[in] iolcv LArCV IOManager containing event data
   * @param[in] ioll  LArLite storage_manager containing event data
   */
  void MCKeypointMaker::process( 
        larcv::IOManager& iolcv,
	      larlite::storage_manager& ioll,
        ublarcvapp::mctools::MCPixelLabelMaker* pmcpixmaker )
  {
    
    auto ev_adc      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,_adc_image_treename);
    //auto ev_segment  = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"segment");
    auto ev_mctrack  = (larlite::event_mctrack*)ioll.get_data(  larlite::data::kMCTrack,  "mcreco" );
    auto ev_mcshower = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );
    auto ev_mctruth  = (larlite::event_mctruth*)ioll.get_data(  larlite::data::kMCTruth,  "generator" );
    
    std::vector<larcv::Image2D> badch_v;
    for ( auto const& img : ev_adc->Image2DArray() ) {
      larcv::Image2D blank(img.meta());
      blank.paint(0.0);
      badch_v.emplace_back( std::move(blank) );
    }

    LARCV_INFO() << "[MCKeypointMaker Inputs]" << std::endl;
    LARCV_INFO() << "  adc images: "      << ev_adc->Image2DArray().size() << std::endl;
    LARCV_INFO() << "  badch images: "    << badch_v.size() << std::endl;    
    //LARCV_INFO() << "  segment images: "  << ev_segment->Image2DArray().size() << std::endl;
    LARCV_INFO() << "  mctracks: " << ev_mctrack->size() << std::endl;
    LARCV_INFO() << "  mcshowers: " << ev_mcshower->size() << std::endl;
    LARCV_INFO() << "  mctruths: " << ev_mctruth->size() << std::endl;

    _run    = iolcv.event_id().run();
    _subrun = iolcv.event_id().subrun();
    _event  = iolcv.event_id().event();   

    // create an internal instance of mcpg
    bool local_mcpg = false;
    if ( _mcpg==nullptr ) {
      _mcpg = new ublarcvapp::mctools::MCParticleGraph; 
      local_mcpg = true;
    }
    
    process( ev_adc->Image2DArray(),
             badch_v,
             *ev_mctrack,
             *ev_mcshower,
             *ev_mctruth );

    bool local_pmcpixmaker = false;
    if ( pmcpixmaker==nullptr ) {
      LARCV_INFO() << "Make own MCPixelLabels for keypoints" << std::endl;
      pmcpixmaker = new ublarcvapp::mctools::MCPixelLabelMaker;
      local_pmcpixmaker = true;
      pmcpixmaker->process( ioll, iolcv, "wiremc");
    }
    _adjust_photon_keypoints( 1.0, 5, 0.5, *_mcpg, pmcpixmaker->_pixels_v);

    if ( local_pmcpixmaker ) {
      delete pmcpixmaker;
      pmcpixmaker = nullptr;
    }

    // refine the points to sit on the nearest true spacepoint that matches its trackid
    //_move_floating_keypoints( match_proposals );
    
    //_clear_output();
    //_copy_to_vectors();

    if ( local_mcpg ) {
      delete _mcpg;
      _mcpg = nullptr;
    }
    
  }

  /**
   * process one event, directly given input containers
   *
   */
  void MCKeypointMaker::process( const std::vector<larcv::Image2D>&    adc_v,
                                  const std::vector<larcv::Image2D>&   badch_v,			  
                                  const larlite::event_mctrack&  mctrack_v,
                                  const larlite::event_mcshower& mcshower_v,
                                  const larlite::event_mctruth&  mctruth_v ) {

    LARCV_DEBUG() << "start" << std::endl;
    
    // allocate space charge class
    bool local_sce = false;
    if ( _psce==nullptr ) {
      _psce = new larutil::SpaceChargeMicroBooNE;
      local_sce = true;
    }

    // make particle graph
    LARCV_DEBUG() << "build graph" << std::endl; 
    bool local_mcpg = false;
    if ( _mcpg!=nullptr ) {
      _mcpg->clear();
    }
    else {
      local_mcpg = true;
      _mcpg = new ublarcvapp::mctools::MCParticleGraph;
    }
    ublarcvapp::mctools::MCParticleGraph& mcpg = *_mcpg;
    mcpg.cluster_nu_particles(true);
    //mcpg.set_verbosity( larcv::msg::kDEBUG );
    try {
      mcpg.buildgraph( mcshower_v, mctrack_v, mctruth_v );
    }
    catch (std::exception& err) {
      LARCV_CRITICAL() << "Error running mcpg: " << err.what() << std::endl;
      throw std::runtime_error("error running mcpg");
    }
    LARCV_DEBUG() << "finished graph" << std::endl;
    if ( logger().level()==larcv::msg::kDEBUG ) {
      mcpg.printGraph(nullptr,false);
    }

    // build key-points container
    clear();

    // build crossing points for muon track primaries
    std::vector<MCKeypoint> track_kpd
      = getMuonEndpoints( mcpg, adc_v, mctrack_v, _psce );

    LARCV_NORMAL() << "[Muon Track Endpoint Results] numfound=" << track_kpd.size() << std::endl;
    for ( auto const& kpd : track_kpd ) {
      std::stringstream ss( kpd.str() );
      std::string strline;
      while ( std::getline(ss, strline, '\n') )
        LARCV_DEBUG() << strline << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    std::vector<MCKeypoint> nonmuon_track_kpd 
      = getNonMuonTrackStarts (mcpg, adc_v, mctrack_v, _psce );
    LARCV_NORMAL() << "[Non-muon track start Results] numfound=" << nonmuon_track_kpd.size() << std::endl;
    for ( auto const& kpd : nonmuon_track_kpd ) {
      std::stringstream ss( kpd.str() );
      std::string strline;
      while ( std::getline(ss, strline, '\n') )
        LARCV_DEBUG() << strline << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    // add points for shower starts
    std::vector<MCKeypoint> shower_kpd
      = getShowerStarts( mcpg, adc_v, mcshower_v, _psce );
    LARCV_NORMAL() << "[Shower Endpoint Results] numfound=" << shower_kpd.size() << std::endl;
    int ishr=0; 
    for ( auto const& kpd : shower_kpd ) {

      if (   kpd.keypt_appear[1]>=tpc_bounds[1][0]
          && kpd.keypt_appear[1]<=tpc_bounds[1][1]
          && kpd.keypt_appear[2]>=tpc_bounds[2][0]
          && kpd.keypt_appear[2]<=tpc_bounds[2][1] ) {
        // only keep keypoints inside the TPC
        std::stringstream ss( kpd.str() );
        std::string strline;
        while ( std::getline(ss, strline, '\n') )
          LARCV_DEBUG() << strline << std::endl;
        ishr++;
        _kpd_v.emplace_back( std::move(kpd) );
      }
    }


    // // we change the kptype to neutrino vertex for those on it
    // //LARCV_NORMAL() << "Do Neutrino Keypoint Labeling" << std::endl;
    // int npre_nukp = (int)_kpd_v.size();
    std::vector< MCKeypoint > nu_kp_v = label_nu_keypoints( mctruth_v, adc_v, _psce );
    LARCV_NORMAL() << "[Nu keypoint results] numfound=" << nu_kp_v.size() << std::endl;
    int inu=0; 
    for ( auto const& kpd : nu_kp_v ) {

      if (   kpd.keypt_appear[0]>=tpc_bounds[0][0]
          && kpd.keypt_appear[0]<=tpc_bounds[0][1]
          && kpd.keypt_appear[1]>=tpc_bounds[1][0]
          && kpd.keypt_appear[1]<=tpc_bounds[1][1] 
          && kpd.keypt_appear[2]>=tpc_bounds[2][0]
          && kpd.keypt_appear[2]<=tpc_bounds[2][1] )
      {
        // only keep keypoints inside the TPC
        std::stringstream ss( kpd.str() );
        std::string strline;
        while ( std::getline(ss, strline, '\n') )
          LARCV_DEBUG() << strline << std::endl;
        inu++;
        _kpd_v.emplace_back( std::move(kpd) );
      }
    }

    // filter duplicates
    //filter_duplicates();

    // copy positions of keypoints into flat vector for storage
    //_clear_output();
    //_copy_to_vectors();
    // for ( auto const& kpd : _kpd_v ) {
    //   if ( kpd.kptype>=0 && kpd.kptype<6 ) {
    //     _kppos_v[ kpd.kptype ].push_back( kpd.keypt );
    //     std::vector<int> pdg_trackid(2);
    //     pdg_trackid[0] = kpd.pid;
    //     pdg_trackid[1] = kpd.trackid;
    //     _kp_pdg_trackid_v[ kpd.kptype ].push_back( pdg_trackid );
    //   }
    //   else {
    //     throw std::runtime_error("unrecognized keypoint type");
    //   }          
    // }

    if ( local_mcpg ) {
      delete _mcpg;
      _mcpg = nullptr;
    }

    if ( local_sce ) {
      delete _psce;
      _psce = nullptr;
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
   * @return Vector of MCKeypoint instances, one for each ground truth track end
   *
   */
  std::vector<MCKeypoint>
  MCKeypointMaker::getMuonEndpoints( ublarcvapp::mctools::MCParticleGraph& mcpg,
                                      const std::vector<larcv::Image2D>& adc_v,
                                      const larlite::event_mctrack& mctrack_v,
                                      larutil::SpaceChargeMicroBooNE* psce )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    bool verbose = false;
    
    // get list of primaries
    // std::vector<ublarcvapp::mctools::MCPGNode*> primaries
    //   = mcpg.getPrimaryParticles(false);
    std::vector<ublarcvapp::mctools::MCPGNode*> muons;
    for ( auto& node : mcpg.node_v ) {
      if ( abs(node.pid)==13 )
        muons.push_back( &node );
    }

    auto const& meta0 = adc_v.front().meta();

    // output vector of keypoint data
    std::vector<MCKeypoint> kpd_v;

    for ( auto const& pnode : muons ) {

      if ( abs(pnode->pid)!=13 )
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
          MCKeypoint kpd;
          //kpd.crossingtype = crossingtype;
          kpd.trackid = pnode->tid;
          kpd.pid     = pnode->pid;
          //kpd.vid     = pnode->vidx;
          kpd.is_shower = 0;
          kpd.origin  = pnode->origin;
          kpd.kptype  = larflow::prep::MCKeypoint::kTrackStart;
          std::vector<int> imgcoord = 
            ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                       4050.0, true, 0.3, 0.1,
                                                                                       kpd.keypt_appear, psce, verbose );
          if ( imgcoord.size()!=4 )
            continue;
            
          kpd.imgcoord.resize(4,0);
          for (int i=0;i<3;i++)
            kpd.imgcoord[i] = imgcoord[i+1];
          kpd.imgcoord[3] = imgcoord[0];
          kpd.row  = imgcoord[0];
          kpd.tick = meta0.pos_y(imgcoord[0]);
          kpd.keypt_true = pnode->first_tpc_pos;
          kpd.startpt_appear = kpd.keypt_appear;

          if ( kpd.imgcoord.size()>0 ) {
            kpd_v.emplace_back( std::move(kpd) );            
          }
          
        }

        if ( crossingtype>=0 && crossingtype<=2 ) {
          MCKeypoint kpd;
          //kpd.crossingtype = crossingtype;
          kpd.trackid = pnode->tid;
          kpd.pid     = pnode->pid;
          //kpd.vid     = pnode->vidx;
          kpd.is_shower = 0;
          kpd.origin  = pnode->origin;
          kpd.kptype  = larflow::prep::MCKeypoint::kTrackEnd;          
          std::vector<int> imgcoord = 
            ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( mctrk, adc_v.front().meta(),
                                                                                       4050.0, false, 0.3, 0.1,
                                                                                       kpd.keypt_appear, psce, verbose );
          kpd.imgcoord.resize(4,0);
          for (int i=0;i<3;i++)
            kpd.imgcoord[i] = imgcoord[i+1];
          kpd.imgcoord[3] = imgcoord[0];
          kpd.row  = imgcoord[0];
          kpd.tick = meta0.pos_y(imgcoord[0]);
          kpd.keypt_true = pnode->first_tpc_pos;
          kpd.startpt_appear = kpd.keypt_appear;
                                                                       
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
   * @return Vector of MCKeypoint instances, one for each ground truth shower start
   * 
   */  
  std::vector<MCKeypoint>
  MCKeypointMaker::getShowerStarts( ublarcvapp::mctools::MCParticleGraph& mcpg,
                                     const std::vector<larcv::Image2D>& adc_v,
                                     const larlite::event_mcshower& mcshower_v,
                                     larutil::SpaceChargeMicroBooNE* psce )
  {

    LARCV_DEBUG() << "start" << std::endl;
    
    // output vector of keypoint data
    std::vector<MCKeypoint> kpd_v;

    // loop over nodes, look for electron/gamma pixels
    for ( auto& pnode : mcpg.node_v ) {

      if ( abs(pnode.pid)!=11
           && abs(pnode.pid)!=22 )
        continue;


      // int max_plane_pixels = 0;
      // for (auto const& pix_v : pnode.pix_vv ) {
      //   if ( max_plane_pixels<pix_v.size() )
      //     max_plane_pixels = pix_v.size();
      // }

      // if ( max_plane_pixels<20 )
      //   continue;

      auto const& shower = mcshower_v.at( pnode.vidx );
      LARCV_DEBUG() << "found shower start: "
		    << "tid=" << pnode.tid << ","
		    << "mtid=" << pnode.mtid << ","
		    << "aid=" << pnode.aid << ") "
		    << "process: " << shower.Process()
		    << std::endl;
      std::string process = shower.Process();
      
      // start: pnode.start; //should be in apparent position already
      MCKeypoint kpd;
      //kpd.crossingtype = 2;
      kpd.trackid = pnode.tid;
      kpd.pid     = pnode.pid;
      //kpd.vid     = pnode.vidx;
      kpd.origin  = pnode.origin;
      kpd.is_shower = 1;
      ublarcvapp::mctools::MCPGNode* mothernode   = mcpg.findTrackID( pnode.mtid );
      ublarcvapp::mctools::MCPGNode* ancestornode = mcpg.findTrackID( pnode.aid );
	
      // priveledge showers from muons
      if ( (mothernode && abs(mothernode->pid)==13) || (ancestornode && abs(ancestornode->pid)==13) ) {
        // mother is a muon or ancestor is a muon
        if ( process=="Decay" || process=="muMinusCaptureAtRest")
          kpd.kptype = larflow::prep::MCKeypoint::kMichel;
        else
          kpd.kptype = larflow::prep::MCKeypoint::kDelta;
      }
      else if ( process=="muIoni" || process=="muBrems"  || process=="muPairProd" || process=="eBrem" || process=="muBrem") {
        kpd.kptype = larflow::prep::MCKeypoint::kDelta;
      }
      else {
        // everything else
        kpd.kptype = larflow::prep::MCKeypoint::kShowerStart;
      }
      // }
      // else {
      // 	std::string msg = "MCKeypointMaker::getShowerStarts - unrecognized process! "+process;
      // 	throw std::runtime_error(msg);
      // }

      // std::vector< float > pixsum_v = mcpg.getTruePhotonTrunkPlanePixelSums( pnode.tid );
      // auto const& pointlist = mcpg.getTruePhotonTrunk3DPoints( pnode );
      // std::sort( pixsum_v.begin(), pixsum_v.end() );
      // float ave_toptwo = (pixsum_v[1]+pixsum_v[2])/2.0*0.0162;
      
      std::vector<float> start_reco(4,0.0);
      std::vector<float> creationpt_reco(4,0.0);

      start_reco = ublarcvapp::mctools::MCPos2ImageUtils::Get()->truepos_to_recopos( pnode.start[0], pnode.start[1], pnode.start[2], pnode.start[3]);
      creationpt_reco = start_reco;

      if ( abs(pnode.pid)==22) {
        start_reco = pnode.first_tpc_pos;
      }

      kpd.keypt_appear.resize(3,0);
      kpd.startpt_appear.resize(3,0);
      for (int i=0; i<3; i++) {
        kpd.keypt_appear[i]   = start_reco[i];
        kpd.startpt_appear[i] = creationpt_reco[i];
      }
      LARCV_DEBUG() << "  shower startpt=(" << kpd.keypt_appear[0] << "," << kpd.keypt_appear[1] << "," << kpd.keypt_appear[2] << ")" << std::endl;

      std::vector<double> dpos(3,0);
      for (int i=0; i<3; i++ ) dpos[i] = start_reco[i];

      kpd.imgcoord.resize(4,0.0);
      try {
        for (int p=0; p<3; p++)
          kpd.imgcoord[p] = (int)larutil::Geometry::GetME()->NearestWire( dpos, p );
      }
      catch (...) {
      	LARCV_DEBUG() << "  shower start could not find a proper nearest wire" << std::endl;
        continue;
      }

      float tick = start_reco[3];

      if ( tick>adc_v[0].meta().min_y() && tick<adc_v[0].meta().max_y() ) {
        kpd.imgcoord[3] = adc_v[0].meta().row( tick ); // wants row?
      }
      else {
	      LARCV_DEBUG() << "  shower start has tick outside of image bounds" << std::endl;
        continue;
      }
      kpd.tick = tick;
      kpd.row  = kpd.imgcoord[3];

      // if ( ave_toptwo < 15.0 ) {
      //   LARCV_DEBUG() << "  shower has too little energy deposited in the trunk (when ave. top two planes): " << ave_toptwo << " < 15.0 MeV" << std::endl;
      //   continue;
      // }
      
      kpd_v.emplace_back( std::move(kpd) );

    }//end of node loop
    
    return kpd_v;
  }

    /**
   * make list of end-points for non0-muon track-like particles
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
   * @return Vector of MCKeypoint instances, one for each ground truth shower start
   * 
   */  
  std::vector<MCKeypoint>
  MCKeypointMaker::getNonMuonTrackStarts( ublarcvapp::mctools::MCParticleGraph& mcpg,
                                     const std::vector<larcv::Image2D>& adc_v,
                                     const larlite::event_mctrack& mctrack_v,
                                     larutil::SpaceChargeMicroBooNE* psce )
  {

    LARCV_DEBUG() << "start" << std::endl;
    std::vector<MCKeypoint> output;
    
    // we have to space-charge correct, so we bump a little inside

    // output vector of keypoint data
    std::vector<MCKeypoint> kpd_v;

    auto const& meta0 = adc_v.front().meta();

    // loop over nodes, look for electron/gamma pixels
    for ( auto& pnode : mcpg.node_v ) {

      if (pnode.type!=0)
        continue; // we are querying only nodes generated from the mctrack container

      if ( abs(pnode.pid)==11 || abs(pnode.pid)==22 || pnode.pid==2112 ) {
        // no showers and no neutrons
        continue;
      }
      if ( abs(pnode.pid)==13 ) {
        // no muons
        continue;
      }
      if (pnode.origin==-1)
        continue;

      auto const& track = mctrack_v.at( pnode.vidx );
      // LARCV_INFO() << "  found non-muon track: "
		  //   << "tid=" << pnode.tid << ","
		  //   << "mtid=" << pnode.mtid << ","
		  //   << "aid=" << pnode.aid << ") "
		  //   << "process: " << track.Process()
		  //   << std::endl;
      std::string process = track.Process();

      if ( track.size()==0 ) {
        // there are no steps inside the cryostat by this particle. skip it
        continue;
      }

      // we ignore low energy stuff we really cannot reconstruct
      // we need to verify what is inside and outside the detector...
      // probably should push this back into the mcpg ... along with pion work by Andy
      bool inside_det = false;
      float total_len = 0.;
      float total_len_indet = 0.;
      std::vector<float> pos_start_tpc(4,0);
      std::vector<float> pos_end_tpc(4,0);
      for (int istep=0; istep<(int)track.size()-1; istep++) {
        const TLorentzVector lpt1 = track.at(istep).Position();
        const TLorentzVector lpt2 = track.at(istep+1).Position();
        TVector3 pt1 = lpt1.Vect();
        TVector3 pt2 = lpt2.Vect();
        TVector3 dstep = pt2-pt1;

        bool in_tpc1=false;
        for (int i=0; i<3; i++) {
          if ( tpc_bounds[i][0]<=pt1[i] && pt1[i]<=tpc_bounds[i][1]){
            in_tpc1 = true;
          }
        }

        bool in_tpc2=false;
        for (int i=0; i<3; i++) {
          if ( tpc_bounds[i][0]<=pt2[i] && pt2[i]<=tpc_bounds[i][1]){
            in_tpc2 = true;
          }
        }

        if (!inside_det) {
          if ( in_tpc1 ) {
            pos_start_tpc = std::vector<float>{ (float)lpt1.X(), (float)lpt1.Y(), (float)lpt1.Z(), (float)(lpt1.T()*1.0e-3) };
          }
          else if (in_tpc2) {
            pos_start_tpc = std::vector<float>{ (float)lpt2.X(), (float)lpt2.Y(), (float)lpt2.Z(), (float)(lpt2.T()*1.0e-3) };
          }
          // we do we determine the crossing pt midstep?
          // we could ... not worry for now ... assume geant4 steps are small enough that we can tolerate imprecision
          inside_det = true;
        }
        else if ( inside_det ) {
          if ( in_tpc1 )
            pos_end_tpc = std::vector<float>{ (float)lpt1.X(), (float)lpt1.Y(), (float)lpt1.Z(), (float)(lpt1.T()*1.0e-3) };
          if ( in_tpc2 )
            pos_end_tpc = std::vector<float>{ (float)lpt2.X(), (float)lpt2.Y(), (float)lpt2.Z(), (float)(lpt2.T()*1.0e-3) };
        }

        if (inside_det)
          total_len += dstep.Mag();
      }

      bool above_threshold = true;
      if ( total_len < 3.0 ) {
        // we anticipate using a scoring sigma of 5 cm
        // also, 3 cm is about 10 pixels in microboone. at that point, we should be able to identify it as a prong
        above_threshold = false;
      }

      // also check how much visible energy it has made in the image
      // int nplanes = 0;
      // for (int p=0; p<pnode.pix_vv.size(); p++) {
      //   if ( pnode.pix_vv.at(p).size()/2>=10 ) {
      //     nplanes++;
      //   }
      // }
      // if (nplanes==0) {
      //   above_threshold = false;
      // }

      if ( !above_threshold ) {
        LARCV_INFO() << "track visible energy deposition is below threshold. skip." << std::endl;
      }
      
      // track start
      MCKeypoint kpd_start;
      //kpd_start.crossingtype = 0;
      kpd_start.trackid = pnode.tid;
      kpd_start.pid     = pnode.pid;
      //kpd_start.vid     = pnode.vidx;
      kpd_start.origin  = pnode.origin;
      kpd_start.is_shower = 0;
      kpd_start.kptype = larflow::prep::MCKeypoint::kTrackStart;
      std::vector<float> pos = ublarcvapp::mctools::MCPos2ImageUtils::Get()->truepos_to_recopos( pos_start_tpc[0],
                                                                                                 pos_start_tpc[1],
                                                                                                 pos_start_tpc[2],                                                                                                  
                                                                                                 pos_start_tpc[3]*1.0e-3 );
      kpd_start.keypt_appear = std::vector<float>{ (float)pos[0], (float)pos[1], (float)pos[2] };     
      kpd_start.startpt_appear = kpd_start.keypt_appear;                                                                                  

      std::vector<int> imgcoord = 
            ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( track, adc_v.front().meta(),
                                                                                       4050.0, true, 0.3, 0.1,
                                                                                       kpd_start.keypt_appear, psce, false );
      // ublarcvapp::mctools::MCPixelPGraph::Node_t* mothernode = mcpg.findTrackID( pnode.mtid );
      // ublarcvapp::mctools::MCPixelPGraph::Node_t* ancestornode = mcpg.findTrackID( pnode.aid );
      //std::cout << "(start) imgcoord.size()=" << imgcoord.size() << std::endl;
      if ( imgcoord.size()>=4 ) {
        kpd_start.imgcoord.resize(4,0);
        for (int i=0; i<3; i++)
          kpd_start.imgcoord[i] = imgcoord[1+i];
        kpd_start.row  = imgcoord[0];
        kpd_start.tick = meta0.pos_y( kpd_start.row );
        kpd_start.imgcoord[3] = kpd_start.row;
        output.emplace_back( std::move(kpd_start) );
      }

      // track end 
      MCKeypoint kpd_end;
      //kpd_end.crossingtype = 1;
      kpd_end.trackid = pnode.tid;
      kpd_end.pid     = pnode.pid;
      //kpd_end.vid     = pnode.vidx;
      kpd_end.origin  = pnode.origin;
      kpd_end.is_shower = 0;
      kpd_end.kptype = larflow::prep::MCKeypoint::kTrackEnd;
      // ublarcvapp::mctools::MCPixelPGraph::Node_t* mothernode = mcpg.findTrackID( pnode.mtid );
      // ublarcvapp::mctools::MCPixelPGraph::Node_t* ancestornode = mcpg.findTrackID( pnode.aid );
      std::vector<float> fendpos = ublarcvapp::mctools::MCPos2ImageUtils::Get()->truepos_to_recopos( pos_end_tpc[0],
                                                                                                 pos_end_tpc[1],
                                                                                                 pos_end_tpc[2],                                                                                                  
                                                                                                 pos_end_tpc[3]*1.0e-3 );
      kpd_end.keypt_appear = std::vector<float>{ (float)fendpos[0], (float)fendpos[1], (float)fendpos[2] };
      kpd_end.startpt_appear = kpd_start.keypt_appear;
      imgcoord = 
        ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( track, adc_v.front().meta(),
                                                                                       4050.0, false, 0.3, 0.1,
                                                                                       kpd_end.keypt_appear, psce, false );
      //std::cout << "(end) imgcoord.size()=" << imgcoord.size() << std::endl;
      if (imgcoord.size()>=4) {
        kpd_end.imgcoord.resize(4,0);
        for (int i=0; i<3; i++)
          kpd_end.imgcoord[i] = imgcoord[1+i];
        kpd_end.row  = imgcoord[0];
        kpd_end.tick = meta0.pos_y( kpd_end.row );
        kpd_end.imgcoord[3] = kpd_end.row;
        output.emplace_back( std::move(kpd_end) );
      }

    }//end of node loop
    
    return output;
  }
  
  /**
   * loop through existing keypoints and change type to neutrino
   * if close to neutrino vertex.
   *
   * @param[in]  mctruth_v Truth information about the neutrino interaction.
   * @param[in]  img_v     Wire Images, just for the meta
   * @param[in]  psce      Pointer to space-charge microboone instance
   * @return     kpdata_v  Keypoint elements to potentially change
   *
   */
  std::vector<larflow::prep::MCKeypoint>
  MCKeypointMaker::label_nu_keypoints( const larlite::event_mctruth& mctruth_v,
                                       const std::vector<larcv::Image2D>& img_v,
                                       larutil::SpaceChargeMicroBooNE* psce )
  {

    // loop over all interactions
    int inu = -1;

    auto const& meta0 = img_v.front().meta();

    std::vector<larflow::prep::MCKeypoint> kpnu_v;

    for ( auto const& mct : mctruth_v ) {
      inu++;
      auto const& nu = mct.GetNeutrino();

      auto const& nutraj = nu.Nu().Trajectory();

      if (nutraj.size()>0) {
      
        // get the space-charge corrected neutrino vertex
        std::vector<double> nupos(3,0);
        std::vector<double> nupos_sce(3,0);
        for (int i=0; i<3; i++ )
          nupos[i] = nutraj.front().Position()[i];

        bool applied_sce = false;
        nupos_sce = psce->ApplySpaceChargeEffect( nupos[0], nupos[1], nupos[2], applied_sce );

        // make a neutrino keypoint
        MCKeypoint kpd;
        //kpd.crossingtype = 0;
        kpd.trackid = 0;
        kpd.pid     = nu.Nu().PdgCode();
        //kpd.vid     = inu;
        kpd.is_shower = 0;
        kpd.origin  = 1;
        kpd.kptype  = larflow::prep::MCKeypoint::kNuVertex;
        kpd.keypt_true.resize(3,0);
        kpd.keypt_appear.resize(3,0);
        kpd.startpt_appear.resize(3,0);
        for (int i=0; i<3; i++) {
          kpd.keypt_true[i]   = nupos[i];
          kpd.keypt_appear[i] = nupos_sce[i];
          kpd.startpt_appear[i] = nupos_sce[i];
        }
        kpd.imgcoord.resize(4,0);

        try {
          for (int p=0; p<3; p++)
            kpd.imgcoord[p] = (int)larutil::Geometry::GetME()->NearestWire( nupos_sce, p );
        }
        catch (...) {
          continue;
        }
        float tick = 3200 + nupos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
        if ( tick>meta0.min_y() && tick<meta0.max_y() ) {
          kpd.imgcoord[3] = img_v[0].meta().row( tick );
          kpd.tick = tick;
          kpd.row  = kpd.imgcoord[3];
        }
        else {
          continue;
        }

        kpnu_v.push_back( kpd );
        
      }//end of if neutrino truth object has trajectory point for vertex
    }//end of loop over mctruth elements

    return kpnu_v;
    
  }

  /**
   * dump keypoint locations to standard out
   *
   */
  void MCKeypointMaker::printKeypoints() const
  {

    std::cout << "[MCKeypointMaker::printKeypoints] -----------------" << std::endl;
    for ( size_t i=0; i<_kpd_v.size(); i++ ) {
      auto const& kpd = _kpd_v[i];
      std::cout << "  [" << i << "] "
                << "type=" << kpd.kptype << " "
                << "trackid=" << kpd.trackid << " "
                << "pid=" << kpd.pid << " "
                << "(" << kpd.keypt_appear[0] << "," << kpd.keypt_appear[1] << "," << kpd.keypt_appear[2] << ")"
                << std::endl;
    }
    std::cout << "----------------------------------------------------" << std::endl;    
    
  }

  void MCKeypointMaker::_adjust_photon_keypoints( 
    float edep_cluster_threshold,
    int edep_cluster_size_threshold,
    float edep_point_threshold,
    ublarcvapp::mctools::MCParticleGraph& mcpg,
    ublarcvapp::mctools::EventMCPixelLabels& pixel3d)
  {

    // to make sane shower keypoints, we use the 3D true energy deposit
    // information stored in the EventMCPixelLabels.
    //
    // (1) use the mcpg to find photon trackids
    // (2) we cluster edep positions for each photon using dbscan
    // (3) we use the true momenta to provide a rough time-axis
    // (4) we find the most upstream point within an above threshold clusters 

    std::vector<long> shower_trackids;
    std::vector< std::vector<float> > shower_momenta_dir;
    std::vector< std::vector<float> > shower_start;
    std::vector< std::vector<float> > shower_origin_v;

    for ( auto& node : mcpg.node_v ) {
      if ( abs(node.pid)==11 || node.pid==22 ) {
        
        if (node.first_edep_pos[0]==0
            && node.first_edep_pos[1]==0
            && node.first_edep_pos[2]==0){
          continue;
        }

        float pnorm = 0.;
        for (int i=1; i<4; i++) {
          pnorm += node.mom4[i]*node.mom4[i];
        }
        pnorm = sqrt(pnorm);
        if ( pnorm>0 ) {
          std::vector<float> showerdir(3,0);
          for (int i=0; i<3; i++) {
            showerdir[i] = node.mom4[1+i]/pnorm;

          }

          // only save shower info if we get a momentum
          shower_trackids.push_back( node.tid );
          shower_momenta_dir.push_back( showerdir );
          shower_start.push_back( node.first_edep_pos );
          shower_origin_v.push_back( node.start );
        }
      }
    }

    LARCV_INFO() << "nummber of shower keypoints to adjust: " << shower_trackids.size() << std::endl;

    // after collecting showers, for each:
    //  1) collect cluster
    //  2) find new keypoint
    for (size_t ishower=0; ishower<shower_trackids.size(); ishower++) {
      long showerid = shower_trackids.at(ishower);

      // do we have a keypoint for this trackid
      long kpd_index = -1;
      for (size_t ikp=0; ikp<_kpd_v.size(); ikp++ ) {
        if ( _kpd_v.at(ikp).trackid==showerid ) {
          kpd_index = (long)ikp;
          break;
        }
      }

      if (kpd_index<0) {
        LARCV_INFO() << " shower[" << showerid << "] no matching keypoint" << std::endl;
        continue;
      }
      LARCV_INFO() << " shower[" << showerid << "] matching keypoint" << std::endl;

      // have a keypoint we want to adjust
      auto& kpd = _kpd_v.at(kpd_index);

      std::vector<float> showerdir = shower_momenta_dir.at(ishower);
      std::vector<float> showerpos = kpd.keypt_appear;
      std::vector<float> shower_origin = kpd.startpt_appear;

      std::vector< std::vector<float> > points_v;
      std::vector< std::vector<float> > edep_vv;
      for ( auto const& pix3d : pixel3d._triplets_v ) {
        auto it_id = pix3d.trackids.find( showerid );
        if ( it_id!=pix3d.trackids.end() ) {
          //std::cout << "[pid=" << showerid << "] pos[0]=" << pix3d.pos[0] << " pos_reco[0]=" << pix3d.pos_reco[0] << std::endl;
          std::vector<float> pos(3,0);
          std::vector<float> edep_v(3,0);
          int nedep_above_threshold = 0;
          for (int i=0; i<3; i++) {
            pos[i]    = pix3d.pos_reco[i];
            edep_v[i] = pix3d.edep[i];
            if ( edep_v[i]>edep_point_threshold )
              nedep_above_threshold++;
          }
          if ( nedep_above_threshold>0 ) {
            points_v.push_back( pos );
            edep_vv.push_back( edep_v );
          }
        }
      }

      // now we cluster these points using dbscan
      float maxdist = 3.0;
      float minsize = 4;
      int maxkd = 10;
      std::vector< larflow::recoutils::cluster_t > cluster_v;
      larflow::recoutils::cluster_sdbscan_spacepoints( points_v, cluster_v, maxdist, minsize, maxkd);
      int nclusters = cluster_v.size(); // skip the last cluster which are noise points
      
      std::vector<float> most_upstream_pt(3,0);
      std::vector<float> most_upstream_edep(3,0);
      float min_s = 1e9;
      bool found_qualifying_pt = false;

      LARCV_INFO() << " shower[" << showerid << "] "
        << " num points=" << points_v.size() 
        << " num clusters=" << nclusters
        << std::endl;
      LARCV_INFO() << "  origin=(" << shower_origin[0] << "," << shower_origin[1] << "," << shower_origin[2] << ")" << std::endl;
      LARCV_INFO() << "  start=(" << showerpos[0] << "," << showerpos[1] << "," << showerpos[2] << ")" << std::endl;
      
      for ( int icluster=0; icluster<nclusters; icluster++ ){
        std::vector<float> edep_planesum(3,0.0);
        auto const& cluster = cluster_v.at(icluster);

        for (int ihit=0; ihit<(int)cluster.hitidx_v.size(); ihit++) {
          auto hitidx = cluster.hitidx_v.at(ihit);
          auto const& hitedep = edep_vv.at(hitidx);
          for (int i=0; i<3; i++) {
            edep_planesum[i] += hitedep[i];
          }
        }
        int nabove_threshold_planes = 0;
        if ( cluster.hitidx_v.size()>0 ) {
          for (int i=0; i<3; i++) {
            if ( edep_planesum[i]>edep_cluster_threshold) {
              nabove_threshold_planes++;
            }
          }
        }

        LARCV_INFO() << "[photon id=" << showerid << "] "
          << "   cluster edep: " 
          <<  edep_planesum[0] << ", "
          <<  edep_planesum[1] << ", "
          <<  edep_planesum[2] << " MeV"
          << " nabove=" << nabove_threshold_planes
          << " numhits=" << cluster.hitidx_v.size()
          << std::endl;

        if ( cluster.hitidx_v.size()<edep_cluster_size_threshold )
          continue;

        if ( nabove_threshold_planes>=2 ) {
          // qualifying cluster, get most upstream position
          for ( auto& testpt : cluster.points_v ) {
            std::vector<float> shower_forward(3,0);
            for (int i=0; i<3; i++)
              shower_forward[i] = shower_origin[i] + 10.0*showerdir[i];
            float s = larflow::recoutils::pointRayProjection3f( showerpos, showerdir, testpt );
            float s_origin = larflow::recoutils::pointRayProjection3f( shower_origin, showerdir, testpt );
            float r_origin = larflow::recoutils::pointLineDistance3f( shower_origin, shower_forward, testpt );
            // std::cout << "  [photon id=" << showerid << "] "
            //            << "x=" << testpt[0]
            //           << " test-photon-cluster: s=" << s 
            //           << " s_origin=" << s_origin 
            //           << " r_origin=" 
            //           << r_origin 
            //           << " ratio=" << r_origin/s_origin
            //           << std::endl;
            if ( s < min_s && s_origin>0.0 && r_origin/s_origin<0.5 ) {
              min_s = s;
              most_upstream_pt = testpt;
              found_qualifying_pt = true;
              most_upstream_edep = edep_planesum;
            }
          }
        }

      }

      if ( found_qualifying_pt ) {
        LARCV_NORMAL() << "Adjust shower keypoint" << std::endl;
        LARCV_NORMAL() << "  trackid=" << kpd.trackid << " pid=" << kpd.pid << std::endl;
        LARCV_NORMAL() << "  from: (" << kpd.keypt_appear[0] << ", " 
          << kpd.keypt_appear[1] << ", "
          << kpd.keypt_appear[2] << ")" << std::endl;
        LARCV_NORMAL() << "  to: (" << most_upstream_pt[0] << ", "
          << most_upstream_pt[1] << ", "
          << most_upstream_pt[2] << ")"
          << std::endl;
        LARCV_NORMAL() << "  edep: " << most_upstream_edep[0] << ", "
          << most_upstream_edep[1] << ", "
          << most_upstream_edep[2] << " MeV"
          << std::endl;
        kpd.keypt_appear = most_upstream_pt;
      }

    }//end of loop over shower ids
  }

  void MCKeypointMaker::clear()
  {
    _kpd_v.clear();
    for (int i=0; i<6; i++) {
      _kppos_v[i].clear();
      _kp_pdg_trackid_v[i].clear();
    }
  }

  void MCKeypointMaker::_copy_to_vectors()
  {
    // copy positions of keypoints into flat vector for storage
    for ( auto const& kpd : _kpd_v ) {
      if ( kpd.kptype>=0 && kpd.kptype<6 ) {
        _kppos_v[ kpd.kptype ].push_back( kpd.keypt_appear );
        _kp_startpos_v[ kpd.kptype ].push_back( kpd.startpt_appear );
        std::vector<int> pdg_trackid(2);
        pdg_trackid[0] = kpd.pid;
        pdg_trackid[1] = kpd.trackid;
        _kp_pdg_trackid_v[ kpd.kptype ].push_back( pdg_trackid );
      }
      else {
        throw std::runtime_error("unrecognized keypoint type");
      }          
    }
  }

  void MCKeypointMaker::export_as_hdf( std::string hdf_outfile )
  {
    LARCV_INFO() << "export to " << hdf_outfile << std::endl;

    HighFive::File file(hdf_outfile, HighFive::File::Overwrite);

    std::string group_prefix_name="";
    save_entry_to_hdf( file, group_prefix_name );

  }

  void MCKeypointMaker::save_entry_to_hdf( 
    HighFive::File& file,
    std::string group_prefix_name )
  {

    std::string groupname = "/mckeypoints";
    if ( group_prefix_name!="" ) {
      groupname = group_prefix_name + "/mckeypoints";
    }
    file.createGroup(groupname);

    // export different arrays for export
    int nkeypoints = _kpd_v.size();

    std::vector< std::vector<float> > pos_appear(nkeypoints);
    std::vector< std::vector<float> > start_appear(nkeypoints);
    std::vector< std::vector<int> >   imgcoord(nkeypoints);
    std::vector< int > kptype(nkeypoints);
    std::vector< int > kppid(nkeypoints);
    std::vector< int > kptrackid(nkeypoints);

    int ikp=0;
    for ( auto const& kpd : _kpd_v ) {
      pos_appear[ikp]   = kpd.keypt_appear;
      start_appear[ikp] = kpd.startpt_appear;
      imgcoord[ikp]   = kpd.imgcoord;
      kptype[ikp]     = kpd.kptype;
      kppid[ikp]      = kpd.pid;
      kptrackid[ikp]  = kpd.trackid;
      ikp++;
    }

    H5Easy::dump( file, groupname+"/pos",      pos_appear);
    H5Easy::dump( file, groupname+"/imgcoord", imgcoord);
    H5Easy::dump( file, groupname+"/kptype",   kptype);
    H5Easy::dump( file, groupname+"/pid",      kppid);
    H5Easy::dump( file, groupname+"/trackid",  kptrackid);
    H5Easy::dump( file, groupname+"/startpos", start_appear);

    file.flush();
  }
  
}
}
