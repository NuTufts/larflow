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

//#include "larflow/RecoUtils/cluster_functions.h"

#include "ublarcvapp/MCTools/MCPGNode.h"
#include "ublarcvapp/MCTools/MCParticleGraph.h"
#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"
#include "ublarcvapp/MCTools/crossingPointsAnaMethods.h"

#include <highfive/H5Easy.hpp>

namespace larflow {
namespace prep {


  /**
   * constructor
   */
  MCKeypointMaker::MCKeypointMaker()
    : larcv::larcv_base("MCKeypointMaker"),
    _adc_image_treename("wire")
  {
    _nclose = 0;
    _nfar   = 0;

    // for (int i=0; i<3; i++) hdist[i] = nullptr;
    // for (int i=0; i<4; i++) hdpix[i] = nullptr;
    for (int i=0; i<6; i++) {
      _match_proposal_labels_v[i].clear();
      _kppos_v[i].clear();
      _kp_pdg_trackid_v[i].clear();      
    }
      
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
	      larlite::storage_manager& ioll )
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
    
    process( ev_adc->Image2DArray(),
             badch_v,
             *ev_mctrack,
             *ev_mcshower,
             *ev_mctruth );

    // refine the points to sit on the nearest true spacepoint that matches its trackid
    //_move_floating_keypoints( match_proposals );
    
    //_clear_output();
    //_copy_to_vectors();
    
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
    larutil::SpaceChargeMicroBooNE sce;

    // make particle graph
    LARCV_DEBUG() << "build graph" << std::endl;    
    ublarcvapp::mctools::MCParticleGraph mcpg;
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
    _clear_output();

    // build crossing points for muon track primaries
    std::vector<MCKeypoint> track_kpd
      = getMuonEndpoints( mcpg, adc_v, mctrack_v, &sce );

    LARCV_NORMAL() << "[Muon Track Endpoint Results] numfound=" << track_kpd.size() << std::endl;
    for ( auto const& kpd : track_kpd ) {
      std::stringstream ss( kpd.str() );
      std::string strline;
      while ( std::getline(ss, strline, '\n') )
        LARCV_DEBUG() << strline << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    std::vector<MCKeypoint> nonmuon_track_kpd 
      = getNonMuonTrackStarts (mcpg, adc_v, mctrack_v, &sce );
    LARCV_NORMAL() << "[Non-muon track start Results] numfound=" << nonmuon_track_kpd.size() << std::endl;
    for ( auto const& kpd : nonmuon_track_kpd ) {
      LARCV_DEBUG() << "  " << kpd.str() << std::endl;
      _kpd_v.emplace_back( std::move(kpd) );
    }

    // // add points for shower starts
    // std::vector<MCKeypoint> shower_kpd
    //   = getShowerStarts( mcpg, adc_v, mcshower_v, &sce );
    // LARCV_NORMAL() << "[Shower Endpoint Results] numfound=" << shower_kpd.size() << std::endl;
    // int ishr=0; 
    // for ( auto const& kpd : shower_kpd ) {
    //   LARCV_INFO() << "  [" << ishr << "] " << kpd.str() << std::endl;
    //   ishr++;
    //   _kpd_v.emplace_back( std::move(kpd) );
    // }

    // // we change the kptype to neutrino vertex for those on it
    // //LARCV_NORMAL() << "Do Neutrino Keypoint Labeling" << std::endl;
    // int npre_nukp = (int)_kpd_v.size();
    // _label_nu_keypoints( mctruth_v, adc_v, &sce, _kpd_v );
    // int npost_nukp = (int)_kpd_v.size();
    // LARCV_NORMAL() << "[Nu keypoint results] numfound=" << npost_nukp-npre_nukp << std::endl;

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
          kpd.imgcoord.resize(4,0);
          for (int i=0;i<3;i++)
            kpd.imgcoord[i] = imgcoord[i+1];
          kpd.imgcoord[3] = imgcoord[0];
          kpd.row  = imgcoord[0];
          kpd.tick = meta0.pos_y(imgcoord[0]);
          kpd.keypt_true = pnode->first_tpc_pos;

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
                                                                       
          if ( kpd.imgcoord.size()>0 ) {
            kpd_v.emplace_back( std::move(kpd) );
          }
        }


      }//if track in image

    }//end of primary loop

    return kpd_v;
  }

  // /**
  //  * make list of end-points for shower-like particles
  //  *
  //  * @param[in] mcpg Instance of MCPixelPGraph, which organizes information 
  //  *                 true particle information into graph, while also associating
  //  *                 to each truth particle, the pixels in the image (if any).
  //  *                 We get a list of showers using this graph, and only consider
  //  *                 those who have at least 10 visible pixels in one of the planes.
  //  * @param[in] adc_v Vector of wire charge image, one for each plane
  //  * @param[in] mcshower_v Event container (vector) of mcshower objects, containing truth
  //  *                       information of shower-like particles in the event
  //  * @param[in] psce Pointer to SpaceChargeMicroBooNE class. For converting true
  //  *                 3D trajectory information into the observed trajectory due to
  //  *                 space charge effects
  //  * @return Vector of MCKeypoint instances, one for each ground truth shower start
  //  * 
  //  */  
  // std::vector<MCKeypoint>
  // MCKeypointMaker::getShowerStarts( ublarcvapp::mctools::MCPixelPGraph& mcpg,
  //                                    const std::vector<larcv::Image2D>& adc_v,
  //                                    const larlite::event_mcshower& mcshower_v,
  //                                    larutil::SpaceChargeMicroBooNE* psce )
  // {

  //   LARCV_DEBUG() << "start" << std::endl;
    
  //   // output vector of keypoint data
  //   std::vector<MCKeypoint> kpd_v;

  //   // loop over nodes, look for electron/gamma pixels
  //   for ( auto& pnode : mcpg.node_v ) {

  //     if ( abs(pnode.pid)!=11
  //          && abs(pnode.pid)!=22 )
  //       continue;


  //     int max_plane_pixels = 0;
  //     for (auto const& pix_v : pnode.pix_vv ) {
  //       if ( max_plane_pixels<pix_v.size() )
  //         max_plane_pixels = pix_v.size();
  //     }

  //     if ( max_plane_pixels<20 )
  //       continue;

  //     auto const& shower = mcshower_v.at( pnode.vidx );
  //     LARCV_DEBUG() << "found shower start: "
	// 	    << "tid=" << pnode.tid << ","
	// 	    << "mtid=" << pnode.mtid << ","
	// 	    << "aid=" << pnode.aid << ") "
	// 	    << "process: " << shower.Process()
	// 	    << std::endl;
  //     std::string process = shower.Process();
      
  //     // start: pnode.start; //should be in apparent position already
  //     MCKeypoint kpd;
  //     kpd.crossingtype = 2;
  //     kpd.trackid = pnode.tid;
  //     kpd.pid     = pnode.pid;
  //     kpd.vid     = pnode.vidx;
  //     kpd.origin  = pnode.origin;
  //     kpd.is_shower = 1;
  //     ublarcvapp::mctools::MCPixelPGraph::Node_t* mothernode = mcpg.findTrackID( pnode.mtid );
  //     ublarcvapp::mctools::MCPixelPGraph::Node_t* ancestornode = mcpg.findTrackID( pnode.aid );
	
  //     // priveledge showers from muons
  //     if ( (mothernode && abs(mothernode->pid)==13) || (ancestornode && abs(ancestornode->pid)==13) ) {
  //       // mother is a muon or ancestor is a muon
  //       if ( process=="Decay" || process=="muMinusCaptureAtRest")
  //         kpd.kptype = larflow::kShowerMichel;
  //       else
  //         kpd.kptype = larflow::kShowerDelta;
  //     }
  //     else if ( process=="muIoni" || process=="muBrems"  || process=="muPairProd" || process=="eBrem" || process=="muBrem") {
  //       kpd.kptype = larflow::kShowerDelta;
  //     }
  //     else {
  //       // everything else
  //       kpd.kptype = larflow::kShowerStart;
  //     }
  //     // }
  //     // else {
  //     // 	std::string msg = "MCKeypointMaker::getShowerStarts - unrecognized process! "+process;
  //     // 	throw std::runtime_error(msg);
  //     // }

  //     std::vector< float > pixsum_v = mcpg.getTruePhotonTrunkPlanePixelSums( pnode.tid );
  //     auto const& pointlist = mcpg.getTruePhotonTrunk3DPoints( pnode );
  //     std::sort( pixsum_v.begin(), pixsum_v.end() );
  //     float ave_toptwo = (pixsum_v[1]+pixsum_v[2])/2.0*0.0162;
      
  //     std::vector<float> start_reco(4,0.0);
  //     if ( abs(pnode.pid)==11) {
  //       start_reco = ublarcvapp::mctools::MCPos2ImageUtils::Get()->truepos_to_recopos( pnode.start[0], pnode.start[1], pnode.start[2], pnode.start[3] );
  //     }
  //     else {
  //       start_reco = pnode.first_edep_pos;
  //     }


  //     kpd.keypt.resize(3,0);
  //     for (int i=0; i<3; i++)
  //       kpd.keypt[i]   = start_reco[i];
  //     LARCV_DEBUG() << "  shower startpt=(" << kpd.keypt[0] << "," << kpd.keypt[1] << "," << kpd.keypt[2] << ")" << std::endl;

  //     std::vector<double> dpos(3,0);
  //     for (int i=0; i<3; i++ ) dpos[i] = start_reco[i];

  //     kpd.imgcoord.resize(4,0.0);
  //     try {
  //       for (int p=0; p<3; p++)
  //         kpd.imgcoord[1+p] = (int)larutil::Geometry::GetME()->NearestWire( dpos, p );
  //     }
  //     catch (...) {
  //     	LARCV_DEBUG() << "  shower start could not find a proper nearest wire" << std::endl;
  //       continue;
  //     }
  //     float tick = pnode.imgpos4[3];
  //     if ( tick>adc_v[0].meta().min_y() && tick<adc_v[0].meta().max_y() ) {
  //       kpd.imgcoord[0] = adc_v[0].meta().row( tick ); // wants row?
  //     }
  //     else {
	//       LARCV_DEBUG() << "  shower start has tick outside of image bounds" << std::endl;
  //       continue;
  //     }


  //     if ( ave_toptwo < 15.0 ) {
  //       LARCV_DEBUG() << "  shower has too little energy deposited in the trunk (when ave. top two planes): " << ave_toptwo << " < 15.0 MeV" << std::endl;
  //       continue;
  //     }
      
  //     kpd_v.emplace_back( std::move(kpd) );

  //   }//end of node loop
    
  //   return kpd_v;
  // }

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
    
    Double_t tpc_bounds[3][2] = { {0,255.0},
                                  {-116.5,116.5},
                                  {0.5,1035.5}}; // so dumb that this is hard-coded.
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

      std::vector<int> imgcoord = 
            ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( track, adc_v.front().meta(),
                                                                                       4050.0, true, 0.3, 0.1,
                                                                                       kpd_start.keypt_appear, psce, false );
      // ublarcvapp::mctools::MCPixelPGraph::Node_t* mothernode = mcpg.findTrackID( pnode.mtid );
      // ublarcvapp::mctools::MCPixelPGraph::Node_t* ancestornode = mcpg.findTrackID( pnode.aid );
      std::cout << "(start) imgcoord.size()=" << imgcoord.size() << std::endl;
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
      imgcoord = 
        ublarcvapp::mctools::CrossingPointsAnaMethods::getFirstStepPosInsideImage( track, adc_v.front().meta(),
                                                                                       4050.0, false, 0.3, 0.1,
                                                                                       kpd_end.keypt_appear, psce, false );
      std::cout << "(end) imgcoord.size()=" << imgcoord.size() << std::endl;
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
  
  // /**
  //  * loop through existing keypoints and change type to neutrino
  //  * if close to neutrino vertex.
  //  *
  //  * @param[in]     mctruth_v Truth information about the neutrino interaction.
  //  * @param[in]     img_v     Wire Images, just for the meta
  //  * @param[in]     psce      Pointer to space-charge microboone instance
  //  * @param[out]    kpdata_v  Keypoint elements to potentially change
  //  *
  //  */
  // void MCKeypointMaker::_label_nu_keypoints( const larlite::event_mctruth& mctruth_v,
  //                                             const std::vector<larcv::Image2D>& img_v,
  //                                             larutil::SpaceChargeMicroBooNE* psce,
  //                                             std::vector<MCKeypoint>& kpdata_v  )
  // {

  //   // loop over all interactions
  //   int inu = -1;
  //   for ( auto const& mct : mctruth_v ) {
  //     inu++;
  //     auto const& nu = mct.GetNeutrino();

  //     auto const& nutraj = nu.Nu().Trajectory();

  //     if (nutraj.size()>0) {
      
  //       // get the space-charge corrected neutrino vertex
  //       std::vector<double> nupos(3,0);
  //       for (int i=0; i<3; i++ )
  //         nupos[i] = nutraj.front().Position()[i];

  //       std::vector<double> offsets = psce->GetPosOffsets( nupos[0], nupos[1], nupos[2] );
  //       nupos[0] = nupos[0] - offsets[0] + 0.7;
  //       nupos[1] += offsets[1];
  //       nupos[2] += offsets[2];

  //       // make a neutrino keypoint
  //       MCKeypoint kpd;
  //       kpd.crossingtype = 0;
  //       kpd.trackid = 0;
  //       kpd.pid     = 12;
  //       kpd.vid     = inu;
  //       kpd.is_shower = 0;
  //       kpd.origin  = 1;
  //       kpd.kptype  = larflow::kNuVertex;
  //       kpd.keypt.resize(3,0);
  //       for (int i=0; i<3; i++) kpd.keypt[i] = nupos[i];
  //       kpd.imgcoord.resize(4,0);

  //       try {
  //         for (int p=0; p<3; p++)
  //           kpd.imgcoord[1+p] = (int)larutil::Geometry::GetME()->NearestWire( nupos, p );
  //       }
  //       catch (...) {
  //         continue;
  //       }
  //       float tick = 3200 + nupos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
  //       if ( tick>img_v[0].meta().min_y() && tick<img_v[0].meta().max_y() ) {
  //         kpd.imgcoord[0] = img_v[0].meta().row( tick );
  //       }
  //       else {
  //         continue;
  //       }

  //       kpdata_v.push_back( kpd );
        
  //     }//end of if neutrino truth object has trajectory point for vertex
  //   }//end of loop over mctruth elements
    
  // }

  // /** 
  //  * filter out duplicates
  //  *
  //  */
  // void MCKeypointMaker::filter_duplicates()
  // {

  //   // first count the number of unique points
  //   std::set< std::vector<int> >    unique_coords;
  //   std::vector< std::vector<int> > kpd_index;
  //   int npts = 0;
  //   for ( size_t ikpd=0; ikpd<_kpd_v.size(); ikpd++ ) {
  //     auto const& kpd = _kpd_v[ikpd];

  //     if (kpd.imgcoord.size()>0) {
  //       if ( unique_coords.find( kpd.imgcoord )==unique_coords.end() ) {
  //         kpd_index.push_back( std::vector<int>{(int)ikpd,0} );
  //         unique_coords.insert( kpd.imgcoord );
  //         npts++;
  //       }
  //     }      
  //   }

  //   std::vector<MCKeypoint> kpd_v;
  //   for ( auto const& kpdidx : kpd_index ) {
  //     kpd_v.emplace_back( std::move( _kpd_v[kpdidx[0]] ) );
  //   }
  //   std::swap(kpd_v,_kpd_v);

  // }
  
  // /**
  //  * 
  //  * return a numpy array with keypoints for a given class
  //  *
  //  * @param[in] iclass larflow::KeyPoint_t value, indicating keypoint type
  //  * @return Numpy array with columns [tick,wire-U,wire-V,wire-Y,x,y,z,isshower,origin,pid]
  //  *
  //  */
  // PyObject* MCKeypointMaker::get_keypoint_array( int iclass ) const
  // {

  //   if ( !MCKeypointMaker::_setup_numpy ) {
  //     import_array1(0);
  //     MCKeypointMaker::_setup_numpy = true;
  //   }
    
  //   // first count the number of unique points
  //   std::set< std::vector<int> >    unique_coords;
  //   std::vector< std::vector<int> > kpd_index;
  //   int npts = 0;
  //   for ( size_t ikpd=0; ikpd<_kpd_v.size(); ikpd++ ) {
  //     auto const& kpd = _kpd_v[ikpd];
  //     if ( kpd.kptype!=(larflow::KeyPoint_t)iclass ) continue;
  //     if (kpd.imgcoord.size()>0) {
  //       if ( unique_coords.find( kpd.imgcoord )==unique_coords.end() ) {
  //         kpd_index.push_back( std::vector<int>{(int)ikpd,0} );
  //         unique_coords.insert( kpd.imgcoord );
  //         npts++;
  //       }
  //     }      
  //   }
    
  //   int nd = 2;
  //   npy_intp dims[] = { npts, 11 };
  //   PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

  //   size_t ipt = 0;
  //   for ( auto& kpdidx : kpd_index ) {
      
  //     auto const& kpd = _kpd_v[kpdidx[0]];

  //     if ( kpdidx[1]==0 ) {
  //       // start img coordinates
  //       for ( size_t i=0; i<4; i++ )
  //         *((float*)PyArray_GETPTR2(array,ipt,i)) = (float)kpd.imgcoord[i];
  //       // 3D point
  //       for ( size_t i=0; i<3; i++ )
  //         *((float*)PyArray_GETPTR2(array,ipt,4+i)) = (float)kpd.keypt[i];
  //       // is shower
  //       *((float*)PyArray_GETPTR2(array,ipt,7)) = (float)kpd.is_shower;
  //       // origin
  //       *((float*)PyArray_GETPTR2(array,ipt,8)) = (float)kpd.origin;
  //       // PID
  //       *((float*)PyArray_GETPTR2(array,ipt,9)) = (float)kpd.pid;
  //       // TrackID
  //       *((float*)PyArray_GETPTR2(array,ipt,10)) = (float)kpd.trackid;
  //       ipt++;
  //     }
  //   }// end of loop over keypointdata structs

  //   return (PyObject*)array;
  // }

  // /**
  //  * return a numpy array with keypoint class scores
  //  *
  //  * The score is calculated for each proposed spacepoint using a gaussian where the mean
  //  *  is the 3d position of the closest ground truth keypoint for the given class.
  //  * For any point 50*0.3 cm away from a ground truth keypoint, the score is set to zero.
  //  *
  //  * Assumes that `process` has already been run.
  //  *
  //  * @param[in] sig The sigma used in Gaussian to calculate keypoint class score
  //  * @return Numpy array with shape [num space points, 6 classes ]
  //  *
  //  */
  // PyObject* MCKeypointMaker::get_triplet_score_array( float sig ) const
  // {

  //   if ( !MCKeypointMaker::_setup_numpy ) {
  //     import_array1(0);
  //     MCKeypointMaker::_setup_numpy = true;
  //   }

  //   // get label info for each triplet proposal
  //   int npts = (int)_match_proposal_labels_v[0].size();
  //   for (size_t iclass=0; iclass<6; iclass++) {
  //     if ( _match_proposal_labels_v[iclass].size()!=npts ) {
  //       throw std::runtime_error("number of triplet labels/scores for each class does not match!");
  //     }
  //   }
    
  //   int nd = 2;
  //   npy_intp dims[] = { npts, 6 };
  //   PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

  //   size_t ipt = 0;
  //   for ( size_t ipt=0; ipt<npts; ipt++ ) {
  //     for (size_t iclass=0; iclass<6; iclass++) {
        
  //       auto const& label_v = _match_proposal_labels_v[iclass][ipt];
        
  //       if ( label_v[0]==0.0 ) {
  //         *((float*)PyArray_GETPTR2(array,ipt,iclass)) = 0.0;
  //       }
  //       else {
  //         float dist = 0.;
  //         for (int i=0; i<3; i++) dist += label_v[1+i]*label_v[1+i];
  //         *((float*)PyArray_GETPTR2(array,ipt,iclass)) = exp( -0.5*dist/(sig*sig) );
  //       }
  //     }
  //   }// end of loop over keypointdata structs
    
  //   return (PyObject*)array;
  // }
  
  // /**
  //  * given a set of space point (i.e. triplet match) proposals, 
  //  *  we define a vector<float> which we use to we make ground truth information
  //  *
  //  * vector elements:
  //  *  [0]:   1.0 if has true end-point with 0.3*50 cm, 0.0 if not.
  //  *  [1-3]: shift in 3D points from point to closest end-point
  //  *  [4-7]: shift in 2D pixels from image points to closest end-point: drow, dU, dV, dY
  //  * 
  //  */
  // void MCKeypointMaker::make_proposal_labels( const larflow::prep::PrepMatchTriplets& match_proposals )
  // {

  //   const float max_dist_to_label = 10.0;
    
  //   for (int i=0; i<6; i++) {
  //     _match_proposal_labels_v[i].clear();
  //     _match_proposal_labels_v[i].reserve(match_proposals._triplet_v.size());
  //   }

  //   for (int imatch=0; imatch<match_proposals._triplet_v.size(); imatch++ ) {

  //     // triplet index (index of the sparse matrix coordinates)
  //     const std::vector<int>& triplet = match_proposals._triplet_v[imatch];

  //     // 3D position formed by intersection of the wires
  //     const std::vector<float>& pos   = match_proposals._pos_v[imatch];
  //     // std::cout << "[match " << imatch << "] "
  //     //           << "testpt=(" << pos[0] << "," << pos[1] << "," << pos[2] << ") "
  //     //           << std::endl;

  //     // make a score for each class
  //     for (int ikpclass=0; ikpclass<(int)6; ikpclass++) {

  //       // store label values
  //       // [0]: has a match to a true keypoint
  //       // [1,2,3]: (dx,dy,dz) to closest keypoint
  //       // [4,5,6,7]: (dr,du,dv,dy) shift in row and columns
  //       std::vector<float> label_v(10,0);
  //       float dist = 1.0e9;
      
  //       // dumb assignment, loops over all truth keypoints
  //       // seems fast enough
  //       const MCKeypoint* kpd = nullptr;
  //       std::vector<float> leafpos(3,0);
      
  //       for (auto const& testkpd : _kpd_v ) {
  //         // ignore those not in class
  //         if ( testkpd.kptype!=(larflow::KeyPoint_t)ikpclass )
  //           continue;
          
  //         float testdist = 0.;
  //         for ( int v=0; v<3; v++ )
  //           testdist += (testkpd.keypt[v]-pos[v])*(testkpd.keypt[v]-pos[v]);
  //         testdist = sqrt(testdist);
        
  //         if ( dist>testdist ) {
  //           // update the leadpos and kpd pointer
  //           dist = testdist;
  //           for (int v=0; v<3; v++ )
  //             leafpos[v] = testkpd.keypt[v];
  //           kpd = &testkpd;
  //         }
  //       }//end of loop over true points

  //       // make label vector

  //       // within 50 pixels/15 cm
	//       bool is_close = false;
  //       if ( dist<max_dist_to_label ) {
	//         is_close = true;
  //         label_v[0] = 1.0;
  //         _nclose++;
  //       }
  //       else {
  //         label_v[0] = 0.0;
  //         _nfar++;
  //       }

  //       if ( is_close ) {	
  //         // make shift in 3D label
  //               for (int i=0; i<3; i++ ) {
  //                 label_v[1+i] = leafpos[i]-pos[i];
  //                 if ( hdist[i] ) hdist[i]->Fill(label_v[1+i]);
  //               }

  //               // shift in imgcoords
  //               std::vector<int> imgcoords(4,0);
  //               imgcoords[0] = match_proposals._sparseimg_vv[0][triplet[0]].row;
  //               for (int p=0; p<3; p++ ) {
  //                 imgcoords[1+p] = match_proposals._sparseimg_vv[p][triplet[p]].col;
  //               }
  //               for (int i=0; i<4; i++) {
  //                 label_v[4+i] = imgcoords[i]-kpd->imgcoord[i];
  //                 if ( hdpix[i] ) hdpix[i]->Fill( label_v[4+i] );
  //               }
  //       }
  //       else {
  //         // empy label to save space
  //         //label_v.clear(); // this messes up file
  //       }
  //       _match_proposal_labels_v[ikpclass].push_back(label_v);
  //     }//end of keypoint class loop
  //   }//end of match proposal loop
      
  // }

  // /** 
  //  * write tracking histograms
  //  *
  //  */
  // void MCKeypointMaker::writeHists()
  // {
  //   for (int i=0; i<3; i++ ) {
  //     hdist[i]->Write();
  //   }
  //   for (int i=0; i<4; i++ ) {
  //     hdpix[i]->Write();
  //   }
  // }

  // /**
  //  * define ROOT tree where we save the labels
  //  *
  //  * this TTree is intended to be used to save ground truth information
  //  * we can load during  training
  //  *
  //  */
  // void MCKeypointMaker::defineAnaTree()
  // {
  //   _label_tree = new TTree("keypointlabels","Key point Training Labels");
  //   _label_tree->Branch("run",    &_run,    "run/I");
  //   _label_tree->Branch("subrun", &_subrun, "subrun/I");
  //   _label_tree->Branch("event",  &_event,  "event/I");

  //   std::string kp_type_names[6]
  //     = {"nuvertex","trackstart","trackend","showerstart","showermichel","showerdelta"};
    
  //   _label_tree->Branch("kplabel_nuvertex",     &_match_proposal_labels_v[0]);
  //   _label_tree->Branch("kplabel_trackstart",   &_match_proposal_labels_v[1]);
  //   _label_tree->Branch("kplabel_trackend",     &_match_proposal_labels_v[2]);    
  //   _label_tree->Branch("kplabel_showerstart",  &_match_proposal_labels_v[3]);
  //   _label_tree->Branch("kplabel_showermichel", &_match_proposal_labels_v[4]);
  //   _label_tree->Branch("kplabel_showerdelta",  &_match_proposal_labels_v[5]);    
  //   _label_tree->Branch("kppos_nuvertex",     &_kppos_v[0]);
  //   _label_tree->Branch("kppos_trackstart",   &_kppos_v[1]);
  //   _label_tree->Branch("kppos_trackend",     &_kppos_v[2]);    
  //   _label_tree->Branch("kppos_showerstart",  &_kppos_v[3]);
  //   _label_tree->Branch("kppos_showermichel", &_kppos_v[4]);
  //   _label_tree->Branch("kppos_showerdelta",  &_kppos_v[5]);

  //   // truth meta data branches
  //   for (int i=0; i<6; i++) {
  //     std::string brname = "kptruth_"+kp_type_names[i];
  //     _label_tree->Branch( brname.c_str(), &_kp_pdg_trackid_v[i] );
  //   }

  //   hdist[0] = new TH1F("hdist_x","",2002,-500,500.0);
  //   hdist[1] = new TH1F("hdist_y","",2002,-500,500.0);
  //   hdist[2] = new TH1F("hdist_z","",2002,-500,500.0);
    
  //   hdpix[0] = new TH1F("hdpix_dt","",1001,-500,500);
  //   hdpix[1] = new TH1F("hdpix_du","",1001,-500,500);
  //   hdpix[2] = new TH1F("hdpix_dv","",1001,-500,500);
  //   hdpix[3] = new TH1F("hdpix_dy","",1001,-500,500);        
    
  // }

  // /**
  //  * call the ana tree's Write method
  //  *
  //  */
  // void MCKeypointMaker::writeAnaTree()
  // {
  //   if (_label_tree)
  //     _label_tree->Write();
  // }

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

  // /**
  //  * 
  //  * dump out th2d of scores, for visualization and debugging
  //  *
  //  * @param[in] ikpclass   KeyPoint_t type
  //  * @param[in] sigma      Width of score Gaussian in cm
  //  * @param[in] histname   Stem of name to use for TH2D
  //  * @param[in] tripmaker  Instance of PrepMatchTriplet with prepared triplets
  //  * @param[in] adc_v      vector of Image2D images, for meta
  //  * @return  vector of TH2D, one for each 
  //  * 
  //  */
  // std::vector<TH2D> MCKeypointMaker::makeScoreImage( const int ikpclass, const float sigma,
  //                                                     const std::string histname,
  //                                                     const larflow::prep::PrepMatchTriplets& tripmaker,
  //                                                     const std::vector<larcv::Image2D>& adc_v ) const
  // {

  //   std::vector<TH2D> hist_v;
  //   for ( size_t p=0; p<adc_v.size(); p++ ) {
  //     std::stringstream ss;
  //     ss << histname << "_p" << (int)p;
  //     TH2D hist( ss.str().c_str(), ss.str().c_str(),
  //                adc_v[p].meta().cols(), adc_v[p].meta().min_x(), adc_v[p].meta().max_x(),
  //                adc_v[p].meta().rows(), adc_v[p].meta().min_y(), adc_v[p].meta().max_y() );

  //     for (size_t ipt=0; ipt<tripmaker._triplet_v.size(); ipt++) {
  //       int r = tripmaker._sparseimg_vv[p][ tripmaker._triplet_v[ipt][p] ].row;
  //       int c = tripmaker._sparseimg_vv[p][ tripmaker._triplet_v[ipt][p] ].col;

  //       auto const& label_v = _match_proposal_labels_v[ikpclass][ipt];
        
  //       if ( label_v[0]==0.0 && hist.GetBinContent(c+1,r+1)<0.01 ) {
  //         hist.SetBinContent( c+1, r+1, 0.01 );
  //       }
  //       else if (label_v[0]>0.0) {
  //         float dist = 0.;
  //         for (int i=0; i<3; i++) dist += label_v[1+i]*label_v[1+i];
  //         float score = exp( -0.5*dist/(sigma*sigma) );
  //         if ( hist.GetBinContent(c+1,r+1)<score )
  //           hist.SetBinContent( c+1, r+1, score );
  //       }
  //     }
  //     hist_v.emplace_back( std::move(hist) );
  //   }
    
  //   return hist_v;
  // }

  // /**
  //  *
  //  * @brief move keypoints such that all are within some distance from a reconstructable spacepoint
  //  *
  //  * We want to avoid 'floating' keypoints that are not on an energy deposit.
  //  *
  //  */
  // void MCKeypointMaker::_move_floating_keypoints(  const larflow::prep::PrepMatchTriplets& match_proposals )
  // {
  //   LARCV_INFO() << "adjust " << _kpd_v.size() << " keypoints" << std::endl;
  //   int ikp=-1;
  //   for ( auto& kpd : _kpd_v ) {
  //     ikp++;
  //     LARCV_INFO() << "[" << ikp << "] check to move keypoint to reconstructable spacepoint" << std::endl;
      
  //     // dont do this for the neutrino keypoint
  //     if ( kpd.kptype==larflow::kNuVertex ) {
  //       LARCV_INFO() << "  do not move neutrino vertices" << std::endl;
  //       continue;
  //     }

  //     // loop through 
  //     float min_dist = 1.0e9;
  //     int ipt = -1;

  //     int npts = match_proposals._instance_id_v.size();

  //     // find the closest 20 pts to cluster
  //     struct spacepoint_t {
  //       float dist; // distance from keypoint
  //       float pos[3];
  //       bool operator< ( spacepoint_t& rhs ) {
  //         if ( dist<rhs.dist )
  //           return true;
  //         return false;
  //       };
  //     };
  //     std::vector<spacepoint_t> pt_v;
  //     pt_v.reserve( npts );
      
  //     for (int ii=0; ii<npts; ii++) {

  //       if ( match_proposals._instance_id_v[ii]==kpd.trackid ) {
  //         spacepoint_t sp;
  //         float dist = 0.;
  //         for (int v=0; v<3; v++) {
  //           float dd = match_proposals._pos_v[ii][v]-kpd.keypt[v];
  //           dist += dd*dd;
  //           sp.pos[v] = match_proposals._pos_v[ii][v];
  //         }
  //         dist = sqrt(dist);
  //         sp.dist = dist;
  //         if ( dist < min_dist ) {
  //           ipt = ii;
  //           min_dist = dist;
  //         }
  //         pt_v.push_back( sp );
  //       }
	
  //     }//end of loop over npts
      
  //     std::sort( pt_v.begin(), pt_v.end() );

  //     // cluster and get pca
  //     int maxn = 200;
  //     int n = ( pt_v.size()<maxn ) ? pt_v.size() : maxn;
  //     std::vector< std::vector<float> > point_vv;
  //     for (int i=0; i<n; i++) {
  //       auto& pt = pt_v[i];
  //       point_vv.push_back( std::vector<float>{pt.pos[0],pt.pos[1],pt.pos[2]} );
  //     }
      
  //     std::vector< larflow::recoutils::cluster_t > cluster_v;
  //     float maxdist = 20.0;
  //     int minsize=5;
  //     int maxkd=200;
  //     larflow::recoutils::cluster_spacepoint_v( point_vv, cluster_v, maxdist, minsize, maxkd );
  //     LARCV_INFO() << "  clustered same-trackid spacepoints. nclusters= " << cluster_v.size() << std::endl;

  //     bool cluster_ok = true;
      
  //     if ( cluster_v.size()>0 ) {
  //       try {
  //         larflow::recoutils::cluster_runpca( cluster_v );
  //       }
  //       catch (...) {
  //         cluster_ok = false;
  //       }
  //     }
  //     else {
  //     	cluster_ok = false;
  //     }

  //     std::vector<float> orig = kpd.keypt;
  //     int closest_end = -1;
  //     if ( !cluster_ok ) {
  //       if ( ipt>=0 ) {
  //         // found a reconstructable spacepoint
  //         // we'll place the point there (should there be a limit?)
  //         for (int v=0; v<3; v++) {
  //           kpd.keypt[v] = match_proposals._pos_v[ipt][v];
  //         }
  //       }
  //     }//end of cluster not ok
  //     else {

  //       int nlargest = 0;
  //       int ilargest = 0;
  //       float cmin_dist = 1e9;
  //       int min_ic = 0;
  //       std::vector<float> new_end;
  //       for (int ic=0; ic<(int)cluster_v.size(); ic++) {

  //         // which end is the keypoint on?
  //         int index0 = cluster_v[ic].ordered_idx_v.front();
  //         std::vector<float> end0 = cluster_v[ic].points_v.at(index0);
  //         int index1 = cluster_v[ic].ordered_idx_v.back();
  //         std::vector<float> end1 = cluster_v[ic].points_v.at(index1);


  //         int closest_end = larflow::recoutils::cluster_closest_pcaend( cluster_v[ic], kpd.keypt );
  //         std::vector<float> testend;
  //         if ( closest_end==0 )
  //           testend = end0;
  //         else
  //           testend = end1;

  //         float dd = 0.;
  //         for (int v=0; v<3; v++) {
  //           dd += ( testend[v]-kpd.keypt[v])*(testend[v]-kpd.keypt[v]);
  //         }
  //         if ( dd < cmin_dist ) {
  //           cmin_dist = dd;
  //           min_ic = ic;
  //           new_end = testend;
  //         }

  //         if ( (int)cluster_v[ic].points_v.size()>nlargest ) {
  //           ilargest = ic;
  //           nlargest = cluster_v[ic].points_v.size();
  //         }
  //       }
  //       LARCV_INFO() << "  largest cluster idx=" << ilargest << " nlargest=" << nlargest << std::endl;
  //       LARCV_INFO() << "  cluster index with closest endpt: " << min_ic << std::endl;
  //       auto* cluster = &(cluster_v.at(min_ic));

  //       // which end is the keypoint on?
  //       int index0 = cluster->ordered_idx_v.front();
  //       std::vector<float> end0 = cluster->points_v.at(index0);
  //       int index1 = cluster->ordered_idx_v.back();
  //       std::vector<float> end1 = cluster->points_v.at(index1);
  //       LARCV_INFO() << "  number of clusters with same trackid=" << cluster_v.size() << std::endl;
  //       LARCV_INFO() << "  largest cluster=" << nlargest << std::endl;
  //       LARCV_INFO() << "  1st PCA endpoints: pt0=(" << end0[0] << "," << end0[1] << "," << end0[2] << ") pt1=(" << end1[0] << "," << end1[1] << "," << end1[2] << ")" << std::endl;
        
        
  //       closest_end = larflow::recoutils::cluster_closest_pcaend( *cluster, kpd.keypt );
  //       if ( closest_end==0 ) {
  //         kpd.keypt = end0;
  //       }
  //       else if (closest_end==1) {
  //         kpd.keypt = end1;
  //       }
  //     }
  //     LARCV_INFO() << "  trackid=" << kpd.trackid 
	// 	   << " original=(" << orig[0] << "," << orig[1] << "," << orig[2] << ") --> "
	// 	   << " keypt=(" << kpd.keypt[0] << "," << kpd.keypt[1] << "," << kpd.keypt[2] << ")"
	// 	   << " dist=" << min_dist
	// 	   << " n=" << n
	// 	   << " end=" << closest_end
	// 	   << " clok=" << cluster_ok
	// 	   << std::endl;
  //   }// End of loop over keypoint list
    
  // }

  void MCKeypointMaker::_clear_output()
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

    file.createGroup("/mckeypoints");

    // export different arrays for export
    int nkeypoints = _kpd_v.size();

    std::vector< std::vector<float> > pos_appear(nkeypoints);
    std::vector< std::vector<int> >   imgcoord(nkeypoints);
    std::vector< int > kptype(nkeypoints);
    std::vector< int > kppid(nkeypoints);
    std::vector< int > kptrackid(nkeypoints);

    int ikp=0;
    for ( auto const& kpd : _kpd_v ) {
      pos_appear[ikp] = kpd.keypt_appear;
      imgcoord[ikp]   = kpd.imgcoord;
      kptype[ikp]     = kpd.kptype;
      kppid[ikp]      = kpd.pid;
      kptrackid[ikp]  = kpd.trackid;
      ikp++;
    }

    H5Easy::dump( file, "/mckeypoints/pos",      pos_appear);
    H5Easy::dump( file, "/mckeypoints/imgcoord", imgcoord);
    H5Easy::dump( file, "/mckeypoints/kptype",   kptype);
    H5Easy::dump( file, "/mckeypoints/pid",      kppid);
    H5Easy::dump( file, "/mckeypoints/trackid",  kptrackid);

    file.flush();
  }
  
}
}
