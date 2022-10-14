#include "TripletTruthFixer.h"

#include "larlite/LArUtil/LArUtilConfig.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/TruthTrackSCE.h"
#include "ublarcvapp/MCTools/NeutrinoVertex.h"
#include "ublarcvapp/RecoTools/DetUtils.h"
#include "larflow/Reco/geofuncs.h"


namespace larflow {
namespace prep {


  TripletTruthFixer::TripletTruthFixer()
    : larcv::larcv_base("TripletTruthFixer"),
      _kExcludeCosmicShowers(true),
      _p_sce(nullptr)
  {
    if ( larutil::LArUtilConfig::Detector()==larlite::geo::kMicroBooNE )
      _p_sce = new larutil::SpaceChargeMicroBooNE();
    else
      _p_sce = nullptr;
  }

  TripletTruthFixer::~TripletTruthFixer()
  {
    if ( _p_sce )
      delete _p_sce;
    _p_sce = nullptr;
  }
  
  /**
   * @brief calculate reassignments for larmatch triplet instance labels
   *
   * @param[in] tripmaker Instance of PrepMatchTriplet which made larmatch propsals and initial truth labels
   * @param[in] iolcv IO manager for larcv data
   * @param[in] ioll  IO manager for larlite data
   */
  void TripletTruthFixer::calc_reassignments( PrepMatchTriplets& tripmaker,
                                              larcv::IOManager& iolcv,
                                              larlite::storage_manager& ioll ) 
  {

    
    // get the mc pixel graph
    ublarcvapp::mctools::MCPixelPGraph mcpg;
    mcpg.buildgraphonly( ioll );
    if ( logger().level()==larcv::msg::kDEBUG )
      mcpg.printGraph(nullptr,false);
    std::vector<ublarcvapp::mctools::MCPixelPGraph::Node_t*> nu_v = mcpg.getNeutrinoParticles();

    larlite::event_mcshower* ev_mcshower
      = (larlite::event_mcshower*)ioll.get_data( larlite::data::kMCShower, "mcreco" );
    larlite::event_mctrack* ev_mctrack
      = (larlite::event_mctrack*)ioll.get_data( larlite::data::kMCTrack, "mcreco" );
    std::vector<float> nuvtx = ublarcvapp::mctools::NeutrinoVertex::getPos3DwSCE( ioll, getSCE() );

    // fix up missing cosmic pixels if any
    _label_cosmic_pid( tripmaker, mcpg, iolcv );
    
    // we start by resolving instance and class consistency checks
    //_enforce_instance_and_class_consistency( tripmaker, mcpg, *ev_mctrack );    
    
    // start off by separately clustering points with different segment ids
    // then match clusters with mcshower objects using shower profile
    // these seed shower objects to which we assign instance labels
    _shower_info_v.clear();
    _make_shower_info( *ev_mcshower, _shower_info_v, _kExcludeCosmicShowers );
    
    // we can then choose to assign fragments of disconnect shower pieces back to the differe
    // shower trunks

    std::vector<int> pid_v;
    std::vector<int> shower_instance_v;
    std::vector<larflow::reco::cluster_t> cluster_v;
    _cluster_same_showerpid_spacepoints( _shower_info_v, cluster_v, pid_v, shower_instance_v, tripmaker, true );


    // fix up proton id to not have such small clusters,
    // will reassign to ancestor or nearest proton id
    larcv::EventImage2D* ev_instance
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D, "instance" );
    _reassignSmallTrackClusters( tripmaker, ev_instance->as_vector(), 10 );
    
    // associate shower cluster fragments
    // for each larlite mcshower and mctrack, we find closest trunk.
    // then we absorb fragments. save as graph
    std::vector<larflow::reco::cluster_t> merged_showers_v;    
    _merge_shower_fragments( cluster_v, pid_v, shower_instance_v, merged_showers_v );
    _reassign_merged_shower_instance_labels( merged_showers_v, _shower_info_v, tripmaker );

    // track relabel
    // we follow along the path of the trunk. hits within 0.5 cm are absorbed as track
    _reassign_showers_along_tracks( tripmaker, *ev_mctrack, *ev_mcshower, nuvtx );

  }

  /**
   * @brief make clusters for electron and gamma labeled points
   *
   * @param[in]  shower_info_v container of shower info objects, use it protect shower fragments
   * @param[out] cluster_v container of clusters to be returned
   * @param[out] pid_v     container of PID labels for the clusters
   * @param[out] shower_instance_v container of shower ID labels for the clusters
   * @param[inout] tripmaker larmatch triplet proposals and truth labels
   * @param[in] reassign_instance_labels if true, reassign instance labels in PrepMatchTriplets instance
   */
  void TripletTruthFixer::_cluster_same_showerpid_spacepoints( const std::vector<ShowerInfo_t>& shower_info_v,
                                                               std::vector<larflow::reco::cluster_t>& cluster_v,
                                                               std::vector<int>& pid_v,
                                                               std::vector<int>& shower_instance_v,
                                                               larflow::prep::PrepMatchTriplets& tripmaker,
                                                               bool reassign_instance_labels )
  {
    
    std::stringstream msg;
    msg << " protected instances: ";
    std::set<int> mcshower_instances;
    std::map<int,larflow::reco::cluster_t> mcshower_fragments;
    for ( auto const& info : shower_info_v ) {
      mcshower_instances.insert( info.trackid );
      mcshower_fragments[info.trackid] = larflow::reco::cluster_t();
      msg << info.trackid << " ";
    }
    msg << std::endl;
    cluster_v.clear();
    pid_v.clear();
    shower_instance_v.clear();

    LARCV_INFO() << msg.str();
    
    for ( auto& matchdata : tripmaker._match_triplet_v ) {    
      
      // make a list of instance id's from mcshower instances
      // we do this to protect the fragment. this will
      // prevent showers close enough to get clustered together
      // from being merged incorrectly. eg. includes
      // parallel gamma-showers converting around the same distance
      // and gamma-showers converting at the vertex.


      const float maxdist = 0.75;
      const int minsize = 10;
      const int maxkd = 50;
    
      std::vector<int> shower_ids = { 3, 4, 5 };
      for (int iid=0; iid<3; iid++) {
	int pid = shower_ids[iid];

	std::vector< std::vector<float> > point_v;
	std::vector< int > tripmaker_idx_v;
	point_v.reserve( matchdata._pos_v.size() );
	tripmaker_idx_v.reserve( matchdata._pos_v.size() );

	for ( int ipt=0; ipt<(int)matchdata._pos_v.size(); ipt++) {
	  if ( matchdata._pdg_v[ipt]==pid && matchdata._origin_v[ipt]==1 ) {
	    int instanceid = matchdata._instance_id_v[ipt];
	    if ( mcshower_instances.find(instanceid)!=mcshower_instances.end() ) {
	      // protected instance id
	      auto& mcshower_fragment = mcshower_fragments[instanceid];
	      mcshower_fragment.points_v.push_back( matchdata._pos_v[ipt] );
	      mcshower_fragment.hitidx_v.push_back( ipt );
	    }
	    else  {
	      // unprotected instance id
	      std::vector<float> pos = matchdata._pos_v[ipt];
	      point_v.push_back(pos);
	      tripmaker_idx_v.push_back(ipt);
	    }
	  }
	}
	
	std::vector< larflow::reco::cluster_t > pid_cluster_v;
	larflow::reco::cluster_sdbscan_spacepoints( point_v, pid_cluster_v, maxdist, minsize, maxkd );

	LARCV_INFO() << "PID[" << pid << "] has " << pid_cluster_v.size() << " dbscan clusters" << std::endl;

	for (auto & cluster : pid_cluster_v ) {
	  // for each cluster, we decide which label to use based on majority vote
	  std::map<int,int> idcount;
	  for ( auto const& hitidx : cluster.hitidx_v ) {
	    int idx = matchdata._instance_id_v[ tripmaker_idx_v[hitidx] ];
	    if ( idcount.find(idx)==idcount.end() )
	      idcount[idx] = 0;
	    idcount[idx]++;
	  }

	  int maxcount = 0;
	  int maxid = 0;
	  for ( auto it=idcount.begin(); it!=idcount.end(); it++ ) {
	    if ( maxcount<it->second ) {
	      maxcount = it->second;
	      maxid = it->first;
	    }
	  }

	  // reassign the origina instance label.
	  // also reassign the hit index to point back to triplet vector
	  for ( auto & hitidx : cluster.hitidx_v ) {
	    matchdata._instance_id_v[ tripmaker_idx_v[hitidx] ] = maxid;
	    hitidx =  tripmaker_idx_v[hitidx];
	  }

	  //xfer cluster to event container
	  cluster_v.emplace_back( std::move(cluster) );
	  pid_v.push_back( pid );
	  shower_instance_v.push_back( maxid );
	}//end of loop over found clusters
	
      }//end of loop over pid

      // add mcshower fragments into cluster list                
      for ( auto const& info : shower_info_v ) {
	auto& mcshower_fragment = mcshower_fragments[info.trackid];
	LARCV_INFO() << "mcshower fragment trackid[" << info.trackid << "] "
		     << "size=" << mcshower_fragment.points_v.size()
		     << std::endl;
	cluster_v.emplace_back( std::move(mcshower_fragment) );
	switch (info.pid) {
	case -11:
	case 11:        
	  pid_v.push_back( 3 );
	  break;
	case 111:
	  pid_v.push_back( 5 );
	  break;
	default:
	  pid_v.push_back( 4 );
	  break;
	}
	shower_instance_v.push_back( info.trackid );
      }//end of shower info loop
    
    }//end of matchdata loop
  }
  
  /**
   * @brief use neighboring pixels to reassign track clusters with small number of voxels
   *
   * These are often secondary protons or pions created by proton/pion reinteractions.
   * Small proton/pion will usually have some larger track with a proper instance nearby
   *
   */
  void TripletTruthFixer::_reassignSmallTrackClusters( larflow::prep::PrepMatchTriplets& tripmaker,
                                                       const std::vector< larcv::Image2D >& instanceimg_v,
                                                       const float threshold )
  {
    
    for ( auto& matchdata : tripmaker._match_triplet_v ) {        

      std::vector< const larcv::Image2D* > pinstance_v
	= ublarcvapp::recotools::DetUtils::getTPCImages( instanceimg_v, matchdata._tpcid, matchdata._cryoid );
      
      const int dvoxel = 3;
      const int nrows  = pinstance_v.front()->meta().rows();
      const int ncols  = pinstance_v.front()->meta().cols();    
      auto const& meta = pinstance_v.front()->meta(); // assuming metas the same for all planes
      
      // make list of track pixels to reassign: proton and charged pion labels
      std::vector<int> track_idx_v;
      // also make count of pixels in each instance
      std::map<int,int> track_instance_count; // key: instance id, value: count
      track_idx_v.reserve( matchdata._pos_v.size() );
      for ( size_t i=0; i<matchdata._pos_v.size(); i++ ) {
	if ( matchdata._truth_v[i]==1 &&
	     (matchdata._pdg_v[i]>=7 && matchdata._pdg_v[i]<=9) ) {
	  track_idx_v.push_back(i);

	  int instanceid = matchdata._instance_id_v[i];
	  if ( track_instance_count.find(instanceid)==track_instance_count.end() )
	    track_instance_count[i] = 0;
	  track_instance_count[i]++;
	}
      }

      // use this to decide which id should be used to replace
      // subthreshold instances
      struct ReplacementTally_t {
	int orig_instanceid;
	std::set<int> replacement_ids;
	ReplacementTally_t()
	  : orig_instanceid(0) {};
	ReplacementTally_t(int id)
	  : orig_instanceid(id) {};
      };
      std::map<int,ReplacementTally_t> replacement_tally_v;
    
      for ( size_t iidx=0; iidx<track_idx_v.size(); iidx++ ) {

	int triplet_idx = track_idx_v[iidx];

	std::vector<int> imgcoord = matchdata.get_triplet_imgcoord_rowcol( triplet_idx );
	int truth_instance_index  = matchdata._instance_id_v[triplet_idx];
      
	auto it=track_instance_count.find( truth_instance_index );
	if ( it==track_instance_count.end() )
	  continue; /// unexpected (should throw an error)
      
	if ( it->second>threshold )
	  continue; // above threshold for reassignment
      
	auto it_t = replacement_tally_v.find( truth_instance_index );
	if ( it_t==replacement_tally_v.end() )
	  replacement_tally_v[truth_instance_index] = ReplacementTally_t(truth_instance_index);

	auto& tally = replacement_tally_v[truth_instance_index];

      
	for (int dr=-dvoxel; dr<=dvoxel; dr++) {

	  if ( imgcoord[3]<=meta.min_y() || imgcoord[3]>=meta.max_y() )
	    continue;
	  int row = (int)meta.row( imgcoord[3] ); // tick to row
        
	  for (int p=0; p<3; p++) {
	    for (int dc=-dvoxel; dc<=dvoxel; dc++) {
	      int col = imgcoord[p]+dc;
	      if (col<0 || col>=ncols ) continue;

	      int iid = pinstance_v[p]->pixel(row,col,__FILE__,__LINE__);

	      // ignore own instanceid
	      if ( iid==truth_instance_index || iid<0)
		continue;
	      
	      // this checks that its a track instance            
	      if ( track_instance_count.find(iid)!=track_instance_count.end() )
		// tally track instance
		tally.replacement_ids.insert( iid );
	      else {
		//std::cout << "trying to replace id=" << truth_instance_index << " but id=" << iid << " not in track count dict" << std::endl;
	      }
	    }//end of col loop
	  }//end of plane loop
	}//end of row loop
      }
    
      // choose the replacement id
      // we pick the id associated to the largest track cluster
      std::map<int,int> replace_trackid;
      for ( auto it=track_instance_count.begin(); it!=track_instance_count.end(); it++ ) {
	if ( it->second>threshold)
	  continue;
      
	auto& tally = replacement_tally_v[it->first];
	int max_nhits = 0;
	int max_replacement_id = 0;
	for (auto& id : tally.replacement_ids ) {
	  if ( track_instance_count[id]>max_nhits ) {
	    max_nhits = track_instance_count[id];
	    max_replacement_id = id;
	  }
	}
	replace_trackid[it->first] = max_replacement_id;
	if ( max_replacement_id>0 ) {
	  LARCV_INFO() << "successfully replace trackid=" << it->first
		       << "( w/ " << it->second << " counts)"
		       << " with " << max_replacement_id << " (w/ " << max_nhits << ")"
		       << std::endl;
	  track_instance_count[max_replacement_id] += it->second;
	  it->second = 0;
	  // need to propagate this reassignment to past reassignments
	  for ( auto it_r=replace_trackid.begin(); it_r!=replace_trackid.end(); it_r++ ) {
	    if ( it_r->second==it->first )
	      it_r->second = max_replacement_id;
	  }
	}
      }
    
      // execute the replacement
      int nreassigned = 0;
      for ( auto& tripidx : track_idx_v ) {
	int instanceid = matchdata._instance_id_v[tripidx];      
	if ( replace_trackid.find( instanceid )!=replace_trackid.end() ) {
	  matchdata._instance_id_v[tripidx] = replace_trackid[instanceid];
	  nreassigned++;
	}
      }
      std::cout << "[TripletTruthFixer::_reassignSmallTrackClusters] num reassigned = " << nreassigned << std::endl;
    }//loop over matchdata
  }

  /**
   * @brief Merge shower fragments onto showers that have mcshower instances
   *
   */
  void TripletTruthFixer::_merge_shower_fragments( std::vector<larflow::reco::cluster_t>& shower_fragments_v,
                                                   std::vector<int>& pid_v,
                                                   std::vector<int>& shower_instance_v,
                                                   std::vector<larflow::reco::cluster_t>& merged_showers_v )
  {

    merged_showers_v.clear();
    
    // assignment of pixel cluster to shower info objects (which were tied to mcshower objects)
    std::vector<int> claimed_cluster_v( shower_fragments_v.size(), 0 );
    int iidx = 0;
    for ( auto& info : _shower_info_v ) {
      int match_cluster_idx = _find_closest_cluster( shower_fragments_v,
                                                     claimed_cluster_v,
                                                     info.shower_vtx,
                                                     info.shower_dir );
      _shower_info_v[iidx].matched_cluster = match_cluster_idx;
      if ( info.origin==1 ) {
        std::cout << "[TripletTruthFixer::_merge_shower_fragments.L" << __LINE__ << "] "
                  << "ShoweIinfo_t[" << iidx << "] pid=" << info.pid
                  << " tid=" << info.trackid
                  << " origin=" << info.origin
                  << " vtx=(" << info.shower_vtx[0] << "," << info.shower_vtx[1] << "," << info.shower_vtx[2] << ")"
                  << "   matched cluster index: " << match_cluster_idx
                  << std::endl;
      }
      iidx++;        
    }

    // now abosrb showers
    _trueshowers_absorb_clusters( _shower_info_v,
                                  shower_fragments_v,
                                  pid_v,
                                  claimed_cluster_v,
                                  merged_showers_v );
    
  }


  /**
   * @brief compile most important data from the larlite::mcshower objects
   *
   * @param[in] ev_mcshower container of mcshower event objects
   * @param[out] info_v container of shower info data we collected
   * @param[in] exclude_cosmic_showers ignore mcshower objects from cosmic origin
   *
   */
  void TripletTruthFixer::_make_shower_info( const larlite::event_mcshower& ev_mcshower,
                                             std::vector< TripletTruthFixer::ShowerInfo_t>& info_v,
                                             bool exclude_cosmic_showers )
  {
    
    for ( size_t idx=0; idx<ev_mcshower.size(); idx++ ) {
      auto const& mcsh = ev_mcshower.at( idx );

      if ( exclude_cosmic_showers && mcsh.Origin()==2 )
        continue;
      
      ShowerInfo_t info;
      info.idx = idx;
      info.trackid = mcsh.TrackID();
      info.pid = mcsh.PdgCode();
      info.origin = mcsh.Origin();
      info.highq_plane = 0;
      info.E_MeV = mcsh.Start().E();
      for (int p=0; p<3; p++) {
        if ( mcsh.Charge()[p]>info.highq_plane )
          info.highq_plane = mcsh.Charge()[p];
      }

      if ( info.highq_plane==0 )
        continue;

      // rank the shower objects
      if ( info.origin==1 && abs(info.pid)==11 )
        info.priority = 0; // from neutrino is electron
      else if ( info.origin==1 && abs(info.pid)==22 )
        info.priority = 1; // from neutrino is gamma
      else if ( info.origin==1 )
        info.priority = 2; // from neutrino other
      else if ( info.origin==2 && abs(info.pid)==11 )
        info.priority = 3; // from cosmic is electron
      else
        info.priority = 4; // from cosmic is gamma

      info.shower_dir.resize(3,0);
      info.shower_vtx.resize(3,0);
      //info.shower_dir_sce.resize(3,0);
      //info.shower_vtx_sce.resize(3,0);
      auto const& detprofdir = mcsh.DetProfile().Momentum().Vect();
      for (int i=0; i<3; i++) {
        info.shower_dir[i] = detprofdir[i]/detprofdir.Mag();
        info.shower_vtx[i] = mcsh.DetProfile().Position()[i];
      }
      //std::cout << "** dir-truth=(" << info.shower_dir[0] << "," << info.shower_dir[1] << "," << info.shower_dir[2] << ")" << std::endl;
      //std::cout << "** vertex-truth=(" << info.shower_vtx[0] << "," << info.shower_vtx[1] << "," << info.shower_vtx[2] << ")" << std::endl;      
      if ( std::isnan(info.shower_vtx[0]) || std::isinf(info.shower_vtx[0])
           || std::isnan(info.shower_dir[0]) || std::isinf(info.shower_dir[0]) ) {
        std::cout << "** shower idx=" << idx << " tid=" << info.trackid << " bad detprof values **" << std::endl;
        continue;
      }

      float dirnorm = 0.;
      for (int v=0; v<3; v++)
        dirnorm += info.shower_dir[v]*info.shower_dir[v];
      dirnorm = sqrt(dirnorm);
      for (int v=0; v<3; v++)
        info.shower_dir[v] /= dirnorm;

      // adjust shower dir due to SCE
      info.cos_sce = 1.0; // not used

      info_v.push_back( info );
    }

    std::sort( info_v.begin(), info_v.end() );
    
  }

  /**
   * @brief Find closest cluster to given shower start and direction
   *
   * @param[in] shower_fragment_v sub-clusters of shower space points.
   * @param[in] claimed_cluster_v Clusters to search
   * @param[in] shower_vtx Vector giving 3D shower start point/vertex for which we find the closest cluster
   * @param[in] shower_dir Vector describing initial 3D shower direction. Not used.
   * 
   */ 
  int TripletTruthFixer::_find_closest_cluster( std::vector< larflow::reco::cluster_t >& shower_fragment_v,
                                                std::vector<int>& claimed_cluster_v,
                                                std::vector<float>& shower_vtx,
                                                std::vector<float>& shower_dir )
  {

    // we define the trunk of the cluster as the one with the most hits
    const float max_s = 10.0;
    const float min_s = -0.5;
    const float max_r = 1.0;

    int most_nhits = 2;
    std::vector<float> trunk_endpt(3,0);
    for (int i=0; i<3; i++) {
      trunk_endpt[i] = shower_vtx[i] + 3.0*shower_dir[i];
    }

    // loop over class member container cluster_v
    int best_matched_cluster = -1;
    float min_dist_2_vtx = 1.0e9;
    for ( size_t idx=0; idx<shower_fragment_v.size(); idx++ ) {

      auto& cluster = shower_fragment_v[idx];
      int nhits_cluster = 0;
      for (int ihit=0; ihit<(int)cluster.points_v.size(); ihit++) {
        float r = larflow::reco::pointLineDistance3f( shower_vtx, trunk_endpt, cluster.points_v[ihit] );
        float s = larflow::reco::pointRayProjection( shower_vtx, shower_dir, cluster.points_v[ihit] );
        float dist2_vtx = 0.;
        for (int i=0; i<3; i++) {
          dist2_vtx += (cluster.points_v[ihit][i]-shower_vtx[i])*(cluster.points_v[ihit][i]-shower_vtx[i]);
        }
        dist2_vtx = sqrt(dist2_vtx);
        if ( s>min_s && s<max_s && r<max_r ) {
          nhits_cluster++;
        }
        if ( min_dist_2_vtx > dist2_vtx )
          min_dist_2_vtx = dist2_vtx;
      }
      if ( nhits_cluster>most_nhits ) {
        most_nhits = nhits_cluster;
        best_matched_cluster = idx;
      }
      // std::cout << "[cluster " << idx << "] mindist2vtx=" << mindist2vtx
      //           << " start=(" << cluster.pca_ends_v[0][0] << "," << cluster.pca_ends_v[0][1] << "," << cluster.pca_ends_v[0][2] << ") "
      //           << " end=("   << cluster.pca_ends_v[1][0] << "," << cluster.pca_ends_v[1][1] << "," << cluster.pca_ends_v[1][2] << ") "        
      //           << std::endl;
    }
    
    // std::cout << "[ ShowerLikelihoodBuilder::_analyze_clusters ] "
    //           << "trunk cluster index=" << best_matched_cluster << " "
    //           << "hits near trunk=" << most_nhits
    //           << " min dist 2 vtx=" << min_dist_2_vtx
    //           << std::endl;
    
    if ( best_matched_cluster<0 )
      return -1;

    claimed_cluster_v[ best_matched_cluster ] += 1;
    return best_matched_cluster;
  }  

  /**
   * @brief absorb other clusters to true mcshower-matched clusters
   *
   * @param[in]  shower_info_v     List of true shower objects
   * @param[in]  shower_fragment_v sub-clusters of shower space points.
   * @param[in]  fragment_pid_v    true particle id for shower fragments.
   * @param[in]  cluster_used_v    flag indicating with cluster was merged into another cluster
   * @param[out] merged_cluster_v  Output merged clusters
   */
  void TripletTruthFixer::_trueshowers_absorb_clusters( std::vector<ShowerInfo_t>& shower_info_v,
                                                        std::vector<larflow::reco::cluster_t>& shower_fragment_v,
                                                        std::vector<int>& fragment_pid_v,
                                                        std::vector<int>& cluster_used_v,
                                                        std::vector<larflow::reco::cluster_t>& merged_cluster_v )
  {

    std::cout << "[TripletTruthFixer::_trueshowers_absorb_clusters.L" << __LINE__ << "] "
              << "merge fragments to " << shower_info_v.size() << " mcshower objects"
              << std::endl;

    // absorb clusters, we loop over all shower trunks, keeping best one.
    // O(N^2) unfortunately ...

    bool apply_max_len = false;

    // we start off by creating clusters for each shower_info_v object
    merged_cluster_v.clear();
    // we calculate a trunk endpt as well
    std::vector< std::vector<float> > trunk_end_vv(shower_info_v.size());
    for ( size_t ish=0; ish<shower_info_v.size(); ish++ ) {
      auto& info = shower_info_v[ish];
      if (info.matched_cluster>=0 ) {
        std::cout << "[TripletTruthFixer::_trueshowers_absorb_clusters.L" << __LINE__ << "] "
                  << " add shower fragment for trunk" << std::endl;
        merged_cluster_v.push_back( shower_fragment_v[info.matched_cluster] );
        cluster_used_v[info.matched_cluster] = 1;
      }
      else {
        // an empty cluster
        std::cout << "[TripletTruthFixer::_trueshowers_absorb_clusters.L" << __LINE__ << "] "
                  << " no fragment found for shower info" << std::endl;
        merged_cluster_v.push_back( larflow::reco::cluster_t() );
      }
      auto& trunk_end_v = trunk_end_vv[ish];
      trunk_end_v.resize(3,0);
      for (int i=0; i<3; i++)
        trunk_end_v[i] = info.shower_vtx[i] + 3.0*info.shower_dir[i];
    }

    // loop over all fragments and assign to best matching shower
    for (int icluster=0; icluster<(int)shower_fragment_v.size(); icluster++) {
      
      if ( cluster_used_v[icluster]==1 )
        continue;
      
      auto& cluster = shower_fragment_v[icluster];
      int fragment_pid = fragment_pid_v[icluster];

      int max_nhits_in_cone = 0;
      int best_shower_index = -1;
      float closest_shower_dist = 1e9;
    
      for (size_t ish=0; ish<shower_info_v.size(); ish++ ) {
        
        auto& info = shower_info_v.at(ish);
        if ( info.matched_cluster<0 )
          continue;

        int trunk_pid = fragment_pid_v[info.matched_cluster];
        if ( trunk_pid!=fragment_pid )
          continue;
      
        std::vector<float>& trunk_end_v = trunk_end_vv[ish];

        // calculate max range, 14 cm radiation length
        // cut-off at 1 MeV
        float f_E = 0.001/info.E_MeV;
        float maxlen_cm = -log(f_E)*14.0;
        if ( maxlen_cm<500.0 )
          maxlen_cm = 500.0;
        
        int nhits_in_cone = 0;
        float min_dist = 1e9;
        
        for (int ihit=0; ihit<(int)cluster.points_v.size(); ihit++) {
            
          float r = larflow::reco::pointLineDistance3f(  info.shower_vtx, trunk_end_v, cluster.points_v[ihit] );
          float s = larflow::reco::pointRayProjection3f( info.shower_vtx, info.shower_dir, cluster.points_v[ihit] );

          float dist = 0.;
          for (int i=0; i<3; i++) {
            float dx = cluster.points_v[ihit][i] - info.shower_vtx[i]; 
            dist += dx*dx;
          }
          dist = sqrt(dist);
          if ( dist<min_dist )
            min_dist = dist;
          
          float rovers = 0;
          if ( s>0.0 )
            rovers = r/s;
          
          //std::cout << " s=" << s << " rovers=" << rovers << std::endl;
                                                             
          if ( s>0 && rovers<9.0/14.0 && ( !apply_max_len || s<maxlen_cm ) )
            nhits_in_cone++;
          
        }
        
        if ( nhits_in_cone<(int)cluster.points_v.size() ) {
          if ( nhits_in_cone>max_nhits_in_cone ) {
            best_shower_index = ish;
            max_nhits_in_cone = nhits_in_cone;
            closest_shower_dist = min_dist;            
          }
        }
        else {
          if ( min_dist<closest_shower_dist ) {
            best_shower_index = ish;
            max_nhits_in_cone = nhits_in_cone;
            closest_shower_dist = min_dist;
          }
        }
        
      }//end of shower loop

      float frac = float(max_nhits_in_cone)/float(cluster.points_v.size());
      //std::cout << "ShowerLikelihoodBuilder:: cluster[" << icluster << "] most frac inside shower cone info[" << best_shower_index << "] = " << frac << std::endl;
      
      if ( frac>0.5 && best_shower_index>=0 ) {
        // add cluster
        // std::cout << "[TripletTruthFixer::_trueshowers_absorb_clusters.L" << __LINE__ << "] "
        //           << " merge fragment into shower[" << best_shower_index << "]"
        //           << " nhits-before-merge=" << merged_cluster_v[best_shower_index].points_v.size()
        //           << " fragment-size=" << cluster.points_v.size()
        //           << std::endl;

        cluster_used_v[icluster] = 1;
        shower_info_v[best_shower_index].absorbed_cluster_index_v.push_back(icluster);
        // merged_cluster_v[best_shower_index].push_back( truehit_v[hitidx] ); // copy of hit
        larflow::reco::cluster_append( merged_cluster_v[best_shower_index], cluster );
      }
      else  {
        // std::cout << "[TripletTruthFixer::_trueshowers_absorb_clusters.L" << __LINE__ << "] "
        //           << " do not merge fragment. frac=" << frac
        //           << " best_shower_index=" << best_shower_index
        //           << std::endl;
      }
      
    }//end of cluster loop
    
  }//end of _trueshowers_absorb_clusters

  /**
   * @brief reassign instance labels for merged clusters
   * 
   * @param[in] merged_showers_v
   * @param[in] shower_info_v
   * @param[inout] tripmaker
   */
  void TripletTruthFixer::_reassign_merged_shower_instance_labels( std::vector<larflow::reco::cluster_t>& merged_showers_v,
                                                                   std::vector<ShowerInfo_t>& shower_info_v,
                                                                   larflow::prep::PrepMatchTriplets& tripmaker )
  {

    if ( merged_showers_v.size()!=shower_info_v.size() ) {
      std::stringstream msg;
      msg << "[TripletTruthFixer::_reassign_merged_shower_instance_labels.L" << __LINE__ << "] "
	  << "num merged showers (" << merged_showers_v.size() << ") != "
	    << "num shower info (" << shower_info_v.size() << ")"
	  << std::endl;
      throw std::runtime_error( msg.str() );
    }
    
    for ( auto& matchdata : tripmaker._match_triplet_v ) {        
    

      for ( int ish=0; ish<(int)shower_info_v.size(); ish++ ) {
	auto& info = shower_info_v[ish];
	if ( info.matched_cluster<0 ) continue;
	int use_label = info.trackid;
	for ( auto& hitidx : merged_showers_v[ish].hitidx_v ) {
	  matchdata._instance_id_v[hitidx] = use_label;
	}
      }
      
    }
    
  }

  /**
   * @brief enforce instance and class consistency
   * 
   * make sure instance spacepoints are associated with only one type of class
   *
   */
  void TripletTruthFixer::_enforce_instance_and_class_consistency( larflow::prep::PrepMatchTriplets& tripmaker,
                                                                   ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                                   const larlite::event_mctrack& ev_mctrack )
  {

    // we loop over tpcs
    for ( auto& matchdata : tripmaker._match_triplet_v ) {    
    
      int max_iid_value = 0;
      std::map<int,std::vector<int> > instance_segment_counts;
      std::map<int,int> instance_origin_counts;
      
      for ( size_t idx=0; idx<matchdata._instance_id_v.size(); idx++ ) {
	int iid = matchdata._instance_id_v[idx];
	int pid = matchdata._pdg_v[idx];
	if ( iid<=0 ) continue;

	if ( instance_segment_counts.find( iid )==instance_segment_counts.end() ) {
	  instance_segment_counts[iid] = std::vector<int>( larcv::kROITypeMax, 0 );
	  instance_origin_counts[iid] = 0;
	}

	if ( pid<0 ) pid = 0;
	instance_segment_counts[iid][pid]++;
	if ( iid>max_iid_value )
	  max_iid_value = iid;
	if ( matchdata._origin_v[idx]==1 )
	  instance_origin_counts[iid]++;
      }

      for ( auto it=instance_segment_counts.begin(); it!=instance_segment_counts.end(); it++ ) {
	int n_non_zero = 0;
	int max_pid = 0;
	int num_max_pid = 0;
	for ( int pid=0; pid<(int)larcv::kROITypeMax; pid++) {
	  if ( it->second[pid]>0 ) n_non_zero++;
	  if ( pid>0 && it->second[pid]>num_max_pid ) {
	    num_max_pid = it->second[pid];
	    max_pid = pid;
	  }
	}
	
	int second_pid = 0;
	int num_second_pid = 0;
	if ( n_non_zero>1 ) {
	  for ( int pid=0; pid<(int)larcv::kROITypeMax; pid++) {
	    if ( pid!=max_pid && it->second[pid]>num_second_pid ) {
	      second_pid = pid;
	      num_second_pid = it->second[pid];
	    }
	  }
	}
	
	if ( n_non_zero>1 ) {
	  
	  // if ( max_pid<larcv::kROIEminus )
	  //   max_pid = larcv::kROIGamma;

	  ublarcvapp::mctools::MCPixelPGraph::Node_t* pNode = mcpg.findTrackID( it->first );
	  if ( pNode!=nullptr && pNode->tid==it->first && pNode->type==0 ) {
	    // is a track!
	    // no need to vote
	    if ( max_pid==larcv::kROIEminus || max_pid==larcv::kROIGamma ) {
	      // switch the labels
	      second_pid = max_pid;
	    }
	    switch( pNode->pid ) {
	    case 2212:
	    case 2112:            
	      max_pid = larcv::kROIProton;
	      break;
	    case 13:
	    case -13:
	      max_pid = larcv::kROIMuminus;
	      break;
	    case 321:
	    case -321:          
	      max_pid = larcv::kROIKminus;
	      break;            
	    default:
	      max_pid = larcv::kROIPiminus;
	      break;
	    }
	  }
        
	  if ( max_pid>larcv::kROIGamma && (second_pid==larcv::kROIEminus || second_pid==larcv::kROIGamma )) {
	    // we are going to shave off shower points into new instance id
	    // get direction of proton
	    if ( pNode!=nullptr && pNode->tid==it->first && pNode->type==0 && pNode->origin==1) {
	      // do this for neutrino for now -- come back to cosmic later
	      
	      int nrelabeled = 0;
	      std::cout << "[TripletTruthFixer::_enforce_instance_and_class_consistency.L" << __LINE__ << "] "
			<< "using proton truth direction to relabel pixels as shower" << std::endl;

	      
	      ublarcvapp::mctools::TruthTrackSCE trackutil( _p_sce );
	      //trackutil.set_verbosity( larcv::msg::kDEBUG );
	      auto const& mct = ev_mctrack.at( pNode->vidx );
	      larlite::track sce_track = trackutil.applySCE( mct );
	      //std::cout << "debug: " << sce_track.NumberTrajectoryPoints() << std::endl;
	      //std::cin.get();

	      for ( size_t idx=0; idx<matchdata._instance_id_v.size(); idx++ ) {
		if ( matchdata._instance_id_v[idx]==it->first ) {
		  const std::vector<float>& pos = matchdata._pos_v[idx];
		  float min_r = 1.0e9;
		  int min_step = -1;
		  trackutil.dist2track( pos, sce_track, min_r, min_step );
                
		  if ( min_r>2.0 ) {
		    matchdata._pdg_v[idx] = second_pid;
		    matchdata._instance_id_v[idx] = max_iid_value+1;
		    nrelabeled++;
		  }
		}
	      }
	      
	      std::cout << "[TripletTruthFixer::_enforce_instance_and_class_consistency.L" << __LINE__ << "] "
			<< "number relabeled=" << nrelabeled
			<< " to instance-id=" << max_iid_value+1
			<< " with pid=" << second_pid
			<< std::endl;
	      max_iid_value++;
	    }//end of node found, so make modifications
	  }
	  
	  float origin_denom = 0.;
	  for (auto& ncount : instance_segment_counts[it->first] )
	    origin_denom += (float)ncount;
               
	  float origin_frac = 0.;
	  if ( n_non_zero>0 )
	    origin_frac = float( instance_origin_counts[it->first] )/origin_denom;
	  int origin_label = 0;
	  if ( origin_frac>0.9 )
	    origin_label = 1;
        
	  // majority wins
	  std::cout << "[TripletTruthFixer::_enforce_instance_and_class_consistency.L" << __LINE__ << "] "
		    << "instance[" << it->first << "] has " << n_non_zero << " classes: ";
	  for ( int pid=0; pid<(int)larcv::kROITypeMax; pid++) {
	    if ( it->second[pid]>0 )
	      std::cout << "[" << pid << "](" << it->second[pid] << ") ";
	  }
	  std::cout << " origin[" << origin_label << "," << origin_frac << "] ";
	  std::cout << " :: set to " << max_pid << std::endl;

        
	  for ( size_t idx=0; idx<matchdata._instance_id_v.size(); idx++ ) {
	    if ( matchdata._instance_id_v[idx]==it->first ) {
	      matchdata._pdg_v[idx] = max_pid;
	      matchdata._origin_v[idx] = origin_label;
	    }
	  }
	}//end of if n_non_zero>1
	else {
	}
      
      }//end of loop over instances
    }//matchdata loop
  }

  /**
   * @brief provide missing pid labels to true cosmic pixels
   *
   * truth for corsika tracks are missing the PID/segment labels (of course).
   * but they do have instance and ancestor labels.
   * setting track labels based on presence in pgraph.
   * setting shower labels based on missing presence in pgraph.
   *
   */
  void TripletTruthFixer::_label_cosmic_pid( PrepMatchTriplets& tripmaker,
                                             ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                             larcv::IOManager& iolcv )
  {

    // we loop over tpcs
    for ( auto& matchdata : tripmaker._match_triplet_v ) {    
    
	int nrelabels = 0;
	for ( size_t itrip=0; itrip<matchdata._instance_id_v.size(); itrip++ ) {
	  int trueflow = matchdata._truth_v[itrip];
	  if ( trueflow==0 )
	    continue;
      
	  int iid = matchdata._instance_id_v[itrip]; // instance id
	  int aid = matchdata._ancestor_id_v[itrip]; // ancestor id

	  // look for track id in graph
	  bool has_graphnode = false;
	  ublarcvapp::mctools::MCPixelPGraph::Node_t* pNode_instance = mcpg.findTrackID( iid );
	  if ( pNode_instance!=nullptr && pNode_instance->tid==iid )
	    has_graphnode = true;

	  if ( has_graphnode && pNode_instance->origin==2) {

	    nrelabels++;
	    int relabel = 0;
        
	    // fix up cosmic only
	    switch ( pNode_instance->pid ) {
	    case 13:
	    case -13:
	      relabel = larcv::kROIMuminus;
	      break;
	    case 2212:
	    case 2112:
	      relabel = larcv::kROIProton;
	      break;
	    case 11:
	    case -11:
	      relabel = larcv::kROIEminus;
	      break;
	    case 22:
	      relabel = larcv::kROIGamma;
	      break;
	    case 111:
	      relabel = larcv::kROIPizero;
	      break;
	    case 321:
	    case -321:          
	      relabel = larcv::kROIKminus;
	      break;
	    default:
	      relabel = larcv::kROIPiminus;
	      break;
	    }

	    //std::cout << "cosmic relabel: from [" << tripmaker._pdg_v[itrip] << "] to [" << relabel << "]" << std::endl;
	    matchdata._pdg_v[itrip] = relabel;
	    continue; // got a good label. move on.
	  }

	  // no graph node for track id, but node for ancestor
	  bool has_ancestornode = false;
	  ublarcvapp::mctools::MCPixelPGraph::Node_t* pNode_ancestor = mcpg.findTrackID( aid );
	  if ( pNode_ancestor!=nullptr && pNode_ancestor->tid==aid )
	    has_ancestornode = true;
	  
	  if ( has_ancestornode && pNode_ancestor->origin==2 ) {
	    nrelabels++;
	    int relabel = 0;        
	    // fix up cosmic-secondary
	    switch ( pNode_ancestor->pid ) {
	    case 13:
	    case -13:
	      relabel = larcv::kROICosmic; //delta
	      break;
	    case 2212:
	    case 2112:
	    case 321:
	    case -321:
	      relabel = larcv::kROIProton;
	      break;
	    case 11:
	    case -11:
	      relabel = larcv::kROIEminus;
	      break;
	    case 22:
	    case 111:          
	      relabel = larcv::kROIGamma;
	      break;
	    default:
	      relabel = larcv::kROICosmic;
	      break;
	    }
	    
	    //std::cout << "cosmic relabel: from [" << tripmaker._pdg_v[itrip] << "] to [" << relabel << "]" << std::endl;
	    matchdata._pdg_v[itrip] = relabel;
	    continue;// ok got the label    
	  }
	}//end of triplet loop

	LARCV_NORMAL() << "=== (cryoid,tpcid)=(" << matchdata._cryoid << "," << matchdata._tpcid << ") =======" << std::endl;	
	LARCV_NORMAL() << "  number of cosmic-relables: " << nrelabels << std::endl;	
    }// matchdata loop over tpc data
    
    
  }

  void TripletTruthFixer::_reassign_showers_along_tracks( PrepMatchTriplets& tripmaker,
							  larlite::event_mctrack& ev_mctrack,
							  larlite::event_mcshower& ev_mcshower,
							  std::vector< float >& vtx )
  {

    const int vertex_dist = 1000.0;
    const int min_r_threshold = 1.0;
    
    for ( auto& matchdata : tripmaker._match_triplet_v ) {
      
      // we limit things to near the neutrino, vertex, else too much.
      std::vector<int> near_vertex_list_v;
      near_vertex_list_v.reserve( 1000 );
      for (int idx=0; idx<(int)matchdata._triplet_v.size(); idx++) {
	// if ( matchdata._instance_id_v[idx]==0
	//      || matchdata._truth_v[idx]==0 ) {
	if ( matchdata._truth_v[idx]==0 ) {
	  continue;
	}
	
	int pdg = matchdata._pdg_v[idx];
	// if ( pdg!=(int)larcv::kROIEminus && pdg!=larcv::kROIGamma && pdg!=0)
	//   continue;
	
	auto const& pos = matchdata._pos_v[idx];
	float dist =0.;
	for (int v=0; v<3; v++) {
	  dist += (pos[v]-vtx[v])*(pos[v]-vtx[v]);
	}
	dist = sqrt(dist);
	if ( dist<vertex_dist ) {
	  near_vertex_list_v.push_back( idx );
	}
      }
      
      if ( near_vertex_list_v.size()==0 )
	return;
      
      int nconverted = 0;    
      ublarcvapp::mctools::TruthTrackSCE converter( getSCE() );
      std::cout << "converter SCE: " << getSCE() << std::endl;

      std::map<int,float> idx2energy;
      std::map<int,float> idx2rmin;      
      
      for (auto& mctrack : ev_mctrack ) {

	if (mctrack.Origin()!=1)
	  continue; // nu only
	if (mctrack.PdgCode()==2112)
	  continue; // no neutrons
      
	larlite::track scetrack = converter.applySCE( mctrack );

	// find points near the line segment
	for (auto const& idx : near_vertex_list_v ) {
	  auto const& pt = matchdata._pos_v[ idx ];
	  int instanceid = matchdata._instance_id_v[idx];
	  // if ( instanceid==mctrack.TrackID() )
	  //   continue;
	
	  float min_r = 1.0e9;
	  int min_step = -1;
	  // if ( mctrack.TrackID()==1499515 )
	  //   converter.set_verbosity(larcv::msg::kDEBUG);
	  converter.dist2track( pt, scetrack, min_r, min_step );
	  //converter.set_verbosity(larcv::msg::kNORMAL);
	  
	  // if ( fabs(mctrack.PdgCode())==211 || mctrack.PdgCode()==2212 ) {
	  // }
	  if ( min_r<min_r_threshold ) {
	    // check for conversion
	    auto it = idx2rmin.find(idx);
	    if ( it==idx2rmin.end() ) {
	      idx2rmin[idx] = 1000.0;
	      it = idx2rmin.find(idx);
	    }
	    if ( min_r<it->second ) {	      
	      matchdata._instance_id_v[idx] = mctrack.TrackID();
	      int newpid = _pdg_to_larcvpid( mctrack.PdgCode() );
	      int origpid = matchdata._pdg_v[idx];
	      if ( newpid!=0 ) { 
		matchdata._pdg_v[idx] = newpid;
	      }
	      nconverted++;
	      // std::cout << "mctrack[" << mctrack.TrackID() << "] "
	      // 		<< " pdg=" << mctrack.PdgCode()
	      // 		<< " min_r=" << min_r
	      // 		<< " idx=" << idx
	      // 		<< " pt=(" << pt[0] << "," << pt[1] << "," << pt[2] << ")";
	      // if ( min_step>=0 )
	      // 	std::cout << " minstep=(" << mctrack.at(min_step).X() << "," << mctrack.at(min_step).Y() << "," << mctrack.at(min_step).Z() << ")";
	      // std::cout << " -- converted to from " << origpid << " to " << matchdata._pdg_v[idx] << std::endl;		
	      idx2rmin[idx] = min_r;
	    }
	  }
	  
	}
      }//end of mctrack loop
      std::cout << "TripletTruthFixer::converted " << nconverted << " hits out of " << near_vertex_list_v.size() << " near vertex pixels" << std::endl;
    }//end of matchdata loop
  }

  int TripletTruthFixer::_pdg_to_larcvpid( int pdg ) const
  {
    int max_pid = 0;
    switch( pdg ) {
    case 2212:
    case 2112:            
      max_pid = larcv::kROIProton;
      break;
    case 13:
    case -13:
      max_pid = larcv::kROIMuminus;
      break;
    case 321:
    case -321:          
      max_pid = larcv::kROIKminus;
      break;
    case 211:
    case -211:
      max_pid = larcv::kROIPiminus;
      break;
    default:
      max_pid = 0;
      break;
    }
    return max_pid;
  }
  
}
}
