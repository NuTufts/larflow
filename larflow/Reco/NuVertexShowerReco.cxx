#include "NuVertexShowerReco.h"

#include "larlite/DataFormat/mctruth.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/MCPos2ImageUtils.h"

#include "geofuncs.h"
#include "cluster_functions.h"

namespace larflow {
namespace reco {

  NuVertexShowerReco::NuVertexShowerReco()
    : larcv::larcv_base("NuVertexShowerReco"),
      _mcpg(nullptr),
      _trunk_maxdist_from_closest_cm(10.0),
      _calc_cosmic_overlap(true),
      _boosterhandle(nullptr)
  {
    std::cout << "CREATE XGBOOST HANDLER" << std::endl;
    _boosterhandle = new BoosterHandle;
    nuvertexshowerreco_safe_xgboost( XGBoosterCreate(NULL, 0, _boosterhandle) );
    
    std::string model_path = std::string( std::getenv("LARFLOW_BASEDIR") )+"/larflow/Reco/data/nuvertexshowerreco_showertrunk_bdt.model";
    std::cout << "LOAD MODEL PARAMETERS" << std::endl;
    std::cout << "path: " << model_path << std::endl;
    try {
      nuvertexshowerreco_safe_xgboost(XGBoosterLoadModel( *_boosterhandle, model_path.c_str()));
    }
    catch ( std::exception& err ) {
      std::stringstream ss;
      ss << "[NuVertexShowerReco::NuVertexShowerReco] Error when trying to load model. Check if BDT model file is in larflow/Reco/data folder." << std::endl;
      ss << err.what() << std::endl;
      throw std::runtime_error( ss.str() );
    }
    
  }

  NuVertexShowerReco::~NuVertexShowerReco()
  {
    if ( _boosterhandle ) {
      nuvertexshowerreco_safe_xgboost(XGBoosterFree(*_boosterhandle));
      delete _boosterhandle;
    }
  }
  
  
  /**
   * @brief process data from one event
   *
   * @param[in] iolcv LArCV IOManager containing event data
   * @param[in] ioll  larlite storage_manager containing event data
   * @param[inout] nu_candidate_v List of neutrino vertex candidates to which we will append shower objects
   * @param[inout] nu_cluster_book_v Book-keeping struct, one for each neutrino vertex candidate. tracks how clusters are used.
   * 
   */
  void NuVertexShowerReco::process( larcv::IOManager& iolcv,
                                    larlite::storage_manager& ioll,
                                    std::vector<NuVertexCandidate>& nu_candidate_v,
				                            std::vector<ClusterBookKeeper>& nu_cluster_book_v )
  {

    loadClusters(ioll);

    if ( _mc_analysis_mode ) {
      // if we do MC analysis to study/tune this algorithm,
      //   we need to determine which is the closest neutrino vertex candidate to the real vertex
      // get true position of neutrino
      larlite::event_mctruth* ev_mctruth =
	      (larlite::event_mctruth*)ioll.get_data(larlite::data::kMCTruth,"generator");

      std::vector<float> true_nu_vtx_pos(4,0); // the true nu interaction position
      std::vector<float> sce_nu_vtx_pos(4,0);  // the observable nu interaction position, after space charge effects and drift

      const larlite::mctruth& mct = ev_mctruth->front();
      true_nu_vtx_pos[0] = mct.GetNeutrino().Nu().Trajectory().front().X();
      true_nu_vtx_pos[1] = mct.GetNeutrino().Nu().Trajectory().front().Y();
      true_nu_vtx_pos[2] = mct.GetNeutrino().Nu().Trajectory().front().Z();
      true_nu_vtx_pos[3] = mct.GetNeutrino().Nu().Trajectory().front().T();

      // convert to apparent position
      sce_nu_vtx_pos = ublarcvapp::mctools::MCPos2ImageUtils::Get()->get_sce_shifted_pos( true_nu_vtx_pos[0],
										  true_nu_vtx_pos[1],
										  true_nu_vtx_pos[2] );
			sce_nu_vtx_pos.resize(4,0);
      sce_nu_vtx_pos[3] = true_nu_vtx_pos[3];

      LARCV_DEBUG() << "true vtx (" << true_nu_vtx_pos[0] << ","
                    << true_nu_vtx_pos[1] << ","
                    << true_nu_vtx_pos[2] << ","
                    << true_nu_vtx_pos[3] << ")" << std::endl;
      LARCV_DEBUG() << "sce vtx (" << sce_nu_vtx_pos[0] << ","
                    << sce_nu_vtx_pos[1] << ","
                    << sce_nu_vtx_pos[2] << ","
                    << sce_nu_vtx_pos[3] << ")" << std::endl;

      // find the closest vertex
      _mcana_index_closest_recovtx = -1;
      _mcana_closest_recovtx_dist = 1.0e9;
      for ( int ivtx=0; ivtx<(int)nu_candidate_v.size(); ivtx++ ) {
	      auto& nuvtx = nu_candidate_v.at(ivtx);
	      float dist = 0.;
      	for (int ii=0; ii<3; ii++) {
	        dist += ( nuvtx.pos[ii]-sce_nu_vtx_pos[ii] )*( nuvtx.pos[ii]-sce_nu_vtx_pos[ii] );
	      }
	      dist = sqrt(dist);
	      if ( dist < _mcana_closest_recovtx_dist ) {
	        _mcana_index_closest_recovtx = ivtx;
          _mcana_closest_recovtx_dist = dist;
	      }
      }
      LARCV_DEBUG() << "Closest reco neutrino vertex: "
		    << " index=" << _mcana_index_closest_recovtx
		    << " closest distance=" << _mcana_closest_recovtx_dist
		    << std::endl;
    }


    for ( size_t ivtx=0; ivtx<nu_candidate_v.size(); ivtx++) {
      auto& nuvtx = nu_candidate_v.at(ivtx);
      auto& book  = nu_cluster_book_v.at(ivtx);
      LARCV_DEBUG() << "Build Vertex Showers: (" << nuvtx.pos[0] << "," << nuvtx.pos[1] << "," << nuvtx.pos[2] << ")" << std::endl;
      _mc_analysis_saveinfo_for_this_vertex = false; // default to false
      if ( _mc_analysis_mode && ivtx==_mcana_index_closest_recovtx ) {
        // if the nu vtx is the closest qualifying vertex, then do the analysis
        _mc_analysis_saveinfo_for_this_vertex = true;
      }
      build_vertex_showers( nuvtx, book, iolcv, ioll );
    }
    
  }

  /**
   * @brief collect the clusters to be used
   *
   */
  void NuVertexShowerReco::loadClusters( larlite::storage_manager& ioll )
  {
    // load up the clusters
    LARCV_INFO() << "Number of cluster producers: " << _cluster_producers.size() << std::endl;
    _showercluster_candidates_v.clear();

    for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
      LARCV_INFO() << "Load cluster data with tree name[" << it->first << "]" << std::endl;
      it->second = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, it->first );
      auto it_pca = _cluster_pca_producers.find( it->first );
      if ( it_pca==_cluster_pca_producers.end() ) {
        _cluster_pca_producers[it->first] = nullptr;
        it_pca = _cluster_pca_producers.find( it->first );
      }
      it_pca->second = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, it->first );
      LARCV_INFO() << "clusters from [" << it->first << "]: " << it->second->size() << " clusters" << std::endl;

      // store in container
      for (int icluster=0; icluster<(int)it->second->size(); icluster++) {
        NuVertexCandidate::VtxCluster_t showercluster;
        showercluster.producer = it->first;
        showercluster.type = _cluster_type[ it->first ];
        showercluster.index = icluster;
        _showercluster_candidates_v.push_back( showercluster );
      }
      
    }

    LARCV_INFO() << "Number of clusters registered: " << _showercluster_candidates_v.size() << std::endl;
  }    

  /**
   * @brief [internal] build showers for given neutrino candidate vertex
   *
   * we loop through shower prongs associated to the vertex.
   * first is to look for mislabeled track components along line between vertex and shower start.
   * then we absorb nearby fragments within a cone of the closest fragment
   * 
   * @param[in] nuvtx Neutrino candidate vertex
   * @param[inout] nuclusterbook Tracks how the track and shower clusters in the event are used by the neutrino vertex candidate.
   * @param[in] iolcv LArCV Event data
   * @param[in] ioll  larlite event data
   *
   */
  void NuVertexShowerReco::build_vertex_showers( NuVertexCandidate& nuvtx,
						 ClusterBookKeeper& nuclusterbook,
						 larcv::IOManager& iolcv, 
						 larlite::storage_manager& ioll ) 
  {
    // We build shower prongs using the neutrino vertex as a seed.
    // The neutrino vertex provides a guide for which shower fragment might be
    //  the beginning or "trunk" of a shower. The vertex also provides guidance
    //  as to what direction the shower is flowing, i.e. the shower should point
    //  away from the neutrino vertex.
    // To start, we need to gather information on potential shower prongs.
    //  These prongs represent the beginning of a shower.
    // We use the following struct to represent shower prongs and to store
    //  various information about them.
    // The information in the struct is used to sort the prongs and
    //  set the priority for which prongs will seed the beginning of a shower.


    // these are parameters controlling how the shower prongs are formed and built
    // we need to optimize them
    const float r_mollier = 9.04; // cm, liquid argon
    const float r_trunk   = 3.0;
    const float tau_startpt = 3.0; // cm
    const float max_showerpt_dist = 200.0;
    const float max_showerpt_d2 = max_showerpt_dist*max_showerpt_dist;
    
    // If we perform MC analysis to tune how this code works, we first need to compile info 
    //  on true shower prongs. We want to know how well our reco prongs overlap and reconstruct
    //  the true shower prongs.
    
    if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {

      if ( _mcpg )
	      delete _mcpg;

      LARCV_DEBUG() << " INITIALIZE MC ANALYSIS FOR SHOWER RECO STUDY: build MCPixelPGraph" << std::endl;
      
      // we run the MCPixelPGraph to get truth information
      _mcpg = new ublarcvapp::mctools::MCPixelPGraph();
      _mcpg->set_verbosity( "info" );
      _mcpg->buildgraph( iolcv, ioll );

      // initialization: clear container for ShowerRecoInfo_t
      _map_prongindex_to_mcanainfo.clear();
      LARCV_DEBUG() << "MCPG and MC Analysis Mode Ready." << std::endl;
      
    }//end of mcanalysis mode: initialization, finding the vertex to evaluate
    

    // the following is a container to hold the prongs
    // we will sort this container later
    std::vector<ProngRank_t> seed_rank_v;
    
    //for ( int iprong=0; iprong<(int)nuvtx.cluster_v.size(); iprong++) {
    // loop over entire shower cluster set in the event
    for ( int iprong=0; iprong<(int)_showercluster_candidates_v.size(); iprong++) {
        
      auto const& vtxcluster = _showercluster_candidates_v.at(iprong);
      // -log(exp[-r/tau]) = r/tau
      
      // only deal with showers
      if ( vtxcluster.type!=NuVertexCandidate::kShower && vtxcluster.type!=NuVertexCandidate::kShowerKP ) {
        continue;
      }

      // check to make sure we aren't testing a duplicate cluster
      bool found = false;
      //std::cout << "check seed: " << vtxcluster.producer << " " << vtxcluster.index << std::endl;
      for ( auto& seed : seed_rank_v ) {
        //std::cout << " past seed: " << seed.producer << " " << seed.container_idx << std::endl;
        if ( seed.producer==vtxcluster.producer && seed.container_idx==vtxcluster.index )
          found = true;
        if ( found )
          break;
      }

      if ( found ) {
        LARCV_INFO() << "ShowerProng[" << iprong << "] is a cluster duplicate "
                    << "(" << vtxcluster.producer << ", " << vtxcluster.index << ")"
                    << std::endl;
        continue;
      }

      // get the cluster of hits from the event container
      // (note: who made these?)
      const larlite::larflowcluster& lfcluster =
        ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );

      // if we are running the MC analysis, we try to match this prong to a true shower trunk
      if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
        LARCV_INFO() << "run mcanalysis for prong" << std::endl;
        // convert larlite::cluster into larflow::reco::cluster_t
        larflow::reco::cluster_t showercluster;
        showercluster.points_v.reserve( lfcluster.size() );
        for (int ii=0; ii<(int)lfcluster.size(); ii++) {
          std::vector<float> pt = { lfcluster[ii][0], lfcluster[ii][1], lfcluster[ii][2] };
          showercluster.points_v.push_back( pt );
        }
        NuVertexShowerReco::RecoShowerInfo_t showerinfo;
        _gatherTruthShowerFeatures( showercluster, nuvtx, showerinfo );
        _map_prongindex_to_mcanainfo[ iprong ] = showerinfo;
      }
      

      // define shower start, dir, ll-score
      std::vector<float> shower_start;
      std::vector<float> shower_dir;
      float shower_ll = 0.0;
      int ntrunk_clusters = _make_trunk_cand( nuvtx.pos,
                            lfcluster,
                            shower_start,
                            shower_dir,
                            shower_ll );

      if (ntrunk_clusters==0){
        // the shower cluster wasn't well-formed enough to return a trunk
        LARCV_INFO() << "ShowerProng[" << iprong << "] cannot build trunk." << std::endl;
        if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
          _map_prongindex_to_mcanainfo[ iprong ]._reco_outcome = kFailPreCuts;
          LARCV_INFO() << " prong ground truth: " <<  _map_prongindex_to_mcanainfo[ iprong ]._correct_outcome << std::endl;
        }
        continue; // to next shower cluster prong candidate
      }

      // we want to calculate parameters for deciding to use as a starting shower prong
      // maybe we train an xgboost model to turn several variables into a single score

      // making an artifical end point (probably should use pca-end points)
      std::vector<float> shower_end(3,0);
      for (int i=0; i<3; i++)
        shower_end[i] = shower_start[i] + 10.0*shower_dir[i];
      
      // // define shower axis -- start point to vertex
      std::vector<float> axis(3,0);
      float a_dist = 0.;
      for (int i=0; i<3; i++) {
        axis[i] = shower_start[i]-nuvtx.pos[i];
        a_dist += axis[i]*axis[i];
      }
      if ( a_dist>0 ) {
        a_dist = sqrt(a_dist);
        for (int i=0; i<3; i++)
          axis[i] /= a_dist;
      }

      // impact parameter
      float b_impact_par = larflow::reco::pointLineDistance<float>( shower_start, shower_end, nuvtx.pos );

      // cosine between axis and shower_dir
      float c_cosine = 0.;
      for (int v=0; v<3; v++) {
        c_cosine += axis[v]*shower_dir[v];
      }

      // get the pixel sum for the cluster
      larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
      std::vector<float> cluster_pixsum_v = _get_cluster_pixsum( ev_adc->as_vector(), lfcluster );
      std::vector<float> cluster_cosmic_pixsum_v(3,0.);

      if ( _calc_cosmic_overlap ) {
	larcv::EventImage2D* ev_thrumu = nullptr;
	try {
	  ev_thrumu = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"thrumu");
	  cluster_cosmic_pixsum_v = _get_cluster_pixsum( ev_thrumu->as_vector(), lfcluster );
	}
	catch (std::exception& err) {
	  // pass
	}
      }
      
      // how to choose pixsum to eval?
      float d_pixsum = cluster_pixsum_v[2]*0.0162;
      if ( d_pixsum < 1.0 ) {
        d_pixsum = ( cluster_pixsum_v[0] > cluster_pixsum_v[1] ) ? cluster_pixsum_v[0]*0.0162 : cluster_pixsum_v[1]*0.0162;
      }

      // combine cosmic contributions max ratio to plane
      float e_cosmic = 0.;
      for (int p=0; p<3; p++) {
	e_cosmic += cluster_cosmic_pixsum_v[p]*0.0162/3.0;
      }
      if ( d_pixsum>0.0 )
	e_cosmic /= d_pixsum;

      // update the mc ana info
      if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
        auto it_mcana = _map_prongindex_to_mcanainfo.find( iprong );
        if ( it_mcana!=_map_prongindex_to_mcanainfo.end() ) {
          auto& mcana_info = it_mcana->second;
          mcana_info._recoshower_dist2vtx  = a_dist;
          mcana_info._recoshower_impactpar = b_impact_par;
          mcana_info._recoshower_cosine    = c_cosine;
          mcana_info._recoshower_pixsum_MeV = d_pixsum;
	  mcana_info._recoshower_cosmic_pixsum = e_cosmic;
          mcana_info._recoshower_trunkdir = std::vector<float>{ 0, 0, 0};
          for (int v=0; v<3; v++)
            mcana_info._recoshower_trunkdir[v] = shower_dir[v];
        }
      }

      // std::vector<float> axis_start(3,0);
      // std::vector<float> axis_end(3,0);      
      // for (int i=0; i<3; i++) {
      //   axis_start[i] = vtxcluster.pos[i];
      //   axis_end[i]   = vtxcluster.pos[i] + 30.0*axis[i];
      // }

      // float score_ll = 0;
      // // -log(P(r)*P(s))
      // // P(r): distance from axis falls off as exp
      // // P(s): projection along axis. exp penalty for being behind vertex
      // // only deal with showers
      
      // for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      //   auto const& hit = lfcluster[ihit];
      //   std::vector<float> pt(3,0);
      //   for (int i=0; i<3; i++)
      //     pt[i] = hit[i];
        
      //   float r = pointLineDistance3f( axis_start, axis_end, pt );
      //   float s = pointRayProjection3f( axis_start, axis, pt );

      //   score_ll += r/r_mollier;
      //   if (s<0)
      //     score_ll += -s/tau_startpt;
      // }
      // if ( lfcluster.size()>0 )
      //   score_ll /= float(lfcluster.size());

      // simplified score just based on distance to vertex
      float score_ll = 0;
      if ( lfcluster.size()>10 ) {
        score_ll = a_dist;
      }
      else {
        score_ll = 10000.0 + a_dist; // blerg
      }

      ProngRank_t rank( vtxcluster.producer, iprong, vtxcluster.index, score_ll );
      rank.axis = shower_dir;
      rank.axis_start = shower_start;
      rank.axis_end   = shower_end;
      rank.dist2vtx   = a_dist;
      rank.impactpar  = b_impact_par;
      rank.cosine     = c_cosine;
      rank.pixsum     = d_pixsum;
      rank.cosmic     = e_cosmic;

      seed_rank_v.push_back( rank );
    }//end of loop over prong

    // use information for each prong (in ProngRank_t) to run BDT
    // and get score for use in building showers
    getBDTseedscore( seed_rank_v );

    LARCV_INFO() << "===============================" << std::endl;
    LARCV_INFO() << " Start Building Showers" << std::endl;
    LARCV_INFO() << "===============================" << std::endl;

    // notes
    // (1) first isolate prongs as those that pass quality cut
    // (2) then sort quality prongs by distance
    // (3) then allow it to absorb shower hit clusters
    struct ProngDistanceSorter_t {
      int index;
      float distance;
      ProngDistanceSorter_t( int idx, float dist )
	: index(idx),
	  distance(dist)
      {};
      bool operator<( ProngDistanceSorter_t& rhs ) {
	if ( distance < rhs.distance )
	  return true;
	return false;
      };
    };
    std::vector< ProngDistanceSorter_t > sort_by_distance;
    sort_by_distance.reserve( seed_rank_v.size() );
    for (int idx=0; idx<(int)seed_rank_v.size(); idx++) {
      sort_by_distance.push_back( ProngDistanceSorter_t(idx,seed_rank_v.at(idx).dist2vtx) );
    }
    std::sort( sort_by_distance.begin(), sort_by_distance.end() );

    // now we can begin to build out a shower
    // we need to track which clusters we used up
    std::vector<int> prong_used_v( _showercluster_candidates_v.size(), 0 );
    // the other clusters
    std::map< std::string, std::vector<int> > cluster_used_v;
    for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++  ) {
      if ( it->second!=nullptr )
        cluster_used_v[it->first] = std::vector<int>( it->second->size(), 0 );
    }

    int num_seeds_defined = 0;
    
    // loop through the seeds, sorted by smallest distance to last, and start building showers
    for ( int idist=0; idist<(int)sort_by_distance.size(); idist++ ) {
      
      auto const& prong_by_distance = sort_by_distance.at(idist);

      auto& rankedprong = seed_rank_v.at( prong_by_distance.index );
      int prongidx = rankedprong.prong_idx;
      
      // if prong used skip it
      if ( prong_used_v[prongidx]!=0 ) {
        continue;
      }

      // decide if this is going to be a seeding prong
      bool passes = false;
      if ( rankedprong.cosmic < 0.5 ) {
	// non-cosmic seeds
	// be more confident for small showers
	if ( rankedprong.pixsum<50.0 && rankedprong.score>=0.0 )
	  passes = true;
	// be more open for large showers
	if ( rankedprong.pixsum>=50.0 && rankedprong.score>=-3.0 )
	  passes = true; // loose cut, rely on prong CNN to reject garbage
      }
      else {
	// cosmic seeds: requires more confident prong score for both large and small showers
	if ( rankedprong.score>=0.0 )
	  passes = true;
      }

      if ( passes )
	num_seeds_defined++;

      // use the cluster to seed
      auto const& vtxcluster = _showercluster_candidates_v.at(prongidx);

      // if doing mcanalysis, get it's ShowerRecoInfo_t struct
      NuVertexShowerReco::RecoShowerInfo_t* pmcinfo = nullptr;
      if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
        auto it_mcana = _map_prongindex_to_mcanainfo.find( prongidx );
        if ( it_mcana!=_map_prongindex_to_mcanainfo.end() )
          pmcinfo = &(it_mcana->second);
      }

      // check we havent already absorbed the cluster already
      if ( cluster_used_v.find( vtxcluster.producer )!=cluster_used_v.end() ) {
        if ( cluster_used_v[vtxcluster.producer].at( vtxcluster.index )!=0 ) {
          // looks like the cluster is already being used
          if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex && pmcinfo ) {
            pmcinfo->_reco_outcome = kSubCluster;
          }
          continue;
        }
      }

      LARCV_INFO() << "------------------------------------------------------------" << std::endl;
      LARCV_INFO() << "ShowerProng[" << prongidx << "] proposed as shower start." << std::endl;
      LARCV_INFO() << "   dist2vtx: " << rankedprong.dist2vtx << " cm" << std::endl;
      LARCV_INFO() << "   impactpar: " << rankedprong.impactpar << " cm" << std::endl;
      LARCV_INFO() << "   cosine: " << rankedprong.cosine << std::endl;
      LARCV_INFO() << "   pixsum: " << rankedprong.pixsum << " MeV-ish" << std::endl;
      LARCV_INFO() << "   cosmic: " << rankedprong.cosmic << std::endl;
      LARCV_INFO() << "   score (bdt logit): " << rankedprong.score << std::endl;

      
      // if ( rankedprong.score>-4.0 ) {
      // 	passes = true;
      // if ( rankedprong.pixsum<10.0 ) {
      //   // reject, minimum threshold?
      //     LARCV_INFO() << "  fails minimum energy" << std::endl;	
      //   if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
      //     pmcinfo->_reco_outcome = kFailPreCuts;
      //   }
      //   continue;
      //       }
      //       if ( rankedprong.dist2vtx > 50.0 && rankedprong.pixsum<50.0 ) {
      //   // reject, too far
      //   LARCV_INFO() << "  fail precut: shower fragments < 50 MeV must be within 50 cm" << std::endl;	
      //   if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
      //     pmcinfo->_reco_outcome = kFailPreCuts;
      //   }
      //   continue;
      // }
      if ( passes ) {
	LARCV_INFO() << "  ** accepted as prong seed" << std::endl;
      }
      else {
	LARCV_INFO() << "  ( rejected as prong seed )" << std::endl;	
      }

      if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
        pmcinfo->_reco_outcome = kAccept; // as starting prong
        LARCV_INFO() << "  ** Ground Truth=" << pmcinfo->_correct_outcome << std::endl;
        LARCV_INFO() << "     - trackid: " << pmcinfo->_trueprong_trackid << std::endl; 
        LARCV_INFO() << "     - true prong completeness: " << pmcinfo->_frac_truetrunk << std::endl; 
        LARCV_INFO() << "     - reco purity: " << pmcinfo->_frac_recopurity << std::endl;
      }
      LARCV_INFO() << "------------------------------------------------------------" << std::endl;
      

      if ( !passes ) {
	// reject
	continue;
      }

      // using cluster as seed
      cluster_used_v[vtxcluster.producer][vtxcluster.index] = 1;
      prong_used_v[prongidx] = 1;
      
      // get the shower cluster
      const larlite::larflowcluster& lfcluster =
        ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, vtxcluster.producer))->at( vtxcluster.index );

      // LARCV_INFO() << "ShowerProng[" << vtxcluster.producer << "," << rankedprong.container_idx << ", prong " << prongidx << "] "
      //               << " score=" << rankedprong.score
      //               << " npts=" << lfcluster.size()
      //               << std::endl;

      
      // This cluster will become the basis for a potential new shower prong.
      // The container, shower_hit_v, below will represent this new shower object, 
      //   which is a cluster of 3d hits
      larlite::larflowcluster shower_hit_v;
      // absorb hits into shower_hit_v
      for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
        shower_hit_v.push_back( lfcluster[ihit] );
      }

      // loop over TRACK clusters, find those along the shower axis.
      // we are assuming this is ssnet mislabeling
      int ntrunk_hits_added = 0;
      std::vector<float> track_s_v;
      larlite::larflowcluster trunk_hit_v;
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
        if ( _cluster_type[it->first]==NuVertexCandidate::kTrack ) {
          // loop over track cluster in this event container
          for ( auto const& track_lfcluster : *it->second ) {
            // track cluster
            int nclose_to_axis = 0;
            for (auto const& trackhit : track_lfcluster ) {
              std::vector<float> trackpt = { (float)trackhit[0], (float)trackhit[1], (float)trackhit[2] };
              float r = pointLineDistance3f(  rankedprong.axis_start, rankedprong.axis_end, trackpt );
              float s = pointRayProjection3f( rankedprong.axis_start, rankedprong.axis, trackpt );
              float vtxdist = 0.;
              for (int i=0; i<3; i++)
                vtxdist += ( trackpt[i]-nuvtx.pos[i] )*( trackpt[i]-nuvtx.pos[i] );

              if ( s>-rankedprong.dist2vtx && s<0 && ((vtxdist<1.0 && r<0.5) || (vtxdist>0.0 && r<2.0)) )  {
                trunk_hit_v.push_back( trackhit );

                float trunk_s = pointRayProjection3f( nuvtx.pos, rankedprong.axis, trackpt );
                track_s_v.push_back( trunk_s );
                
                ntrunk_hits_added++;
              }
              
            }
            
          }          
          
        }
      }
      track_s_v.push_back( 0 );      
      track_s_v.push_back( rankedprong.dist2vtx );
      std::sort( track_s_v.begin(), track_s_v.end() );
      float max_gap_s = 0;
      for (int i=1; i<(int)track_s_v.size(); i++) {
        //std::cout << " gap: " << track_s_v[i] << "-" << track_s_v[i-1] << " = " << fabs(track_s_v[i]-track_s_v[i-1]) << std::endl;
        if ( max_gap_s < fabs(track_s_v[i]-track_s_v[i-1]) )
          max_gap_s = fabs(track_s_v[i]-track_s_v[i-1]);
      }
      LARCV_DEBUG() << "Number of TRACK hits found: " << ntrunk_hits_added << " max gap=" << max_gap_s << std::endl;      
      if ( max_gap_s<3.0 ) {
        for (auto& hit : trunk_hit_v )
          shower_hit_v.push_back(hit);
      }
      
      // absorb shower clusters within cone. we sample from all producers given, not just within vertex.
      // we call the cluster we are testing to add as a "subcluster"
      for (int jj=idist+1; jj<sort_by_distance.size(); jj++) {

	// by starting at index jj=idist+1 in the sort_by_distance container,
	// we only add clusters further from the vertex
	
	auto& subprong = seed_rank_v.at( sort_by_distance.at(jj).index );

	int sub_prongidx = subprong.prong_idx; // we track usage with this index (refers to position in seed_rank_v)
	auto const& sub_showercluster = _showercluster_candidates_v.at(sub_prongidx);
	std::string sub_producer = sub_showercluster.producer;
	int sub_index = sub_showercluster.index; // we access larflow cluster and its hits with this index

	// don't absorb points from seeding cluster
	// vtxcluster is a struct containing info from the seeding prong shower
	if ( vtxcluster.producer==sub_producer && vtxcluster.index==sub_index )
	  continue;

	// don't absorb points from previous used cluster
	if ( prong_used_v[sub_prongidx]==1 )
	  continue;
	
	//for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
	
        auto const& cluster_type = _cluster_type[sub_producer];
        if ( cluster_type==NuVertexCandidate::kShowerKP ||
             cluster_type==NuVertexCandidate::kShower ) {

	  // the cluster is a "shower type"

	  // get the cluster
	  larlite::event_larflowcluster* sub_cluster_v = _cluster_producers[sub_producer];	  
	  auto const& shower_lfcluster = (*sub_cluster_v).at(sub_index);
	  
	  // skip zero clusters
	  if ( shower_lfcluster.size()==0 )
	    continue;
            
	  // make sure its not the clsuter we are using as the seed
	  int nhits_within_cone = 0;
	  for ( auto const& showerhit : shower_lfcluster ) {
	    std::vector<float> showerpt = { showerhit[0], showerhit[1], showerhit[2] };
	    float r = pointLineDistance3f(  rankedprong.axis_start, rankedprong.axis_end, showerpt );
	    float s = pointRayProjection3f( rankedprong.axis_start, rankedprong.axis,     showerpt );
	    
	    // set max distance from prong start to the point in question
	    float d2 = 0.;
	    for (int i=0; i<3; i++)
	      d2 += ( rankedprong.axis_start[i]-showerpt[i] )*( rankedprong.axis_start[i]-showerpt[i] );

	    if ( s>0.0 && d2<max_showerpt_d2 ) {
	      float rovers = r/s;
	      //if ( rovers < 9.0/14.0 ) {
	      if ( (s<5.0 && r<r_trunk) || (s>=5.0 && r<r_mollier ) ) {
		// mollier/radiation length
		nhits_within_cone++;
	      }
	    }
	  }//end of loop over hits in shower cluster

	  float frac_within_cone = nhits_within_cone/float(shower_lfcluster.size());
	  if ( frac_within_cone>0.5 ) {
	    // add the shower cluster
	    LARCV_INFO() << "Shower(sub)Prong[" << sub_prongidx << "] added to Shower(seed)Prong[" << prongidx << "] "
			 << " frac_within_cone=" << frac_within_cone 
			 << std::endl;
	    for ( auto const& showerhit : shower_lfcluster )
	      shower_hit_v.push_back( showerhit );
	    cluster_used_v[sub_producer][sub_index] =  1;
	    prong_used_v[sub_prongidx] = 1;
	  }//end of if inside cone
        }//end of if cluster is shower type
      }//loop over producers to build showers


      // get pca of final shower
      larflow::reco::cluster_t shower_cluster_t = larflow::reco::cluster_from_larflowcluster(shower_hit_v);
      larflow::reco::cluster_pca( shower_cluster_t );
      larlite::pcaxis shower_hit_pca = larflow::reco::cluster_make_pcaxis( shower_cluster_t );
      
      larlite::track shower_trunk = larflow::reco::cluster_make_trunk( shower_cluster_t, nuvtx.pos );
    
      // larlite::track shower_trunk_dir;
      // shower_trunk_dir.add_vertex( TVector3(rankedprong.axis_start[0],
      //                                       rankedprong.axis_start[1],
      //                                       rankedprong.axis_start[2]) );
      // shower_trunk_dir.add_vertex( TVector3(rankedprong.axis_end[0],
      //                                       rankedprong.axis_end[1],
      //                                       rankedprong.axis_end[2]) );
      // double trunkDir[3] = { rankedprong.axis_end[0] - rankedprong.axis_start[0],
      //                        rankedprong.axis_end[1] - rankedprong.axis_start[1],
      //                        rankedprong.axis_end[2] - rankedprong.axis_start[2] };
      // double trunkMag = sqrt( pow(trunkDir[0],2) + pow(trunkDir[1],2) + pow(trunkDir[2],2) );
      // shower_trunk_dir.add_direction( TVector3(trunkDir[0]/trunkMag,
      //                                          trunkDir[1]/trunkMag,
      //                                          trunkDir[2]/trunkMag) );
      // shower_trunk_dir.add_direction( TVector3(trunkDir[0]/trunkMag,
      //                                          trunkDir[1]/trunkMag,
      //                                          trunkDir[2]/trunkMag) );

      
      // save shower to nuvtx candidate object
      nuvtx.shower_v.emplace_back( std::move(shower_hit_v) );
      nuvtx.shower_trunk_v.emplace_back( std::move(shower_trunk) );
      nuvtx.shower_pcaxis_v.emplace_back( std::move(shower_hit_pca) );

    }//end of seed prong loop
    
    LARCV_INFO() << "Number of showers added to vertex: " << nuvtx.shower_v.size() << std::endl;

    // book the clusters we used
    // loop over pairs of (producer, used vector)
    for ( auto itc=cluster_used_v.begin(); itc!=cluster_used_v.end(); itc++ ) {
      for (size_t idx=0; idx<itc->second.size(); idx++) {
        if ( itc->second[idx]>0 ) {
          // this cluster was used by this vertex
          const larlite::larflowcluster& lfcluster =
            ( (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, itc->first))->at( idx );
          if ( lfcluster.matchedflash_idx>=0 && lfcluster.matchedflash_idx<nuclusterbook.cluster_status_v.size() ) {
            nuclusterbook.cluster_status_v[ lfcluster.matchedflash_idx ] = 1; // book it!
          }
          else {
            LARCV_WARNING() << "Used cluster index outside the cluster book range!" << std::endl;
          }
        }
      }
    }

    if ( _mc_analysis_mode && _mc_analysis_saveinfo_for_this_vertex ) {
      // save the results of the mc analysis records
      _fill_mcanalysis_tree();
    }
    
  }

  /**
   * @brief Define the shower trunk
   *
   * @param[in] pos Position of vertex.
   * @param[in] lfcluster Cluster to find trunk for.
   * @param[out] shower_start Start of defined shower trunk.
   * @param[out] shower_dir   Direction of defined shower trunk.
   * @param[out] shower_ll    Score for choosing best trunk for shower cluster.
   *
   * What we do is fit the points within the cluster closest to the neutrino vertex position (pos).
   * The points we fit for the trunk are no more than an addition 3.5 cm
   *   over the minimum distance of the cluster points to the vertex.
   * We also provide a score for how "straight" the trunk is and how well it fits within a cone.
   *   this has never been tuned ...
   */
  int NuVertexShowerReco::_make_trunk_cand( const std::vector<float>& pos,
                                             const larlite::larflowcluster& lfcluster,
                                             std::vector<float>& shower_start,
                                             std::vector<float>& shower_dir,
                                             float& shower_ll )
  {

    // calculate distance to vertex for every hit in the cluster
    std::vector<float> dist2vertex(lfcluster.size(),0);
    float min_dist = 1e9; // the 
    for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      float dist = 0.;
      for (int i=0; i<3; i++) {
        dist += (lfcluster[ihit][i]-pos[i])*(lfcluster[ihit][i]-pos[i]);
      }
      dist2vertex[ihit] = sqrt(dist);
      if ( min_dist>dist2vertex[ihit] )
        min_dist = dist2vertex[ihit];
    }

    std::vector< std::vector<float> > close_hit_v;
    close_hit_v.reserve( lfcluster.size() );
    
    for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
      if ( dist2vertex[ihit]-min_dist < _trunk_maxdist_from_closest_cm ) {
        std::vector<float> pt = { lfcluster[ihit][0], lfcluster[ihit][1], lfcluster[ihit][2] };
        close_hit_v.push_back( pt );
      }
    }

    std::vector<cluster_t> trunk_cand_v;
    larflow::reco::cluster_spacepoint_v( close_hit_v, trunk_cand_v );
    if ( trunk_cand_v.size()==0 ) {
      return 0;
    }    

    struct CandRank_t {
      int idx;
      float llscore;
      std::vector<float> start;
      std::vector<float> dir;
      CandRank_t( int ii, float ll )
        : idx(ii), llscore(ll)
      {};
      bool operator<( const CandRank_t& rhs ) {
        if ( llscore<rhs.llscore )
          return true;
        return false;
      }
    };

    std::vector< CandRank_t > rank_v;

    for ( int icluster=0; icluster<(int)trunk_cand_v.size(); icluster++) {
      
      auto& trunk = trunk_cand_v[icluster];
      if ( trunk.points_v.size()<5 ) {
        CandRank_t rank( icluster, 1e9 );
        rank_v.push_back( rank );
        continue;
      }

      larflow::reco::cluster_pca( trunk );

      // determine direction
      // we want to use the pca axis, but we can switch to vertex->centroid if the trunk is bad
      std::vector<float> vtx2centroid(3,0);
      std::vector<float> pca1(3,0);
      float lenv2c = 0.;
      float lenpca = 0.;
      float cos_pca_v2c = 0.;
      for (int i=0; i<3; i++) {
        vtx2centroid[i] = trunk.pca_center[i]-pos[i];
        lenv2c += vtx2centroid[i]*vtx2centroid[i];
        pca1[i] = trunk.pca_axis_v[0][i];
        lenpca += pca1[i]*pca1[i];
        cos_pca_v2c += pca1[i]*vtx2centroid[i];
      }
      lenv2c = sqrt(lenv2c);
      lenpca = sqrt(lenpca);
      if ( lenv2c>0 && lenpca>0 ) {
        for (int i=0; i<3; i++)
          vtx2centroid[i] /= lenv2c;
        for (int i=0; i<3; i++)
          pca1[i] /= lenpca;
        cos_pca_v2c /= (lenpca*lenv2c);
      }

      // get start point of pca line
      int pca_start;
      int pca_end;
      float pcaend_dist[2] = {0,0};
      for (int i=0; i<3; i++) {
        pcaend_dist[0] += (trunk.pca_ends_v[0][i]-pos[i])*(trunk.pca_ends_v[0][i]-pos[i]);
        pcaend_dist[1] += (trunk.pca_ends_v[1][i]-pos[i])*(trunk.pca_ends_v[1][i]-pos[i]);
      }
      if ( pcaend_dist[0]<pcaend_dist[1] ) {
        pca_start = 0;
        pca_end = 1;
      }
      else {
        pca_start = 1;
        pca_end = 0;
      }
      std::vector<float> pcadir(3,0);
      float len_pcadir = 0.;
      for (int i=0; i<3; i++) {        
        pcadir[i] = trunk.pca_ends_v[pca_end][i]-trunk.pca_ends_v[pca_start][i];
        len_pcadir += pcadir[i]*pcadir[i];
      }
      len_pcadir = sqrt(len_pcadir);
      for (int i=0; i<3; i++)
        pcadir[i] /= len_pcadir;
             

      float score_pca = 0.;
      float score_v2c = 0.;
      std::vector<float> v2c_start(3,0);
      float max_s_v2c = 1e9;
      for (int ihit=0; ihit<(int)lfcluster.size(); ihit++) {
        std::vector<float> pt = { lfcluster[ihit][0], lfcluster[ihit][1], lfcluster[ihit][2] };

        // pca score
        float r_pca = larflow::reco::pointLineDistance3f( trunk.pca_ends_v[pca_start], trunk.pca_ends_v[pca_end], pt );
        float s_pca = larflow::reco::pointRayProjection3f( trunk.pca_ends_v[pca_start], pcadir, pt );
        if ( s_pca>3.0 )
          score_pca += r_pca/( (s_pca/14.0)*9.0 );
        else if (s_pca>0.0 && s_pca<3.0 )
          score_pca += r_pca/1.0;
        else
          score_pca += -s_pca/1.0;

        // v2c score
        float r_v2c = larflow::reco::pointLineDistance3f( pos, trunk.pca_center, pt );
        float s_v2c = larflow::reco::pointRayProjection3f( pos, vtx2centroid, pt )-min_dist;

        if ( s_v2c<max_s_v2c ) {
          max_s_v2c = s_v2c;
          v2c_start = pt;
        }
        
        if ( s_v2c>3.0 )
          score_v2c += r_v2c/( (s_v2c/14.0)*9.0 );
        else if (s_v2c>0.0 && s_v2c<3.0 )
          score_v2c += r_v2c/1.0;
        else
          score_v2c += -s_v2c/1.0;
        
      }

      if ( fabs(cos_pca_v2c)>0.7 ) {
        // use the pca score
        CandRank_t rank( icluster, score_pca );
        rank.start = trunk.pca_ends_v[pca_start];
        rank.dir   = pcadir;
        rank_v.push_back( rank );
      }
      else {
        CandRank_t rank( icluster, score_v2c );
        rank.start = v2c_start;
        rank.dir   = vtx2centroid;
        rank_v.push_back( rank );
      }
      
      
    }//loop over trunk candidates


    std::sort( rank_v.begin(), rank_v.end() );

    shower_start = rank_v.front().start;
    shower_dir   = rank_v.front().dir;
    shower_ll    = rank_v.front().llscore;

    return rank_v.size();

  }//end of NuVertexShowerReco::_make_trunk_cand

  /**
   * @brief here we record both truth-based and reco-based quantities to evaluate/tune 
   *        the shower to neutrino vertex attachment algorithm
   *
   */
  void NuVertexShowerReco::_gatherTruthShowerFeatures( larflow::reco::cluster_t& prong,
						    larflow::reco::NuVertexCandidate& vtx,
                NuVertexShowerReco::RecoShowerInfo_t& showerinfo )
  {
    // What we want to know?
    // Ultimately, should we attach the shower prong to the vertex, treating it like a trunk
    // right now the criteria is:
    // prong.impactdist<20.0 && prong.nhits>10
    // does this make sense? can we do something more sophisticated? (e.g. xgboost)
    
    // we call this before making pre-cuts on the shower
    // this is based on the number of hits essentially.
    // we ask: how well are reconstructing the size of the cluster?
    // does this reco prong correlate to a detectable trunk? a fragment? partially the trunk?

    // define a struct to hold the information we want to gather for each reco prong
    struct TrueShowerInfo_t {
      int geant_track_id;
      const std::vector< larcv::Image2D >* pix_mask_v; //< we are going to crop out the image per plane around the trunk
      std::vector< float > pix_sum_v; //< the pixel sum per plane
      TrueShowerInfo_t()
      : geant_track_id(-1),
      pix_mask_v(nullptr)
      {};
    };

    std::vector< TrueShowerInfo_t > trueprong_v;

    // first we loop througuh the true shower in the events
    bool exclude_neutrons = true;
    auto pnode_v = _mcpg->getNeutrinoParticles( exclude_neutrons );
    for (int inode=0; inode<(int)pnode_v.size(); inode++) {
      auto& pnode = pnode_v.at(inode);
      if ( pnode->pid==22 || std::abs(pnode->pid)==11 ) {
	      // photons
	      auto const& pointlist = _mcpg->getTruePhotonTrunk3DPoints( *pnode );
	      // check if this particle node has a photon trunk
	      if ( pointlist.size()==0 )
	        continue;

	      TrueShowerInfo_t prongtruth;
	      prongtruth.geant_track_id = pnode->tid;
	      prongtruth.pix_sum_v  = _mcpg->getTruePhotonTrunkPlanePixelSums( pnode->tid );
	      prongtruth.pix_mask_v = &(_mcpg->getTruePhotonTrunkPlaneImage2DMasks( pnode->tid ));
	      trueprong_v.emplace_back( std::move( prongtruth ) );
      } // if gamma node      
    }// end of loop over nodes of particles recorded by the simulation

    // we have to convert the 3d positions of the shower fragment to pixel locations
    typedef std::set< std::pair<int,int> > PixelSet_t;
    std::vector< PixelSet_t > plane_pixelsets_v(3);
    for (int ihit=0; ihit<(int)prong.points_v.size(); ihit++) {
      auto const& pt = prong.points_v.at(ihit);
      std::vector<float> imgpos =
	      ublarcvapp::mctools::MCPos2ImageUtils::Get()->to_imagepos( pt[0], pt[1], pt[2], 0.0 );
      // imagepos is (u,v,y,tick)
      if ( imgpos[3]<=0 ) {
	      // if the tick equals to 0000, then 3d point outside tpc
	      continue;
      }
      
      for (int p=0; p<3; p++) {
	      plane_pixelsets_v[p].insert( std::pair<int,int>( (int)imgpos[p], (int)imgpos[3] ) );
      }
    }//end of pixel gathering portion
    // Now we loop over the shower fragments and calculate variables for them
    float max_frac = 0;
    int max_frac_index = -1;
    int max_frac_trackid = -1;
    int max_frac_plane = -1;
    float max_frac_masksum = 0.;
    float max_frac_coverage = 0.;
    for (int iphoton=0; iphoton<(int)trueprong_v.size(); iphoton++) {
      auto const& trueprong = trueprong_v.at(iphoton);
      
      // get the plane with the most pixels
      int maxplane = -1;
      int maxplane_sum = 0;
      for (int p=0; p<(int)trueprong.pix_mask_v->size(); p++) {
	      const larcv::Image2D& maskcrop = trueprong.pix_mask_v->at(p);
	      float masksum = 0.;
	      for ( auto& mask : maskcrop.as_vector()  )
	        masksum += mask;
	      if (masksum>maxplane_sum) {
	        maxplane_sum = masksum;
	        maxplane = p;
	      }
      }

      // completely empty
      if ( maxplane==-1 || maxplane_sum<=0.0 )
	      continue;
      
      // ok now compare reco shower pixelset to ours in the larcv images
      // we loop over the reco prong's pixels
      int num_on_mask = 0;
      const larcv::Image2D& maxtruecrop = trueprong.pix_mask_v->at(maxplane);
      for ( auto& pix : plane_pixelsets_v[ maxplane ] ) {
	      float wire = (float)pix.first;
	      float tick = (float)pix.second;
	      if ( maxtruecrop.meta().contains( wire,tick ) ) {
          int pixrow = maxtruecrop.meta().row( tick );
          int pixcol = maxtruecrop.meta().col( wire );
	        if ( maxtruecrop.pixel( pixrow, pixcol, __FILE__, __LINE__ )>0.5 )
	          num_on_mask++;
	      }
      }
      // calculate the fraction of reco fragment's pixels belong to the true prong
      float frac = float(num_on_mask)/float(plane_pixelsets_v[maxplane].size());
      if ( frac>0.0 && frac > max_frac ) {
	      max_frac = frac;
	      max_frac_index = iphoton;
	      max_frac_trackid = trueprong.geant_track_id;
	      max_frac_coverage = frac;//float(num_on_mask)/float(maxplane_sum);
	      max_frac_plane = maxplane;
      }
    }//end of loop over true photons

    if ( max_frac_index>=0 ) {
      LARCV_DEBUG() << "Shower Reco fragment matched to true photon trunk (trackid=" << max_frac_trackid << "): "
		    << " frac pixel overlap=" << max_frac
		    << std::endl;
      LARCV_DEBUG() << "True prong pixelsums: ("
		    << trueprong_v.at( max_frac_index ).pix_sum_v[0] << ", "
		    << trueprong_v.at( max_frac_index ).pix_sum_v[1] << ", "
		    << trueprong_v.at( max_frac_index ).pix_sum_v[2] << ")"
		    << std::endl;
    }
    else {
      LARCV_DEBUG() << "Shower Reco fragment unmatched to true photon trunks" << std::endl;
      // fill sentinal values and return
      showerinfo._trueprong_trackid   = -1;  ///< unmatched sentinal valuesthe true prong that had the highest pixel overlap with the reco fragment's pixels
      showerinfo._frac_truetrunk      = 0.; ///< the fraction of the true prong's pixels covered by the reco fragment
      showerinfo._frac_recopurity     = -1.0;          ///< how much of the reco fragment pixels overlap with the matched true prong
      showerinfo._cluster_pixsum_MeV  = 0.;
      showerinfo._true_trunkdir       = std::vector<float>{ 0, 0, 0 };
      showerinfo._trueprong_dist2vtx  = -1.0;
      showerinfo._recoshower_dist2vtx = -1.0;
      showerinfo._recoshower_impactpar = -1.0;
      showerinfo._recoshower_cosine    = -2.0;
      showerinfo._recoshower_pixsum_MeV = -1.0;
      showerinfo._recoshower_trunkdir = std::vector<float>{ 0, 0, 0};
      // we do not match to any true photon trunks
      // so the true outcome for initial attachment should to not attach
      showerinfo._correct_outcome = 0;
      showerinfo._reco_outcome = -1; // not yet determined
      return;
    }

    // get the mcpg node, containing info about the true prong photon
    auto const& pnode_matched_trunk = _mcpg->findTrackID( max_frac_trackid );

    // get the true start point of the prong, in 3d
    std::vector<float> trueprong_first_edep_pos = pnode_matched_trunk->first_edep_pos;

    // we calculate the true trunk direction
    std::vector<float> trueprong_dir(3,0.0);
    auto const& ptlist = _mcpg->getTruePhotonTrunk3DPoints( max_frac_trackid );

    // we need to put the points into the cluster struct
    larflow::reco::cluster_t trueprong_cluster;
    trueprong_cluster.points_v.reserve( ptlist.size() );
    for (auto const& pt : ptlist ) {
      trueprong_cluster.points_v.push_back( pt );
    }

    try {
      // run pca code to calculate principle component of 3d points
      larflow::reco::cluster_pca( trueprong_cluster );

      // extract the ends of a line segment parallel to the 1st pc component that bounds the 3d points
      // nuvtx.pos is a reference point, meant to ensure that the pc axis line segment
      // has the closest point first.
      larlite::pcaxis clust_axis
	      = larflow::reco::cluster_make_pcaxis_wrt_point( trueprong_cluster, vtx.pos );
      float trueprong_mag = 0.;
      for (int v=0; v<3; v++) {
        trueprong_dir[v] = (clust_axis.getEigenVectors().at(4)[v]-clust_axis.getEigenVectors().at(3)[v]);
        trueprong_mag += trueprong_dir[v]*trueprong_dir[v];
      }
      trueprong_mag = sqrt(trueprong_mag);
      if ( trueprong_mag>0.0 ) {
      for (int v=0; v<3; v++)
        trueprong_dir[v] /= trueprong_mag;
      }
    }
    catch (...) {
      // problem running pca code. we define the trunk direction as the direction between the closest and first point from the trunk start
      float maxdist = 0.0;
      std::vector<float> maxpt(3,0.0);
      for ( auto const& pt : ptlist ) {
        float dist = 0.;
        for (int v=0; v<3; v++) {
          dist += (pt[v]-trueprong_first_edep_pos[v])*(pt[v]-trueprong_first_edep_pos[v]);
        }
        if ( dist > maxdist )  {
          maxpt = pt;
          maxdist = dist;
        }
      }
      maxdist = sqrt(maxdist);
      if ( maxdist>0.0 ) {
        for (int v=0; v<3; v++) {
          trueprong_dir[v] = (maxpt[v]-trueprong_first_edep_pos[v])/maxdist;
        }
      }
    }

    // calculate distance between true prong and reco vertex
    float dist2vtx = 0.;
    for (int v=0; v<3; v++) {
      float dd = trueprong_first_edep_pos[v]-vtx.pos[v];
      dist2vtx += dd*dd;
    }
    dist2vtx = sqrt(dist2vtx);

    // put info into the struct assigned to each reco fragment
    showerinfo._trueprong_trackid   = max_frac_trackid;  ///< the true prong that had the highest pixel overlap with the reco fragment's pixels
    showerinfo._frac_truetrunk      = max_frac_coverage; ///< the fraction of the true prong's pixels covered by the reco fragment
    showerinfo._frac_recopurity     = max_frac;          ///< how much of the reco fragment pixels overlap with the matched true prong
    showerinfo._cluster_pixsum_MeV  = trueprong_v.at( max_frac_index ).pix_sum_v[max_frac_plane]*0.0162;
    showerinfo._true_trunkdir       = trueprong_dir;
    showerinfo._trueprong_dist2vtx  = dist2vtx;
    showerinfo._recoshower_dist2vtx = -1.0;
    showerinfo._recoshower_impactpar = -1.0;
    showerinfo._recoshower_cosine    = -2.0;
    showerinfo._recoshower_pixsum_MeV = -1.0;
    showerinfo._recoshower_trunkdir = std::vector<float>{ 0, 0, 0};
    // what is the "ground truth correct" outcome?
    // set detectable threshold
    if ( showerinfo._cluster_pixsum_MeV>10.0 )
      showerinfo._correct_outcome = 1;
    else
      showerinfo._correct_outcome = 0;
    
  }//end of _gatherTruthShowerFeatures
  
  void NuVertexShowerReco::createMCAnalysisTree( TFile* outfile )
  {
    outfile->cd();
    _mcana_per_recoshower_tree = new TTree("nushowerbuilder_mcana_tree", "MC Analysis to evaluate and tune NuShowerBuilder Algorithm");

    _mcana_per_recoshower_tree->Branch( "closest_recovtx_dist",   &_mcana_closest_recovtx_dist,   "closest_recovtx_dist/F" );
    _mcana_per_recoshower_tree->Branch( "trueprong_pixsum_MeV",   &_mcana_trueprong_pixsum_MeV,   "trueprong_pixsum_MeV/F" );
    _mcana_per_recoshower_tree->Branch( "trueprong_efficiency",   &_mcana_trueprong_efficiency,   "trueprong_efficiency/F" );
    _mcana_per_recoshower_tree->Branch( "trueprong_dist2vtx",     &_mcana_trueprong_dist2vtx,     "trueprong_dist2vtx/F" );
    _mcana_per_recoshower_tree->Branch( "recofragment_purity",    &_mcana_recofragment_purity,    "recofragment_purity/F" );
    _mcana_per_recoshower_tree->Branch( "recofragment_dist2vtx",  &_mcana_recofragment_dist2vtx,  "recofragment_dist2vtx/F" );
    _mcana_per_recoshower_tree->Branch( "recofragment_impactpar", &_mcana_recofragment_impactpar, "recofragment_impactpar/F" );
    _mcana_per_recoshower_tree->Branch( "recofragment_cosine",    &_mcana_recofragment_cosine,    "recofragment_cosine/F" );
    _mcana_per_recoshower_tree->Branch( "recofragment_pixsum",    &_mcana_recofragment_pixsum,    "recofragment_pixsum/F" );
    _mcana_per_recoshower_tree->Branch( "reco_outcome",           &_mcana_reco_outcome,           "reco_outcome/I" );
    _mcana_per_recoshower_tree->Branch( "groundtruth_outcome",    &_mcana_groundtruth_outcome,    "groundtruth_outcome/I" );
    _mcana_per_recoshower_tree->Branch( "trueprong_trunkdir",     _mcana_trueprong_trunkdir,      "trueprong_trunkdir[3]/F" );
    _mcana_per_recoshower_tree->Branch( "recofragment_trunkdir",  _mcana_recofragment_trunkdir,   "recofragment_trunkdir[3]/F" );

    // clear the variables in the branch
    _set_default_mcana_variable_values();
  }

  void NuVertexShowerReco::_fill_mcanalysis_tree()
  {

    if ( !_mc_analysis_mode && !_mc_analysis_saveinfo_for_this_vertex )
      return;

    // transfer variables for each reco shower fragment that was evaluated
    for ( auto it : _map_prongindex_to_mcanainfo ) {
      int nuvtx_icluster = it.first;
      auto& mcanainfo = it.second;

      _mcana_trueprong_pixsum_MeV   = mcanainfo._cluster_pixsum_MeV;
      _mcana_trueprong_efficiency   = mcanainfo._frac_truetrunk;
      _mcana_trueprong_dist2vtx     = mcanainfo._trueprong_dist2vtx;
      _mcana_recofragment_purity    = mcanainfo._frac_recopurity;
      _mcana_recofragment_dist2vtx  = mcanainfo._recoshower_dist2vtx;
      _mcana_recofragment_impactpar = mcanainfo._recoshower_impactpar;
      _mcana_recofragment_cosine    = mcanainfo._recoshower_cosine;
      _mcana_recofragment_pixsum    = mcanainfo._recoshower_pixsum_MeV;
      _mcana_reco_outcome           = mcanainfo._reco_outcome;
      _mcana_groundtruth_outcome    = mcanainfo._correct_outcome;
      for (int v=0; v<3; v++) {
        _mcana_trueprong_trunkdir[v]    = mcanainfo._true_trunkdir[v];
        _mcana_recofragment_trunkdir[v] = mcanainfo._recoshower_trunkdir[v];
      }

      // save the values of the variables to the tree 
      _mcana_per_recoshower_tree->Fill();
    }
    
  }//end of NuVertexShowerReco::_gatherTruthShowerFeatures

  void NuVertexShowerReco::writeAnaTree()
  {
    if ( _mc_analysis_mode && _mcana_per_recoshower_tree ) {
      _mcana_per_recoshower_tree->Write();
    }
  }

  std::vector<float> NuVertexShowerReco::_get_cluster_pixsum( const std::vector<larcv::Image2D>& adc_v,
                                                              const larlite::larflowcluster& lfcluster ) 
  {
    std::vector<float> pixsum_v( adc_v.size(), 0.0 );
    std::vector< std::set< std::pair<int,int> > > pixvisited_v;
    pixvisited_v.resize( adc_v.size() );

    for (auto const& pt : lfcluster ) {
      std::vector<float> imgpos =
      ublarcvapp::mctools::MCPos2ImageUtils::Get()->to_imagepos( pt[0], pt[1], pt[2], 0.0 );

      for (int p=0; p<(int)adc_v.size(); p++) {
        auto const& img = adc_v.at(p);
        const larcv::ImageMeta& meta = img.meta();

        if ( meta.contains( imgpos[p], imgpos[3] ) ) {
          int pixrow = meta.row( imgpos[3] );
          int pixcol = meta.col( imgpos[p] );

          for (int dr=-1; dr<=1; dr++) {
            for (int dc=-1; dc<=1; dc++) {
              int r = pixrow + dr;
              int c = pixcol + dc;

              std::pair<int,int> pix(r,c);
              auto it_pix = pixvisited_v[p].find( pix );
              if ( it_pix==pixvisited_v[p].end() ) {
                float wire = meta.pos_x(c);
                float tick = meta.pos_y(r);
                if ( meta.contains(wire,tick) ) {
                  pixsum_v[p] += img.pixel(r,c, __FILE__, __LINE__ );
                }
                pixvisited_v[p].insert( pix );
              }

            }//end of col loop
          }//end of row loop
        }//if center pixel is inside the image
      }//end of loop over plane
    }//end of loop over cluster hits

    return pixsum_v;
  }

  /**
   * @brief get the score for all of the seeds using a bdt
   */
  void NuVertexShowerReco::getBDTseedscore( std::vector< NuVertexShowerReco::ProngRank_t >& seed_v )
  {
    // first we transfer the data into a DMatrix form
    int nrows = seed_v.size();
    std::vector<float> data_v( nrows*4, -1 );

    // variable list used during training (in larflow/Reco/ana/nuvertexshowerreco/train_xgboost.py)
    // varlist = ["recofragment_cosine",
    // 	       "recofragment_dist2vtx",
    // 	       "recofragment_impactpar",
    // 	       "recofragment_pixsum"]
    
    
    for (int i=0; i<nrows; i++) {
      auto& seed = seed_v.at(i);
      data_v[ 4*i ]     = seed.cosine;
      data_v[ 4*i + 1 ] = seed.dist2vtx;
      data_v[ 4*i + 2 ] = seed.impactpar;
      data_v[ 4*i + 3 ] = seed.pixsum;
    }
    
    DMatrixHandle dmatrix;
    nuvertexshowerreco_safe_xgboost( XGDMatrixCreateFromMat( data_v.data(), nrows, 4, -1, &dmatrix) );

    float const* out_result = NULL;
    uint64_t const* out_shape;
    uint64_t out_dim;    
    char const config[] =
      "{\"training\": false, \"type\": 0, "
      "\"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}";
    nuvertexshowerreco_safe_xgboost(XGBoosterPredictFromDMatrix(*_boosterhandle, dmatrix, config, &out_shape, &out_dim, &out_result));

    // copy into seed structs
    for (int i=0; i<nrows; i++) {
      auto& seed = seed_v.at(i);
      seed.score = -log( 1.0/out_result[i] - 1.0  );
    }

    std::sort( seed_v.begin(), seed_v.end() );

    nuvertexshowerreco_safe_xgboost(XGDMatrixFree(dmatrix));
    
  }

  void NuVertexShowerReco::_set_default_mcana_variable_values()
  {
    _mcana_closest_recovtx_dist = -1.0;
    _mcana_trueprong_pixsum_MeV = 0.;
    _mcana_trueprong_efficiency = 0.;
    _mcana_trueprong_dist2vtx   = -1.0;
    _mcana_recofragment_purity  = 0.0;
    _mcana_recofragment_dist2vtx = -1.0;
    _mcana_recofragment_impactpar = -1.0;
    _mcana_recofragment_cosine = -2.0;
    _mcana_recofragment_pixsum = 0.0;
    _mcana_reco_outcome = -1;
    _mcana_groundtruth_outcome = -1;
    for (int i=0; i<3; i++) {
      _mcana_trueprong_trunkdir[i] = 0.;
      _mcana_recofragment_trunkdir[i] = 0.;
    }
    
  }

}
}
