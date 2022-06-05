#include "NuVertexMaker.h"

#include "geofuncs.h"
#include "larlite/DataFormat/track.h"
#include "larlite/LArUtil/Geometry.h"
#include "larlite/LArUtil/DetectorProperties.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include "larflow/Reco/NuVertexFitter.h"

namespace larflow {
namespace reco {


  /**
   * @brief default constructor
   *
   * calls default parameter method.
   *
   */
  NuVertexMaker::NuVertexMaker()
    : larcv::larcv_base("NuVertexMaker"),
      _output_stage( kMerged ),
      _num_input_clusters(0),
      _ana_tree(nullptr)
  {
    _set_defaults();
  }

  /**
   * @brief process event data
   *
   * goal of module is to form vertex candidates.
   * start by seeding possible vertices using
   * @verbatim embed:rst:leading-asterisk
   *  * keypoints 
   *  * intersections of particle clusters (not yet implemented)
   *  * vertex activity near ends of partice clusters (not yet implemented)
   * @endverbatim
   *
   * event data inputs expected by algorthm:
   * @verbatim embed:rst:leading-asterisk
   * * keypoint candidates, representd as larflow3dhit, used as vertex seeds. 
   *   Use add_keypoint_producer(...) to provide tree name before calling.
   * * particle cluster candidates, represented as larflowcluster, to associate to vertex seeds. 
   *   Use add_cluster_producer(...) to provide tree name before calling.
   * * particle cluster candidate containers need to be labeled with a certain ClusterType_t.
   * * the cluster type affects how it is added to the vertex and how the vertex candidates are scored
   * @endverbatim
   *
   * output:
   * @verbatim embed:rst:leading-asterisk
   *  * vertex candidates stored in _vertex_v
   *  * need to figure out way to store in larcv or larlite iomanagers
   * @endverbatim
   *
   * @param[in] iolcv Instance of LArCV IOManager with event data
   * @param[in] ioll  Instance of larlite storage_manager containing event data
   *
   */
  void NuVertexMaker::process( larcv::IOManager& iolcv,
                               larlite::storage_manager& ioll )
  {

    // load keypoints
    LARCV_INFO() << "Number of keypoint producers: " << _keypoint_producers.size() << std::endl;
    for ( auto it=_keypoint_producers.begin(); it!=_keypoint_producers.end(); it++ ) {
      LARCV_INFO() << "Load keypoint data with tree name[" << it->first << "]" << std::endl;
      it->second = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, it->first );
      auto it_pca = _keypoint_pca_producers.find( it->first );
      if ( it_pca==_keypoint_pca_producers.end() ) {
        _keypoint_pca_producers[it->first] = nullptr;
        it_pca = _keypoint_pca_producers.find( it->first );
      }
      it_pca->second = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, it->first );
      LARCV_INFO() << "keypoints from [" << it->first << "]: " << it->second->size() << " keypoints" << std::endl;
    }

    // load clusters
    int cluster_index = 0;    
    LARCV_INFO() << "Number of cluster producers: " << _cluster_producers.size() << std::endl;
    for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
      LARCV_INFO() << "Load cluster data with tree name[" << it->first << "]" << std::endl;
      it->second = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, it->first );
      auto it_pca = _cluster_pca_producers.find( it->first );
      if ( it_pca==_cluster_pca_producers.end() ) {
        _cluster_pca_producers[it->first] = nullptr;
        it_pca = _cluster_pca_producers.find( it->first );
      }
      it_pca->second = (larlite::event_pcaxis*)ioll.get_data( larlite::data::kPCAxis, it->first );
      LARCV_INFO() << "clusters from [" << it->first << "]: " << it_pca->second->size() << " pcaxes" << std::endl;

      if ( _cluster_type[it->first]==NuVertexCandidate::kTrack ) {
	auto it_track = _cluster_track_producers.find( it->first );
	it_track->second = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, it->first );
	LARCV_INFO() << "clusters from [" << it->first << "]: " << it_track->second->size() << " tracks" << std::endl;
      }

      // we provide a cluster index label. this is to help downstream algorithms
      // an easy way to identify the same cluster
      for (auto& c : *it->second ) {
	c.matchedflash_idx = cluster_index;
	cluster_index++;
      }
    }
    _num_input_clusters = cluster_index;

    // larcv
    larcv::EventImage2D* ev_img2d =
      (larcv::EventImage2D*) iolcv.get_data( larcv::kProductImage2D, "wire" );
    auto const& adc_v = ev_img2d->as_vector();

    // cryo, tpc loop: we create neutrino vertex candidates, TPC-by-TPC
    auto const geom = larlite::larutil::Geometry::GetME();
    for (int icryo=0; icryo<(int)geom->Ncryostats(); icryo++) {
      for (int itpc=0; itpc<(int)geom->NTPCs(icryo); itpc++) {
	
	int startplaneindex = geom->GetSimplePlaneIndexFromCTP( icryo, itpc, 0 );

	// look for it in the adc_v list
	bool found = false;
	for ( auto const& img : adc_v ) {
	  if ( img.meta().id()==startplaneindex ) {
	    found = true;
	    break;
	  }
	}
	
	if ( !found )
	  continue; // to next tpc
    
	_createCandidates( iolcv, itpc, icryo );
	_merge_candidates( itpc, icryo );
	if ( _apply_cosmic_veto ) {
	  _cosmic_veto_candidates( ioll );
	}

	LARCV_INFO() << "Num NuVertexCandidates: created=" << _vertex_v.size()
		     << "  after-merging=" << _merged_v.size()
		     << "  after-veto=" << _vetoed_v.size()
		     << std::endl;
	
	for ( auto& vertex : _vetoed_v ) {
	  
	  if ( vertex.cluster_v.size()>0 ) {
	    _vertex_v.emplace_back( std::move(vertex) );
	    if ( logger().debug() ) {
	      LARCV_DEBUG() << "Vertex[" << vertex.keypoint_producer << ", " << vertex.keypoint_index << "] " << std::endl;
	      LARCV_DEBUG() << "  number of clusters: " << vertex.cluster_v.size() << std::endl;
	      LARCV_DEBUG() << "  producer: " << vertex.keypoint_producer << std::endl;
	      LARCV_DEBUG() << "  pos: (" << vertex.pos[0] << "," << vertex.pos[1] << "," << vertex.pos[2] << ")" << std::endl;
	      LARCV_DEBUG() << "  score: " << vertex.score << std::endl;
	      for (size_t ic=0; ic<vertex.cluster_v.size(); ic++) {
		LARCV_DEBUG() << "  cluster[" << ic << "] "
			      << " prod=" << vertex.cluster_v[ic].producer
			      << " idx=" << vertex.cluster_v[ic].index 
			      << " impact=" << vertex.cluster_v[ic].impact << " cm"
			      << " gap=" << vertex.cluster_v[ic].gap << " cm"
                          << " npts=" << vertex.cluster_v[ic].npts
			      << std::endl;
	      }
	    }//end of if debug
	  }//end of if has clusters
	}//end of vertex loop
        
	_refine_position( iolcv, ioll );

      }//end of TPC loop
    }//end of CRYO LOOP

    // make cluster book
    _buildClusterBook();
    
  }

  /**
   * @brief create vertex candidates by associating clusters to vertex seeds
   *
   * @return Mutable vector of NuVertex Candidates, which set depends on the _output_stage flag.
   *  
   */
  std::vector<NuVertexCandidate>& NuVertexMaker::get_mutable_output_candidates()
  {
    switch (_output_stage) {
    case kRaw:
      return get_mutable_nu_candidates();
      break;
    case kMerged:
      return get_mutable_merged_candidates();
      break;
    case kVetoed:
      return get_mutable_vetoed_candidates();
      break;
    case kFitted:
      return get_mutable_fitted_candidates();
      break;
    default:
      throw std::runtime_error("NuVertexMaker::get_mutable_output_candidates: unrecognized _output_stage flag setting");
      break;
    };

    return get_mutable_nu_candidates();
  }
  
  /**
   * @brief create vertex candidates by associating clusters to vertex seeds
   *
   * inputs
   * @verbatim embed:rst:leading-asterisk
   *  * uses _keypoint_producers map to get vertex candidates
   *  * uses _cluster_producers map to get cluster candidates
   * @endverbatim
   *
   * outputs
   * @verbatim embed:rst:leading-asterisk
   *  * fills _vertex_v container
   * @endverbatim
   * 
   */
  void NuVertexMaker::_createCandidates(larcv::IOManager& iolcv,
					const int tpcid, const int cryoid )
  {

    LARCV_DEBUG() << "Associate clusters to vertices via impact par and gap distance. "
		  << "(Cryo,TPC)=(" << cryoid << "," << tpcid << ")"
		  << std::endl;
    
    // loop over vertices, calculate impact parameters to all clusters, keep if close enough.
    // limit pairings by gap distance (different for shower and track)

    // need to get a meta
    larcv::EventImage2D* adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    auto const& meta = adc->as_vector().front().meta();

    // make initial vertex objects
    std::vector<NuVertexCandidate> seed_v;
    for ( auto it=_keypoint_producers.begin(); it!=_keypoint_producers.end(); it++ ) {
      if ( it->second==nullptr ) continue;

      for ( size_t vtxid=0; vtxid<it->second->size(); vtxid++ ) {

        auto const& lf_vertex = it->second->at(vtxid);

	if ( lf_vertex.targetwire[4]!=tpcid || lf_vertex.targetwire[5]!=cryoid ) {
	  // not the TPC, CRYO
	  continue;
	}
        
        NuVertexCandidate vertex;
        vertex.keypoint_producer = it->first;
        vertex.keypoint_index = vtxid;
        vertex.keypoint_type = (lf_vertex.size()>3) ? lf_vertex[3] : -1;
        vertex.pos.resize(3,0);
        for (int i=0; i<3; i++)
          vertex.pos[i] = lf_vertex[i];
        vertex.tick  = lf_vertex.tick;
	if ( vertex.tick>meta.min_y() && vertex.tick<meta.max_y() )
	  vertex.row = meta.row( vertex.tick, __FILE__, __LINE__ );
	vertex.tpcid  = tpcid;
	vertex.cryoid = cryoid;
        vertex.col_v = lf_vertex.targetwire;
        vertex.score = 0.0;
        vertex.maxScore = 0.0;
        vertex.avgScore = 0.0;
        vertex.netScore = (lf_vertex.size() > 4) ? lf_vertex[4] : -1.;
        vertex.netNuScore = (vertex.keypoint_type == (int)larflow::kNuVertex) ? vertex.netScore : -1.;
        seed_v.emplace_back( std::move(vertex) );
      }
    }

    // we calculate the track ends

    // associate to cluster objects
    for ( size_t vtxid=0; vtxid<seed_v.size(); vtxid++ ) {
      auto& vertex = seed_v[vtxid];

      LARCV_DEBUG() << "=== ATTACHING TO (" << vertex.pos[0] << "," << vertex.pos[1] << "," << vertex.pos[2] << ") ===" << std::endl;
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
        if ( it->second==nullptr ) continue;

        for ( size_t icluster=0; icluster<it->second->size(); icluster++ ) {
          
          auto const& lfcluster = it->second->at(icluster);
          auto const& lfpca     = _cluster_pca_producers[it->first]->at(icluster);

	  int cluster_tpcid  = lfcluster.at(0).targetwire[4];
	  int cluster_cryoid = lfcluster.at(0).targetwire[5];
	  if ( cluster_tpcid!=tpcid || cluster_cryoid!=cryoid )
	    continue;
	  
          NuVertexCandidate::ClusterType_t ctype   = _cluster_type[it->first];
	  LARCV_DEBUG() << "  testing cluster[" << icluster << "] (cryo,tpc)=(" << cluster_cryoid << "," <<  cluster_tpcid << ")" << std::endl;
          bool attached = _attachClusterToCandidate( vertex, lfcluster, lfpca,
                                                     ctype, it->first, icluster, true );

          LARCV_DEBUG() << "  cluster " << it->first << "[" << icluster << "] attached=" << attached << std::endl;

          
        }//end of cluster loop
      }//end of cluster container loop

      _score_vertex( vertex );
      
    }//end of vertex loop


    std::sort( seed_v.begin(), seed_v.end() );
    
    for ( auto& vertex : seed_v ) {
      
      if ( vertex.cluster_v.size()>0 ) {
        _vertex_v.emplace_back( std::move(vertex) );
        if ( logger().debug() ) {
          LARCV_DEBUG() << "Vertex[" << vertex.keypoint_producer << ", " << vertex.keypoint_index << "] " << std::endl;
          LARCV_DEBUG() << "  number of clusters: " << vertex.cluster_v.size() << std::endl;
          LARCV_DEBUG() << "  producer: " << vertex.keypoint_producer << std::endl;
          LARCV_DEBUG() << "  pos: (" << vertex.pos[0] << "," << vertex.pos[1] << "," << vertex.pos[2] << ")" << std::endl;
          LARCV_DEBUG() << "  score: " << vertex.score << std::endl;
          for (size_t ic=0; ic<vertex.cluster_v.size(); ic++) {
            LARCV_DEBUG() << "  cluster[" << ic << "] "
                          << " prod=" << vertex.cluster_v[ic].producer
                          << " idx=" << vertex.cluster_v[ic].index 
                          << " impact=" << vertex.cluster_v[ic].impact << " cm"
                          << " gap=" << vertex.cluster_v[ic].gap << " cm"
                          << " npts=" << vertex.cluster_v[ic].npts
                          << std::endl;
          }
        }//end of if debug
      }//end of if has clusters
    }//end of vertex loop


  }

  /**
   * @brief clear all data containers
   */
  void NuVertexMaker::clear()
  {
    _vertex_v.clear();
    _merged_v.clear();
    _keypoint_producers.clear();
    _keypoint_pca_producers.clear();
    _cluster_producers.clear();
    _cluster_pca_producers.clear();
  }

  /** 
   * @brief set parameter defaults
   *
   * @verbatim embed:rst:leading-asterisk
   * * _cluster_type_max_impact_radius: per cluster type, maximum radius to accept candidate into vertex
   * * _cluster_type_max_gap: per cluster type, maximum gap to accept candidate into vertex
   * @endverbatim
   *
   */
  void NuVertexMaker::_set_defaults()
  {
    // Track
    _cluster_type_max_impact_radius[ NuVertexCandidate::kTrack ] = 5.0;
    _cluster_type_max_gap[ NuVertexCandidate::kTrack ] = 10.0;

    // ShowerKP
    _cluster_type_max_impact_radius[ NuVertexCandidate::kShowerKP ] = 10.0;
    _cluster_type_max_gap[ NuVertexCandidate::kShowerKP ]           = 50.0;

    // Shower
    _cluster_type_max_impact_radius[ NuVertexCandidate::kShower ] = 10.0;
    _cluster_type_max_gap[ NuVertexCandidate::kShower ]           = 50.0;

    _apply_cosmic_veto = false;
    _num_input_clusters = 0;
  }

  /**
   * @brief provide score to vertex seeds 

   * Scores used to rank vertex candidates.
   * Scores not used to cut at this stage.
   *
   * attempting to rank by number of quality cluster associations
   * cluster association quality based on how well cluster points back to vertex
   *
   * @param[in] vtx A candidate neutrino vertex
   *
   */
  void NuVertexMaker::_score_vertex( NuVertexCandidate& vtx )
  {
    vtx.score = 0.;
    vtx.maxScore = 0.;
    const float tau_gap_shower    = 20.0;
    const float tau_impact_shower = 10.0;
    const float tau_ratio_shower = 1.0;

    const float tau_gap_track    = 3.0;
    const float tau_impact_track = 3.0;
    
    for ( auto& cluster : vtx.cluster_v ) {
      float clust_score = 1.0;
      if ( cluster.type==NuVertexCandidate::kTrack ) {
        if ( cluster.gap>3.0 )
          clust_score *= (1.0/tau_gap_track)*exp( -cluster.gap/tau_gap_track );
        if ( cluster.impact>3.0 )
          clust_score *= (1.0/tau_impact_track)*exp( -cluster.impact/tau_impact_track );
      }
      else {
        float ratio = cluster.impact/cluster.gap;
        clust_score *= (1.0/tau_ratio_shower)*exp( -ratio/tau_ratio_shower );
        if ( cluster.impact>3.0 )
          clust_score *= (1.0/tau_impact_shower)*exp( -cluster.impact/tau_impact_shower );
      }
      //std::cout << "cluster[type=" << cluster.type << "] impact=" << cluster.impact << " gap=" << cluster.gap << " score=" << clust_score << std::endl;
      vtx.score += clust_score;
      if(clust_score > vtx.maxScore) vtx.maxScore = clust_score;
    }        
    vtx.avgScore = vtx.score;
    if(vtx.cluster_v.size() > 0) vtx.avgScore /= vtx.cluster_v.size();
  }

  /**
   * @brief add branch to tree that will save container of vertex candidates
   *
   * @param[in] tree ROOT tree to add branches to
   */
  void NuVertexMaker::add_nuvertex_branch( TTree* tree )
  {
    _ana_tree = tree;
    _own_tree = false;
    tree->Branch("nuvertex_v", &_vertex_v );
    tree->Branch("numerged_v", &_merged_v );
    tree->Branch("nuvetoed_v", &_vetoed_v );
    tree->Branch("nufitted_v", &_fitted_v);    
  }


  /**
   * @brief create a TTree into which we will save the vertex container
   */
  void NuVertexMaker::make_ana_tree()
  {
    if ( !_ana_tree ) {
      _ana_tree = new TTree("NuVertexMakerTree","output of NuVertexMaker Class");

      // since we own this tree, we will add the run,subrun,event to it
      _ana_tree->Branch("run",&_ana_run,"run/I");
      _ana_tree->Branch("subrun",&_ana_run,"subrun/I");
      _ana_tree->Branch("event",&_ana_run,"event/I");      
      // add the vertex container
      add_nuvertex_branch( _ana_tree );
      // mark that we own it, so we destroy it later
      _own_tree = true;
    }
  }

  /**
   * @brief merge nearby vertices
   *
   * merge if close. the "winning vertex" has the best score when we take the union of prongs.
   * the winning vertex also gets the union of prongs assigned to it.
   *
   * we fill the _merged_v vector using _vertex_v candidates.
   *
   */
  void NuVertexMaker::_merge_candidates( const int tpcid, const int cryoid )
  {

    LARCV_DEBUG() << "START MERGER (CRYO,TPC)=(" << cryoid << "," << tpcid << ") ====================" << std::endl;
    
    // sort by score
    // start with best score, absorb others, (so N^2 algorithm)

    // clear existing merged
    _merged_v.clear();

    if ( _vertex_v.size()==0 )
      return;
    
    _merged_v.reserve( _vertex_v.size() );
    
    // struct for us to sort by score
    struct NuScore_t {
      const NuVertexCandidate* nu;
      float score;
      NuScore_t( const NuVertexCandidate* the_nu, float s )
        : nu(the_nu),
          score(s)
      {};
      bool operator<( const NuScore_t& rhs ) const
      {
        if ( score>rhs.score ) return true;
        return false;
      }      
    };

    std::vector< NuScore_t > nu_v;
    nu_v.reserve( _vertex_v.size() );
    for ( auto const& nucand : _vertex_v ) {
      nu_v.push_back( NuScore_t(&nucand, nucand.score) );
    }
    std::sort( nu_v.begin(), nu_v.end() );

    std::vector<int> used_v( nu_v.size(), 0 );

    // seed first merger candidate
    _merged_v.push_back( *(nu_v[0].nu) );
    used_v[0] = 1;

    if ( _vertex_v.size()==1 )
      return;
    
    int nused = 0;
    int current_cand_index = 0; // index of candidate in _merged_v, for whom we are currently looking for mergers
    while ( nused<(int)nu_v.size() && current_cand_index<_merged_v.size() ) {

      auto& cand = _merged_v.at(current_cand_index);

      for (int i=0; i<(int)nu_v.size(); i++) {
        if ( used_v[i]==1 ) continue;

        // test vertex
        auto const& test_vtx = *nu_v[i].nu;

	// vertices need to be in the same TPC
	if ( cand.tpcid!=test_vtx.tpcid || cand.cryoid!=test_vtx.cryoid )
	  continue;
        
        // test vertex distance
        float dist = 0.;
        for (int v=0; v<3; v++)
          dist += (test_vtx.pos[v]-cand.pos[v])*(test_vtx.pos[v]-cand.pos[v]);
        dist = sqrt(dist);

        if ( dist==0.0 || dist>5.0 )
          continue; // too far to merge (or the same)

        // if within distance, absorb clusters to make union of two vertices
        // set the pos, keypoint producer based on best score
        used_v[i] = 1;

        // set keypoint network score and type for merged vertex
        if(test_vtx.netScore > cand.netScore) cand.netScore = test_vtx.netScore;
        if(test_vtx.netNuScore > cand.netNuScore) cand.netNuScore = test_vtx.netNuScore;
        
        if ( cand.keypoint_type!=test_vtx.keypoint_type ) {
          if ( test_vtx.keypoint_type<cand.keypoint_type )
            cand.keypoint_type = test_vtx.keypoint_type;
        }

        // loop over clusters inside test vertex we are merging with
        int nclust_before = cand.cluster_v.size();
        for ( auto const& test_clust : test_vtx.cluster_v ) {

          // check the clusters against the current candidate's clusters
          bool is_found = false;
          for (auto const& curr_clust : cand.cluster_v ) {
            if ( curr_clust.index==test_clust.index && curr_clust.producer==test_clust.producer ) {
              is_found = true;
              break;
            }
          }
          
          if ( !is_found ) {
            // if not found, add it to the current candidate vertex
            auto it_clust = _cluster_producers.find( test_clust.producer );
            auto it_pca   = _cluster_pca_producers.find( test_clust.producer );
            auto it_ctype = _cluster_type.find( test_clust.producer );
            if ( it_clust==_cluster_producers.end() || it_pca==_cluster_pca_producers.end() ) {
              throw std::runtime_error("ERROR NuVertexMaker. could not find cluster/pca producer in dict");
            }
            auto const& lfcluster = it_clust->second->at(test_clust.index);
            auto const& lfpca     = it_pca->second->at(test_clust.index);
            auto const& clust_t   = it_ctype->second;
	    LARCV_DEBUG() << "Call _attachClusterToCandidate for merger" << std::endl;
            _attachClusterToCandidate( cand, lfcluster, lfpca, clust_t,
                                       test_clust.producer, test_clust.index, false );
          }
              
        }//after test cluster loop

        // score the current candidate
        _score_vertex( cand );

        // here we create a duplicate,
        // but with the vertex moved to the pos of the test_vtx
        // then we decide which one to keep based on highest score
        
      }//end of loop to absorb vertices

      // get the next unused absorbed vertex, to seed as merger
      for (int i=0; i<(int)nu_v.size(); i++) {
        if ( used_v[i]==1 ) continue;
        _merged_v.push_back( *nu_v[i].nu );
        used_v[i] = 1;
        break;
      }
      
      current_cand_index++;
    }
    
    
  }

  /**
   * @brief add cluster to a neutrino vertex candidate
   *
   * We attach clusters that point towards the vertex candidate.
   * 
   * For clusters with 1st pc axis less than some length, we use the 1st pc axis to determine if
   * it points to the vertex.
   * If for longer clusters, we'll use the spacepoints near the end of the cluster 
   * and determine it's pcaxis.
   *
   * @param[in] vertex NuVertexCandidate instance to add to
   * @param[in] lfcluster cluster in the form of a larflowcluster instance
   * @param[in] lfpca cluster pca info for lfcluster
   * @param[in] ctype type of cluster
   * @param[in] producer tree name that held the cluster being passed in
   * @param[in] icluster cluster index
   * @param[in] apply_cut if true, clusters only attached if satisfies limits on impact parameter and gap distance
   */
  bool NuVertexMaker::_attachClusterToCandidate( NuVertexCandidate& vertex,
                                                 const larlite::larflowcluster& lfcluster,
                                                 const larlite::pcaxis& lfpca,
                                                 NuVertexCandidate::ClusterType_t ctype,
                                                 std::string producer,
                                                 int icluster,
                                                 bool apply_cut )
  {

    // define the ray to to use to check the intersection.
    float pcalen = 0.;    
    std::vector<float> pca_dir(3,0);
    std::vector<float> pca_start(3,0);
    std::vector<float> pca_end(3,0);
    float dist[2] = {0,0};
    float len = 0.;
    std::vector<float> startpt(3,0);
    std::vector<float> endpt(3,0);
    std::vector<float> dir(3,0);

    for (int v=0; v<3; v++) {
      pca_dir[v]   = lfpca.getEigenVectors()[0][v];
      pca_start[v] = lfpca.getEigenVectors()[3][v];
      pca_end[v]   = lfpca.getEigenVectors()[4][v];
      dist[0] += ( pca_start[v]-vertex.pos[v] )*( pca_start[v]-vertex.pos[v] );
      dist[1] += ( pca_end[v]-vertex.pos[v] )*( pca_end[v]-vertex.pos[v] );
      len += (pca_start[v]-pca_end[v])*(pca_start[v]-pca_end[v]);
      pcalen += pca_dir[v]*pca_dir[v];
    }
    pcalen = sqrt(pcalen);
    len = sqrt(len);
    dist[0] = sqrt(dist[0]);
    dist[1] = sqrt(dist[1]);
    int closestend = (dist[0]<dist[1]) ? 0 : 1;
    float gapdist = dist[closestend];
    if ( closestend==0 ) {
      startpt = pca_start;
      endpt   = pca_end;
      dir     = pca_dir;
    }
    else {
      startpt = pca_end;
      endpt   = pca_start;
      for (int v=0; v<3; v++)
	dir[v] = -pca_dir[v];
    }
    
    LARCV_DEBUG() << "  s(" << startpt[0] << "," << startpt[1] << "," << startpt[2] << ") "
                  << "  e(" << endpt[0] << "," << endpt[1] << "," << endpt[2] << ") "
                  << "  len=" << len << std::endl;    
    if (pcalen<0.1 || std::isnan(pcalen) ) {
      return false;
    }

    if ( ctype==NuVertexCandidate::kTrack && len>20.0 ) {

      // replace with end of track
      auto const& track = _cluster_track_producers[producer]->at(icluster);
      int npts = track.NumberTrajectoryPoints();
      // which end is closer
      dist[0] = 0.;
      dist[1] = 0.;
      for (int v=0; v<3; v++) {
	float dx = track.LocationAtPoint(0)[v]-vertex.pos[v];
	dist[0] += dx*dx;
	dx = track.LocationAtPoint(npts-1)[v]-vertex.pos[v];
	dist[1] += dx*dx;
      }
      TVector3 enddir;
      float s = 0.;      
      if ( dist[0]<dist[1] ) {
	for (int ipt=0; ipt<npts-1; ipt++) {
	  auto& current = track.LocationAtPoint(ipt);
	  auto& nextpt  = track.LocationAtPoint(ipt+1);
	  float ds = (current-nextpt).Mag();
	  s+=ds;
	  if ( s>10.0 ) {
	    enddir = nextpt-track.LocationAtPoint(0);
	    for (int v=0; v<3; v++)  {
	      dir[v] = enddir[v]/enddir.Mag();
	      startpt[v] = track.LocationAtPoint(0)[v];
	      endpt[v]   = nextpt[v];
	    }
	  }
	}
	
      }
      else {
	for (int ipt=npts-1; ipt>=1; ipt--) {
	  auto& current = track.LocationAtPoint(ipt);
	  auto& nextpt  = track.LocationAtPoint(ipt-1);
	  float ds = (current-nextpt).Mag();
	  s+=ds;
	  if ( s>10.0 ) {
	    enddir = nextpt-track.LocationAtPoint(npts-1);
	    for (int v=0; v<3; v++)  {
	      dir[v] = enddir[v]/enddir.Mag();
	      startpt[v] = track.LocationAtPoint(npts-1)[v];
	      endpt[v]   = nextpt[v];
	    }
	  }
	}
      }

      LARCV_DEBUG() << "  long track. "
		    << "  s(" << startpt[0] << "," << startpt[1] << "," << startpt[2] << ") "
		    << "  e(" << endpt[0] << "," << endpt[1] << "," << endpt[2] << ") "
		    << "  len=" << len << std::endl;    	

    }//if cluster is track and longer than 20 cm
    
    float r = pointLineDistance( startpt, endpt, vertex.pos );

    float projs = pointRayProjection<float>( startpt, dir, vertex.pos );
    float ends  = pointRayProjection<float>( startpt, dir, endpt );

    if ( apply_cut ) {
      
      LARCV_DEBUG() << "  connection metrics: "
		    << " gapdist=" << gapdist << " [" << (gapdist<_cluster_type_max_gap[ctype]) << "]"
		    << " r=" << r << " [" << (r<_cluster_type_max_impact_radius[ctype]) << "]"
		    << " projs=" << projs
		    << " ends=" << ends
		    << std::endl;
      
      // wide association for now
      if ( gapdist>_cluster_type_max_gap[ctype] )
        return false;
          
      if ( r>_cluster_type_max_impact_radius[ctype] )
        return false;

      if ( ctype==NuVertexCandidate::kShowerKP || ctype==NuVertexCandidate::kShower ) {
        if ( projs>2.0 && projs < (ends-2.0) )
          return false;
      }
    }

    // else attach
    NuVertexCandidate::VtxCluster_t cluster;
    cluster.producer = producer;
    cluster.index = icluster;
    cluster.dir.resize(3,0);
    cluster.pos.resize(3,0);

    if ( closestend==0 ) {
      for ( int i=0; i<3; i++) {
	cluster.dir[i] = pca_dir[i];
	cluster.pos[i] = pca_start[i];
      }
    }
    else {
      for ( int i=0; i<3; i++) {
	cluster.dir[i] = -pca_dir[i];
	cluster.pos[i] = pca_end[i];
      }      
    }
    
    cluster.gap = gapdist;
    cluster.impact = r;
    cluster.type = ctype;
    cluster.npts = (int)lfcluster.size();
    vertex.cluster_v.emplace_back( std::move(cluster) );
    vertex.cluster_pca_v.push_back( lfpca );
    
    return true;
  }
  
  /**
   * @brief apply cosmic track veto to candidate vertices
   *
   * Get tracks from tree, `boudarycosmicnoshift`.
   * Vertices close to these are removed from consideration.
   *
   * @param[in] ioll larlite storage_manager containing event data.
   *
   */
  void NuVertexMaker::_cosmic_veto_candidates( larlite::storage_manager& ioll )
  {

    // get cosmic tracks
    larlite::event_track* ev_cosmic
      = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, "boundarycosmicnoshift" );

    // loop over vertices
    _vetoed_v.clear();
    for ( auto& vtx : _merged_v ) {

      bool close2cosmic = false;
      
      for (int itrack=0; itrack<(int)ev_cosmic->size(); itrack++) {
        const larlite::track& cosmictrack = ev_cosmic->at(itrack);

        float mindist_segment = 1.0e9;
        float mindist_point   = 1.0e9;
        for (int ipt=0; ipt<(int)cosmictrack.NumberTrajectoryPoints()-1; ipt++) {

          const TVector3& pos  = cosmictrack.LocationAtPoint(ipt);
          const TVector3& next = cosmictrack.LocationAtPoint(ipt+1);

          std::vector<float> fpos  = { (float)pos(0),  (float)pos(1),  (float)pos(2) };
          std::vector<float> fnext = { (float)next(0), (float)next(1), (float)next(2) };

          std::vector<float> fdir(3,0);
          float flen = 0.;
          for (int i=0; i<3; i++) {
            fdir[i] = fnext[i]-fpos[i];
            flen += fdir[i]*fdir[i];
          }          
          if ( flen>0 ) {
            flen = sqrt(flen);
            for (int i=0; i<3; i++)
              fdir[i] /= flen;
          }
          else {
            continue;
          }
            

          float r = larflow::reco::pointLineDistance3f( fpos, fnext, vtx.pos );
          float s = larflow::reco::pointRayProjection3f( fpos, fdir, vtx.pos );

          if ( s>=0 && s<=flen && mindist_segment>r) {
            mindist_segment = r;
          }

          float pt1dist = 0.;
          for (int i=0; i<3; i++) {
            pt1dist += (fpos[i]-vtx.pos[i])*(fpos[i]-vtx.pos[i]);
          }
          pt1dist = sqrt(pt1dist);
          if ( pt1dist<mindist_point ) {
            mindist_point = pt1dist;
          }

          if ( ipt+1==(int)cosmictrack.NumberTrajectoryPoints() ) {
            pt1dist = 0;
            for (int i=0; i<3; i++) {
              pt1dist += (fnext[i]-vtx.pos[i])*(fnext[i]-vtx.pos[i]);
            }
            pt1dist = sqrt(pt1dist);
            if ( pt1dist<mindist_point ) {
              mindist_point = pt1dist;
            }            
          }
        }//end of loop over cosmic points


        if ( mindist_segment<10.0 || mindist_point<10.0 ) {
          close2cosmic = true;
        }


        if ( close2cosmic )
          break;
      }//end of loop over tracks


      if ( !close2cosmic ) {
        _vetoed_v.push_back( vtx );
      }
      
    }//end of vertex loop

    LARCV_INFO() << "Vertices after cosmic veto: " << _vetoed_v.size() << " (from " << _merged_v.size() << ")" << std::endl;
    
  }

  /**
   * @brief Use NuVertexFitter to refine vertex position
   * 
   * Also, provide refined prong direction and dQ/dx measure
   * 
   * @param[in] iolcv LArCV IO manager with current event data
   * @param[in] ioll  larlite storage_manager with current event data
   *
   */
  void NuVertexMaker::_refine_position( larcv::IOManager& iolcv,
                                        larlite::storage_manager& ioll )                                        
  {

    auto const geom = larlite::larutil::Geometry::GetME();
    auto const detp = larutil::DetectorProperties::GetME();
    
    // need to get a meta
    larcv::EventImage2D* adc
      = (larcv::EventImage2D*)iolcv.get_data(larcv::kProductImage2D,"wire");
    auto const& meta = adc->as_vector().front().meta();

    larflow::reco::NuVertexFitter fitter;
    if ( _apply_cosmic_veto )
      fitter.process( iolcv, ioll, get_vetoed_candidates() );
    else
      fitter.process( iolcv, ioll, get_merged_candidates() );

    const std::vector< std::vector<float> >& fitted_pos_v
      = fitter.get_fitted_pos();

    _fitted_v.clear();
    if ( _apply_cosmic_veto ) {

      int ivtx=0;
      for ( auto const& vtx : _vetoed_v ) {
        NuVertexCandidate fitcand = vtx;
        fitcand.pos = fitted_pos_v[ivtx];

        Double_t dpos[3] = {  fitcand.pos[0], fitcand.pos[1], fitcand.pos[2] };

	int tpcid  = fitcand.tpcid;
	int cryoid = fitcand.cryoid;
	int nplanes = geom->Nplanes( tpcid, cryoid );
        
        // update row, tick, col
        fitcand.col_v.resize(nplanes);
        for  (int p=0; p<nplanes; p++) 
          fitcand.col_v[p] = geom->WireCoordinate( dpos, p, tpcid, cryoid );
        fitcand.tick = detp->ConvertXToTicks( fitcand.pos[0], 0, tpcid, cryoid );

	if ( fitcand.tick>meta.min_y() && fitcand.tick<meta.max_y() )  {
	  fitcand.row = meta.row( fitcand.tick, __FILE__, __LINE__ );
	  _fitted_v.emplace_back( std::move(fitcand) );
	}
	ivtx++;
      }
      
    }
    
  }

  /**
   * @brief create a class to keep track of clusters attached to this vertex
   *
   */
  void NuVertexMaker::_buildClusterBook()
  {
    _cluster_book_v.clear();
    _cluster_book_v.reserve( get_mutable_output_candidates().size() );

    for (int ivtx=0; ivtx<(int)get_mutable_output_candidates().size(); ivtx++) {
      ClusterBookKeeper book;
      book.cluster_status_v.clear();
      book.cluster_status_v.resize(_num_input_clusters,0);
      auto& nuvtx = get_mutable_output_candidates().at(ivtx);
      for (size_t ic=0; ic<nuvtx.cluster_v.size(); ic++) {
	std::string producer = nuvtx.cluster_v[ic].producer;
	int idx = nuvtx.cluster_v[ic].index;
	int cindex = _cluster_producers[producer]->at(idx).matchedflash_idx;
	book.cluster_status_v.at(cindex) = 1; // has been assigned
      }
      _cluster_book_v.emplace_back( std::move(book) );
    }
  }
  
}
}
