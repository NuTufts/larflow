#include "NuVertexMaker.h"

#include "geofuncs.h"
#include "DataFormat/track.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"

#include "NuVertexFitter.h"

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
      LARCV_INFO() << "clusters from [" << it->first << "]: " << it->second->size() << " clusters" << std::endl;
    }

    _createCandidates();
    _merge_candidates();
    if ( _apply_cosmic_veto ) {
      _cosmic_veto_candidates( ioll );
    }
    LARCV_INFO() << "Num NuVertexCandidates: created=" << _vertex_v.size()
                 << "  after-merging=" << _merged_v.size()
                 << "  after-veto=" << _vetoed_v.size()
                 << std::endl;

    _refine_position( iolcv, ioll );
    
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
  void NuVertexMaker::_createCandidates()
  {

    LARCV_DEBUG() << "Associate clusters to vertices via impact par and gap distance" << std::endl;
    
    // loop over vertices, calculate impact parameters to all clusters, keep if close enough.
    // limit pairings by gap distance (different for shower and track)

    // make vertex objects
    std::vector<NuVertexCandidate> seed_v;
    for ( auto it=_keypoint_producers.begin(); it!=_keypoint_producers.end(); it++ ) {
      if ( it->second==nullptr ) continue;

      for ( size_t vtxid=0; vtxid<it->second->size(); vtxid++ ) {

        auto const& lf_vertex = it->second->at(vtxid);
        
        NuVertexCandidate vertex;
        vertex.keypoint_producer = it->first;
        vertex.keypoint_index = vtxid;
        vertex.pos.resize(3,0);
        for (int i=0; i<3; i++)
          vertex.pos[i] = lf_vertex[i];
        vertex.tick  = lf_vertex.tick;
        vertex.col_v = lf_vertex.targetwire;
        vertex.score = 0.0;
        seed_v.emplace_back( std::move(vertex) );
      }
    }


    // associate to cluster objects
    for ( size_t vtxid=0; vtxid<seed_v.size(); vtxid++ ) {
      auto& vertex = seed_v[vtxid];

      LARCV_DEBUG() << "=== ATTACH TO (" << vertex.pos[0] << "," << vertex.pos[1] << "," << vertex.pos[2] << ") ===" << std::endl;
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
        if ( it->second==nullptr ) continue;

        for ( size_t icluster=0; icluster<it->second->size(); icluster++ ) {
          
          auto const& lfcluster = it->second->at(icluster);
          auto const& lfpca     = _cluster_pca_producers[it->first]->at(icluster);
          NuVertexCandidate::ClusterType_t ctype   = _cluster_type[it->first];

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
    _cluster_type_max_gap[ NuVertexCandidate::kTrack ] = 5.0;

    // ShowerKP
    _cluster_type_max_impact_radius[ NuVertexCandidate::kShowerKP ] = 10.0;
    _cluster_type_max_gap[ NuVertexCandidate::kShowerKP ]           = 50.0;

    // Shower
    _cluster_type_max_impact_radius[ NuVertexCandidate::kShower ] = 10.0;
    _cluster_type_max_gap[ NuVertexCandidate::kShower ]           = 50.0;

    _apply_cosmic_veto = false;
    
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
    }        
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
  void NuVertexMaker::_merge_candidates()
  {

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
        
        // test vertex distance
        float dist = 0.;
        for (int v=0; v<3; v++)
          dist += (test_vtx.pos[v]-cand.pos[v])*(test_vtx.pos[v]-cand.pos[v]);
        dist = sqrt(dist);

        if ( dist>5.0 )
          continue; // too far to merge (or the same)

        // if within distance, absorb clusters to make union
        // set the pos, keypoint producer based on best score
        used_v[i] = 1;

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

    std::vector<float> pcadir(3,0);
    std::vector<float> start(3,0);
    std::vector<float> end(3,0);
    float dist[2] = {0,0};
    float pcalen = 0.;
    float len = 0.;
    for (int v=0; v<3; v++) {
      pcadir[v] = lfpca.getEigenVectors()[0][v];
      start[v]  = lfpca.getEigenVectors()[3][v];
      end[v]    = lfpca.getEigenVectors()[4][v];
      dist[0] += ( start[v]-vertex.pos[v] )*( start[v]-vertex.pos[v] );
      dist[1] += ( end[v]-vertex.pos[v] )*( end[v]-vertex.pos[v] );
      len += (start[v]-end[v])*(start[v]-end[v]);
      pcalen += pcadir[v]*pcadir[v];
    }
    pcalen = sqrt(pcalen);
    len = sqrt(len);
    LARCV_DEBUG() << "  s(" << start[0] << "," << start[1] << "," << start[2] << ") "
                  << "  e(" << end[0] << "," << end[1] << "," << end[2] << ") "
                  << "  len=" << len << std::endl;    
    if (pcalen<0.1 || std::isnan(pcalen) ) {
      return false;
    }          
    dist[0] = sqrt(dist[0]);
    dist[1] = sqrt(dist[1]);
    int closestend = (dist[0]<dist[1]) ? 0 : 1;
    float gapdist = dist[closestend];
    float r = pointLineDistance( start, end, vertex.pos );

    float projs = pointRayProjection<float>( start, pcadir, vertex.pos );
    float ends  = pointRayProjection<float>( start, pcadir, end );

    if ( apply_cut ) {
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
    for ( int i=0; i<3; i++) {
      if ( closestend==0 ) {
        cluster.dir[i] = pcadir[i];
        cluster.pos[i] = start[i];
      }
      else {
        cluster.dir[i] = -pcadir[i];
        cluster.pos[i] = end[i];
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
        
        // update row, tick, col
        fitcand.col_v.resize(3);
        for  (int p=0; p<3; p++) 
          fitcand.col_v[p] = larutil::Geometry::GetME()->WireCoordinate( dpos, p );
        fitcand.tick = fitcand.pos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5+3200;
        fitcand.row = meta.row( fitcand.tick, __FILE__, __LINE__ );
        
        ivtx++;
        _fitted_v.emplace_back( std::move(fitcand) );
      }
      
    }
    
  }
  
  
}
}
