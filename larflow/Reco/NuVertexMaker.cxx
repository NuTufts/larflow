#include "NuVertexMaker.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {


  /**
   * constructor. calls default parameter method.
   *
   */
  NuVertexMaker::NuVertexMaker()
    : larcv::larcv_base("NuVertexMaker"),
    _ana_tree(nullptr)
  {
    _set_defaults();
  }

  /**
   * process event data
   *
   * goal of module is to form vertex candidates.
   * start by seeding possible vertices using
   * - keypoints 
   * - intersections of particle clusters (not yet implemented)
   * - vertex activity near ends of partice clusters (not yet implemented)
   *
   * event data inputs expected by algorthm:
   * - keypoint candidates, representd as larflow3dhit, used as vertex seeds. 
   *   Use add_keypoint_producer(...) to provide tree name before calling.
   * - particle cluster candidates, represented as larflowcluster, to associate to vertex seeds. 
   *   Use add_cluster_producer(...) to provide tree name before calling.
   * - particle cluster candidate containers need to be labeled with a certain ClusterType_t.
   * - the cluster type affects how it is added to the vertex and how the vertex candidates are scored
   *
   * output:
   * - vertex candidates stored in _vertex_v
   * - need to figure out way to store in larcv or larlite iomanagers
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
    
  }

  /**
   * create vertex candidates by associating clusters to vertex seeds
   *
   * inputs
   * - uses _keypoint_producers map to get vertex candidates
   * - uses _cluster_producers map to get cluster candidates
   *
   * outputs
   * - fills _vertex_v container
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
        vertex.score = 0.0;
        seed_v.emplace_back( std::move(vertex) );
      }
    }


    // associate to cluster objects
    for ( size_t vtxid=0; vtxid<seed_v.size(); vtxid++ ) {
      auto& vertex = seed_v[vtxid];
      
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
        if ( it->second==nullptr ) continue;

        for ( size_t icluster=0; icluster<it->second->size(); icluster++ ) {
          
          auto const& lfcluster = it->second->at(icluster);
          auto const& lfpca     = _cluster_pca_producers[it->first]->at(icluster);
          NuVertexCandidate::ClusterType_t ctype   = _cluster_type[it->first];
        
          std::vector<float> pcadir(3,0);
          std::vector<float> start(3,0);
          std::vector<float> end(3,0);
          float dist[2] = {0,0};
          for (int v=0; v<3; v++) {
            pcadir[v] = lfpca.getEigenVectors()[0][v];
            start[v]  = lfpca.getEigenVectors()[3][v];
            end[v]    = lfpca.getEigenVectors()[4][v];
            dist[0] += ( start[v]-vertex.pos[v] )*( start[v]-vertex.pos[v] );
            dist[1] += ( end[v]-vertex.pos[v] )*( end[v]-vertex.pos[v] );
          }
          dist[0] = sqrt(dist[0]);
          dist[1] = sqrt(dist[1]);
          int closestend = (dist[0]<dist[1]) ? 0 : 1;
          float gapdist = dist[closestend];
          float r = pointLineDistance( start, end, vertex.pos );

          // wide association for now
          if ( gapdist>_cluster_type_max_gap[ctype] )
            continue;
          
          if ( r>_cluster_type_max_impact_radius[ctype] )
            continue;

          // else attach
          NuVertexCandidate::VtxCluster_t cluster;
          cluster.producer = it->first;
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
   * clear all data containers
   *
   */
  void NuVertexMaker::clear()
  {
    _vertex_v.clear();
    _keypoint_producers.clear();
    _keypoint_pca_producers.clear();
    _cluster_producers.clear();
    _cluster_pca_producers.clear();
  }

  /** 
   * set parameter defaults
   *
   * _cluster_type_max_impact_radius: per cluster type, maximum radius to accept candidate into vertex
   * _cluster_type_max_gap: per cluster type, maximum gap to accept candidate into vertex
   *
   */
  void NuVertexMaker::_set_defaults()
  {
    // Track
    _cluster_type_max_impact_radius[ NuVertexCandidate::kTrack ] = 5.0;
    _cluster_type_max_gap[ NuVertexCandidate::kTrack ] = 5.0;

    // ShowerKP
    _cluster_type_max_impact_radius[ NuVertexCandidate::kShowerKP ] = 20.0;
    _cluster_type_max_gap[ NuVertexCandidate::kShowerKP ]           = 50.0;

    // Shower
    _cluster_type_max_impact_radius[ NuVertexCandidate::kShower ] = 50.0;
    _cluster_type_max_gap[ NuVertexCandidate::kShower ]           = 100.0;
    
  }

  /**
   * provide score to vertex seeds in order to provide a way to rank them
   * not used to cut or anything.
   *
   * attempting to rank by number of quality cluster associations
   * cluster association quality based on how well cluster points back to vertex
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
      }
      std::cout << "cluster[type=" << cluster.type << "] impact=" << cluster.impact << " gap=" << cluster.gap << " score=" << clust_score << std::endl;
      vtx.score += clust_score;
    }        
  }

  /**
   * add branch to tree that will save container of vertex candidates
   *
   */
  void NuVertexMaker::add_nuvertex_branch( TTree* tree )
  {
    _ana_tree = tree;
    _own_tree = false;
    tree->Branch("nuvertex_v", &_vertex_v );
  }


  /**
   * create a TTree into which we will save the vertex container
   *
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

  
  
}
}
