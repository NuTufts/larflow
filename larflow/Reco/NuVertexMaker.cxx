#include "NuVertexMaker.h"

#include "geofuncs.h"

namespace larflow {
namespace reco {


  NuVertexMaker::NuVertexMaker()
    : larcv::larcv_base("NuVertexMaker")
  {
    _set_defaults();
  }
  
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

  void NuVertexMaker::_createCandidates()
  {

    LARCV_DEBUG() << "Associate clusters to vertices via impact par and gap distance" << std::endl;
    
    // loop over vertices, calculate impact parameters to all clusters, keep if close enough.
    // limit pairings by gap distance (different for shower and track)

    // make vertex objects
    for ( auto it=_keypoint_producers.begin(); it!=_keypoint_producers.end(); it++ ) {
      if ( it->second==nullptr ) continue;

      for ( size_t vtxid=0; vtxid<it->second->size(); vtxid++ ) {

        auto const& lf_vertex = it->second->at(vtxid);
        
        Vertex_t vertex;
        vertex.keypoint_producer = it->first;
        vertex.keypoint_index = vtxid;
        vertex.pos.resize(3,0);
        for (int i=0; i<3; i++)
          vertex.pos[i] = lf_vertex[i];
        vertex.score = 0.0;
        _vertex_v.emplace_back( std::move(vertex) );
      }
    }


    // associate to cluster objects
    for ( size_t vtxid=0; vtxid<_vertex_v.size(); vtxid++ ) {
      auto& vertex = _vertex_v[vtxid];
      
      for ( auto it=_cluster_producers.begin(); it!=_cluster_producers.end(); it++ ) {
        if ( it->second==nullptr ) continue;

        for ( size_t icluster=0; icluster<it->second->size(); icluster++ ) {
          
          auto const& lfcluster = it->second->at(icluster);
          auto const& lfpca     = _cluster_pca_producers[it->first]->at(icluster);
          ClusterType_t ctype   = _cluster_type[it->first];
        
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
          VtxCluster_t cluster;
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


    std::sort( _vertex_v.begin(), _vertex_v.end() );
    
    if ( logger().debug() ) {
      for ( auto const& vertex : _vertex_v ) {
        if ( vertex.cluster_v.size()>=0 ) {
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
        }        
      }//end of vertex loop
    }// if debug

  }

  void NuVertexMaker::clear()
  {
    _vertex_v.clear();
    _keypoint_producers.clear();
    _keypoint_pca_producers.clear();
    _cluster_producers.clear();
    _cluster_pca_producers.clear();
  }

  void NuVertexMaker::_set_defaults()
  {
    // Track
    _cluster_type_max_impact_radius[ kTrack ] = 5.0;
    _cluster_type_max_gap[ kTrack ] = 5.0;

    // ShowerKP
    _cluster_type_max_impact_radius[ kShowerKP ] = 20.0;
    _cluster_type_max_gap[ kShowerKP ]           = 50.0;

    // Shower
    _cluster_type_max_impact_radius[ kShower ] = 20.0;
    _cluster_type_max_gap[ kShower ]           = 50.0;
    
  }

  void NuVertexMaker::_score_vertex( Vertex_t& vtx )
  {
    vtx.score = 0.;
    const float tau_gap_shower    = 20.0;
    const float tau_impact_shower = 10.0;
    const float tau_ratio_shower = 1.0;

    const float tau_gap_track    = 3.0;
    const float tau_impact_track = 3.0;
    
    for ( auto& cluster : vtx.cluster_v ) {
      float clust_score = 1.0;
      if ( cluster.type==kTrack ) {
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
  
}
}
