#include "TrackClusterBuilder.h"

namespace larflow {
namespace reco {

  void TrackClusterBuilder::process( larcv::IOManager& iolcv,
                                     larlite::storage_manager& ioll )
  {

    std::string producer = "trackprojsplit_full";
    
    larlite::event_larflowcluster* ev_cluster
      = (larlite::event_larflowcluster*)ioll.get_data(larlite::data::kLArFlowCluster, producer);
    larlite::event_pcaxis* ev_pcaxis
      = (larlite::event_pcaxis*)ioll.get_data(larlite::data::kPCAxis,producer);

    loadClusterLibrary( *ev_cluster, *ev_pcaxis );
    
  }

  void TrackClusterBuilder::loadClusterLibrary( const larlite::event_larflowcluster& cluster_v,
                                                const larlite::event_pcaxis& pcaxis_v )
  {

    for (int i=0; i<cluster_v.size(); i++) {
      const larlite::larflowcluster& cluster = cluster_v.at(i);
      const larlite::pcaxis& pca = pcaxis_v.at(i);

      // create a segment object
      std::vector<float> start(3,0);
      std::vector<float> end(3,0);
      for (int v=0; v<3; v++){
        start[v] = pca.getEigenVectors()[3][v];
        end[v]   = pca.getEigenVectors()[4][v];
      }

      Segment_t seg( start, end );
      _segment_v.push_back(seg);
      _segment_v.back().idx = (int)_segment_v.size()-1;
    }

    LARCV_INFO() << "Stored " << _segment_v.size() << " track segments" << std::endl;
  }

  /**
   * we go ahead and build all possible connections
   * this is going to be an N^2 algorithm, so use wisely.
   * we use this to develop the code. also, can use it for smallish N.
   *
   */
  void TrackClusterBuilder::buildConnections()
  {

    for (int iseg=0; iseg<(int)_segment_v.size(); iseg++ ) {
      Segment_t& seg_i = _segment_v[iseg];
      
      for (int jseg=iseg+1; jseg<(int)_segment_v.size(); jseg++) {

        // define a connection in both directions
        Segment_t& seg_j = _segment_v[jseg];


        // first we find the closest ends between them
        std::vector<float>* ends[2][2] = { {&seg_i.start,&seg_i.end},
                                           {&seg_j.start,&seg_j.end} };

        std::vector<float> dist(4,0);
        for (int i=0; i<2; i++) {
          for(int j=0; j<2; j++) {
            for (int v=0; v<3; v++) {
              dist[ 2*i + j ] += (ends[0][i]->at(v)-ends[1][j]->at(v))*( ends[0][j]->at(v)-ends[1][j]->at(v) );
            }
            dist[ 2*i + j ] = sqrt( dist[2*i+j] );
          }
        }

        // find the smallest dist
        int idx_small = 0;
        float min_dist = 1.0e9;
        for (int i=0; i<4; i++) {
          if ( min_dist>dist[i] ) {
            idx_small = i;
            min_dist = dist[i];
          }
        }

        std::vector<float>* end_i = ends[0][ idx_small/2 ];
        std::vector<float>* end_j = ends[1][ idx_small%2 ];

        // i->j
        Connection_t con_ij;
        con_ij.node = &seg_j;
        con_ij.from_seg_idx = iseg;
        con_ij.to_seg_idx = jseg;
        con_ij.dist = min_dist;
        con_ij.endidx = idx_small%2;
        con_ij.cosine = 0.;
        for (int v=0; v<3; v++) {
          con_ij.cosine += (end_j->at(v)-end_i->at(v))*seg_i.dir[v]/min_dist;
        }
        
        Connection_t con_ji;
        con_ji.node = &seg_i;
        con_ji.from_seg_idx = jseg;
        con_ji.to_seg_idx = iseg;
        con_ji.dist = min_dist;
        con_ij.endidx = idx_small/2;
        con_ji.cosine = 0.;
        for (int v=0; v<3; v++) {
          con_ji.cosine += (end_i->at(v)-end_j->at(v))*seg_j.dir[v]/min_dist;
        }

        _connect_m[ std::pair<int,int>(iseg,jseg) ] = con_ij;
        _connect_m[ std::pair<int,int>(jseg,iseg) ] = con_ji;
        
      }      
    }

    LARCV_INFO() << "Made " << _connect_m.size() << " connections" << std::endl;
  }

}
}
