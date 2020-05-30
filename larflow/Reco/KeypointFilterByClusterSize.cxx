#include "KeypointFilterByClusterSize.h"

#include "cluster_functions.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/pcaxis.h"

namespace larflow {
namespace reco {

  void KeypointFilterByClusterSize::process( larcv::IOManager& iolcv, larlite::storage_manager& io_ll )
  {

    LARCV_INFO() << "Start" << std::endl;
    
    // get keypoints
    larlite::event_larflow3dhit* ev_keypoint =
      (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, _input_keypoint_tree_name );
    larlite::event_pcaxis* ev_pcaxis =
      (larlite::event_pcaxis*)io_ll.get_data( larlite::data::kPCAxis, _input_keypoint_tree_name );
    

    // get larflow hits
    larlite::event_larflow3dhit* evout_track_hit
      = (larlite::event_larflow3dhit*)io_ll.get_data(larlite::data::kLArFlow3DHit, _input_larflowhits_tree_name );


    // filter hits by score
    std::vector< const larlite::larflow3dhit* > goodhit_v;
    for ( auto const& hit : *evout_track_hit ) {
      if ( hit.track_score>0.7 ) {
        goodhit_v.push_back( &hit );
      }
    }

    // make output containers

    // accept container
    larlite::event_larflow3dhit* ev_keypoint_pass =
      (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, "keypoint_bigcluster" );
    larlite::event_pcaxis* ev_pcaxis_pass =
      (larlite::event_pcaxis*)io_ll.get_data( larlite::data::kPCAxis, "keypoint_bigcluster" );

    // reject container
    larlite::event_larflow3dhit* ev_keypoint_notpass =
      (larlite::event_larflow3dhit*)io_ll.get_data( larlite::data::kLArFlow3DHit, "keypoint_smallcluster" );
    larlite::event_pcaxis* ev_pcaxis_notpass =
      (larlite::event_pcaxis*)io_ll.get_data( larlite::data::kPCAxis, "keypoint_smallcluster" );
    
    
    for ( size_t ikp=0; ikp<ev_keypoint->size(); ikp++ ) {
      
      auto const& keypoint = ev_keypoint->at(ikp);
      auto const& keypointpca = ev_pcaxis->at(ikp);

      // for each keypoint, get hits within 20 cm box
      std::vector< std::vector<float> > nearby_v;
      int keypoint_idx = -1; // mark which larflow hit is the same as the keypoint in question
      float mindist = 1e9;
      int idx = 0;
      for ( auto const& phit : goodhit_v ) {

        auto const& hit = *phit;
        
        float dist=0;
        for (int v=0; v<3; v++) {
          dist += ( hit[v]-keypoint[v] )*( hit[v]-keypoint[v] );
        }
        dist = sqrt(dist);

        if ( dist<20.0 ) {
          nearby_v.push_back( std::vector<float>( {hit[0], hit[1], hit[2]} ) );
          if ( dist<mindist ) {
            mindist = dist;
            keypoint_idx = (int)nearby_v.size()-1;
          }
        }
        idx++;
      }

      if ( keypoint_idx == -1 ) {
        LARCV_DEBUG() << "keypoint[" << ikp << "] too far from any cluster. mindist=" << mindist << std::endl;
        // no cluster close enough, reject this keypoint
        ev_keypoint_notpass->push_back( keypoint );
        ev_pcaxis_notpass->push_back( keypointpca );
        LARCV_DEBUG() << "keypoint[" << ikp << "] saved as small-cluster index[" << (int)ev_keypoint_notpass->size()-1 << "]" << std::endl;        
        continue; // move to next point
      }

      LARCV_DEBUG() << "keypoint[" << ikp << "] neighboring points = " << nearby_v.size()
                    << ", kpindex=" << keypoint_idx << std::endl;
      
      // cluster hits
      std::vector< cluster_t > cluster_v;
      cluster_spacepoint_v( nearby_v, cluster_v, 10, 3.0, 30 );

      // find cluster keypoint is on and get its size
      int keypoint_cluster_size = -1;
      for ( size_t i=0; i<cluster_v.size(); i++ ) {
        auto& c = cluster_v[i];

        for (int ihit=0; ihit<c.hitidx_v.size(); ihit++ ) {
          if ( c.hitidx_v[ihit]==keypoint_idx ) {
            keypoint_cluster_size = (int)c.points_v.size();
          }

          if ( keypoint_cluster_size!=-1 ) break;
        }

        if ( keypoint_cluster_size!=-1 ) break;
      }

      if ( keypoint_cluster_size<50 ) {
        LARCV_DEBUG() << "keypoint[" << ikp << "] too small. size=" << keypoint_cluster_size << ", min dist to cluster=" << mindist << std::endl;
        ev_keypoint_notpass->push_back( keypoint );
        ev_pcaxis_notpass->push_back( keypointpca );
        LARCV_DEBUG() << "keypoint[" << ikp << "] saved as small-cluster index[" << (int)ev_keypoint_notpass->size()-1 << "]" << std::endl;
      }
      else {
        LARCV_DEBUG() << "keypoint[" << ikp << "] passes. size=" << keypoint_cluster_size << ", min dist to cluster=" << mindist << std::endl;                
        ev_keypoint_pass->push_back( keypoint );
        ev_pcaxis_pass->push_back( keypointpca );
        LARCV_DEBUG() << "keypoint[" << ikp << "] saved as big-cluster index[" << (int)ev_keypoint_pass->size()-1 << "]" << std::endl;        
      }
      
    }//end of keypoint loop
    
  }
  
}
}
