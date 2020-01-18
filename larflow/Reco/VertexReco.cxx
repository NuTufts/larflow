#include "VertexReco.h"

#include "TVector3.h"
#include <fstream>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/larflowcluster.h"

namespace larflow {
namespace reco {

  std::vector<VertexReco::Candidate_t> VertexReco::findVertices( larcv::IOManager& iolcv,
                                                                 larlite::storage_manager& ioll ) {
    
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    
    larlite::event_larflowcluster* ev_lftrack
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "pcacluster" );
    larlite::event_larflowcluster* ev_lfshower
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfshower" );
    
    std::vector<VertexReco::Candidate_t> trackshower_vtx_v = trackShowerIntersections( *ev_lftrack,
                                                                                       *ev_lfshower,
                                                                                       ev_adc->Image2DArray(),
                                                                                       10.0,
                                                                                       10.0 );

    return trackshower_vtx_v;
  }
  
  std::vector<VertexReco::Candidate_t> VertexReco::trackShowerIntersections( const larlite::event_larflowcluster& lftrack_v,
                                                                             const larlite::event_larflowcluster& lfshower_v,
                                                                             const std::vector<larcv::Image2D>& adc_v,
                                                                             const float max_end_dist,
                                                                             const float max_inter_dist ) {
    std::vector<Candidate_t> candidate_v;

    // convert track and shower back to cluster_t
    std::vector<cluster_t> shower_v;
    std::vector<cluster_t> track_v;

    for ( auto const& lfc : lftrack_v ) {
      cluster_t cluster = cluster_from_larflowcluster( lfc );
      track_v.emplace_back( std::move(cluster) );
    }

    for ( auto const& lfc : lfshower_v ) {
      cluster_t cluster = cluster_from_larflowcluster( lfc );
      shower_v.emplace_back( std::move(cluster) );
    }

    // ok, look at each shower, test against each track cluster
    // shitty N^2. Maybe we enforce hit size
    std::vector<int> track_used_v( track_v.size(), 0 );
    for ( auto const& shower : shower_v ) {

      // enforce min size
      if ( shower.points_v.size()<10 ) continue;

      // create a candidate
      Candidate_t vtx;
      vtx.type = kUnconnectedTrackShower;
      vtx.cluster_v.push_back( shower );
      vtx.prong_v.push_back( kShowerProng );
          
      for ( auto const& track : track_v ) {
        if ( track.points_v.size()<10 || track.pca_len<3.0 ) continue;

        // test for closest ends
        std::vector< std::vector<float> > endpts;
        float endptdist = cluster_closest_endpt_dist( shower, track, endpts );

        // get shortest line between skew points
        // http://mathworld.wolfram.com/Line-LineDistance.html
        // Using TVector3 because I am lazy!
        TVector3 a( shower.pca_ends_v[1][0]-shower.pca_ends_v[0][0],
                    shower.pca_ends_v[1][1]-shower.pca_ends_v[0][1],
                    shower.pca_ends_v[1][2]-shower.pca_ends_v[0][2] ); // x2-x1 (ends of shower pca line segment)
        TVector3 b( track.pca_ends_v[1][0]-track.pca_ends_v[0][0],
                    track.pca_ends_v[1][1]-track.pca_ends_v[0][1],
                    track.pca_ends_v[1][2]-track.pca_ends_v[0][2] );   // x4-x3 (ends of track pca line segment)
        TVector3 c( track.pca_ends_v[0][0]-shower.pca_ends_v[0][0],
                    track.pca_ends_v[0][1]-shower.pca_ends_v[0][1],
                    track.pca_ends_v[0][2]-shower.pca_ends_v[0][2] ); // x2-x1 (ends of shower pca)

        TVector3 axb = a.Cross(b);
        float linelinedist = fabs(c.Dot(axb))/axb.Mag();

        if ( endptdist < 5.0 && linelinedist < 5.0 ) {

          // add prong to candidate
          if ( endptdist<1.0 )
            vtx.type = kConnectedTrackShower;

          if ( vtx.pos.size()==0 ) {
            // set for first time
            vtx.pos.resize(3);
            for (int i=0; i<3; i++ ) vtx.pos[i] = 0.5*( endpts[0][i]+endpts[1][i] ); // mid point of closest line for now.
          }
          vtx.cluster_v.push_back( track );

          vtx.prong_v.push_back( kTrackProng );

        }
        
      }//end of track loop
      
      // save if we have more than the shower in the cluster
      if ( vtx.cluster_v.size()>1 ) {
        candidate_v.emplace_back( std::move(vtx) );
      }
    }
    
    dumpCandidates2json( candidate_v, "out_prototype_vertex.json" );

    return candidate_v;
  }

  void VertexReco::dumpCandidates2json( const std::vector< VertexReco::Candidate_t >& vtx_v, std::string outfile ) {

    nlohmann::json j;
    std::vector<nlohmann::json> jvtx_v;
    
    for ( auto const& vtx : vtx_v ) {

      nlohmann::json jvtx;

      std::vector< nlohmann::json > jcluster_v;
      std::vector<int>              ctype_v(vtx.cluster_v.size());
      int ii=0;
      for ( auto const& cluster: vtx.cluster_v ) {
        jcluster_v.push_back( cluster_json(cluster) );
        if ( vtx.prong_v[ii]==kTrackProng ) ctype_v[ii] = 0;
        else if ( vtx.prong_v[ii]==kShowerProng ) ctype_v[ii] = 1;
        else ctype_v[ii] = 2;
        ii++;
      }
      jvtx["clusters"] = jcluster_v;
      jvtx["pos"]      = vtx.pos;
      jvtx["cluster_types"] = ctype_v;

      jvtx_v.emplace_back( std::move(jvtx) );
    }

    j["vertices"] = jvtx_v;

    std::ofstream o(outfile.c_str());
    j>>o;
    o.close();
    
  }
  
}
}
