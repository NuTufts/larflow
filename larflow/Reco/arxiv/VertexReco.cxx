#include "VertexReco.h"

#include "TVector3.h"
#include <fstream>
#include <set>
#include <ctime>

#include "larcv/core/DataFormat/EventImage2D.h"
#include "DataFormat/larflowcluster.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/Geometry.h"

#include "ublarcvapp/UBWireTool/UBWireTool.h"

namespace larflow {
namespace reco {

  std::vector<VertexReco::Candidate_t> VertexReco::findVertices( larcv::IOManager& iolcv,
                                                                 larlite::storage_manager& ioll )
  {
    
    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    
    larlite::event_larflowcluster* ev_lftrack
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "pcacluster" );
    larlite::event_larflowcluster* ev_lfshower
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfshower" );

    // convert track and shower back to cluster_t
    std::vector<cluster_t> shower_v;
    std::vector<cluster_t> track_v;
    
    for ( auto const& lfc : *ev_lftrack ) {
      cluster_t cluster = cluster_from_larflowcluster( lfc );
      track_v.emplace_back( std::move(cluster) );
    }
    
    for ( auto const& lfc : *ev_lfshower ) {
      cluster_t cluster = cluster_from_larflowcluster( lfc );
      shower_v.emplace_back( std::move(cluster) );
    }

    
    std::vector<VertexReco::Candidate_t> trackshower_vtx_v
      = trackShowerIntersections( track_v, shower_v, ev_adc->Image2DArray(), 10.0, 10.0 );

    std::vector<VertexReco::Candidate_t> showerActivity_vtx_v
      = showerEndActivity( track_v, shower_v, ev_adc->Image2DArray() );

    // transfer info one vector
    for ( auto& cand : showerActivity_vtx_v )
      trackshower_vtx_v.emplace_back( std::move(cand) );

    dumpCandidates2json( trackshower_vtx_v, "out_prototype_vertex.json" );

    // return
    return trackshower_vtx_v;
  }
  
  std::vector<VertexReco::Candidate_t> VertexReco::trackShowerIntersections( const larlite::event_larflowcluster& lftrack_v,
                                                                             const larlite::event_larflowcluster& lfshower_v,
                                                                             const std::vector<larcv::Image2D>& adc_v,
                                                                             const float max_end_dist,
                                                                             const float max_inter_dist ) {
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

    return trackShowerIntersections( track_v, shower_v, adc_v, max_end_dist, max_inter_dist );

  }

  
  std::vector<VertexReco::Candidate_t> VertexReco::trackShowerIntersections( const std::vector<cluster_t>& track_v,
                                                                             const std::vector<cluster_t>& shower_v,
                                                                             const std::vector<larcv::Image2D>& adc_v,
                                                                             const float max_end_dist,
                                                                             const float max_inter_dist ) {
    std::vector<Candidate_t> candidate_v;


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

    return candidate_v;
  }

  std::vector< VertexReco::Candidate_t > VertexReco::showerEndActivity( const std::vector<cluster_t>& track_v,
                                                                        const std::vector<cluster_t>& shower_v,
                                                                        const std::vector<larcv::Image2D>& adc_v ) {
    std::vector<Candidate_t> candidate_v;

    // project both pca-line ends into all planes
    // in each plane, find maximum pixel
    // sum pixel values in box in each plane
    // are any of these really the end point?
    // can make (3*2)/2=3 3d points. is it certain distance to pca-ends?
    // finally, need to argue that points around vertex end, is empty, excluding attached cluster
    //   - each point, find cluster 3d vertex point is in.
    //   - excluding this cluster + shower cluster, no more than X other-cluster points can be near activity point
    //     though dont count points with Y cm from 3D point (so its a 3D shell of points must be empty)
    // finish this, then can try to see what efficiency is.

    // first, merge showers
    std::vector<cluster_t> merged_shower_v = _mergeShowerClusters( shower_v, 50.0 );
    std::cout << "[VertexReco] num of shower clusters after merger: " << merged_shower_v.size() << std::endl;

    std::vector<cluster_t> merged_shower_track_v = _mergeShowerAndTrackClusters( merged_shower_v, track_v );

    // loop over each shower, find candidate vertex activity points
    int showeridx = -1;
    std::set< std::vector<int> > past_coord;    
    for ( auto const& shower : merged_shower_track_v ) {
    //for ( auto const& shower : shower_v ) {
      
      showeridx++;      

      // get meta for first image -- assume its the same for all
      const larcv::ImageMeta& meta = adc_v.front().meta();
      
      // find max pts in region around ends
      std::vector< std::vector<float> > maxpts_v(2*adc_v.size());
      for (size_t p=0; p< adc_v.size(); p++ ) {
        // get the maximum within the window, must be above threshold given. if nothing found, returns empty vector.
        maxpts_v[2*p]   = _findMaxChargePixel( shower.pca_ends_v[0], adc_v[p], 20, 100 );
        maxpts_v[2*p+1] = _findMaxChargePixel( shower.pca_ends_v[1], adc_v[p], 20, 100 );        
      }

      // can we make a 3d consistent point with the maxima?
      std::vector< std::vector<float> > vertex_pts;
      
      for (int ise=0; ise<2; ise++ )  {
        // we loop over the start and end point check
        for ( int i=0; i<adc_v.size(); i++ ) {

          // top points
          std::vector<float> coord_i = maxpts_v[ 2*i+ise ];
          if ( coord_i.size()==0 ) continue; // nothing was found
        
          for (int j=i+1; j<adc_v.size(); j++ ) {
            std::vector<float> coord_j = maxpts_v[ 2*j+ise ];
            if ( coord_j.size()==0 ) continue; // nothing was found 

            // compare tick
            float tick_diff = fabs(coord_i[0]-coord_j[0]);
            if ( tick_diff<meta.pixel_height() ) {

              // close enough. get 3D position from wire intersection
              std::vector< float > poszy;
              int crosses;
              int otherplane;
              int otherwire;
              ublarcvapp::UBWireTool::getMissingWireAndPlane( i, meta.col(coord_i[1],__FILE__,__LINE__),
                                                              j, meta.col(coord_j[1],__FILE__,__LINE__),
                                                              otherplane, otherwire, poszy, crosses );


              // pick the tick, we're going to use
              int pixradius = -1;
              int use_tick = -1;
              if ( i==2 ) {
                use_tick = (int)coord_i[0];
                pixradius = coord_i[3];
              }
              else if ( j==2 ) {
                use_tick = (int)coord_j[0];
                pixradius = coord_j[3];                
              }
              else {
                // use the max
                if ( coord_i[2]>coord_j[2] ) {
                  use_tick = (int)coord_i[0];
                  pixradius = coord_i[3];
                }
                else {
                  use_tick = (int)coord_j[0];
                  pixradius = coord_j[3];
                }
              }
              
              // store info for this vertex: (wire1,wire2,wire3,tick,posx,poy,posz)
              std::vector<float> result(8,0);
              result[ i ] = coord_i[1];
              result[ j ] = coord_j[1];
              result[ otherplane ] = (float)otherwire;
              result[ 3 ] = use_tick;
              result[ 4 ] = ((float)use_tick-3200.0)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
              result[ 5 ] = poszy[1];
              result[ 6 ] = poszy[0];
              result[ 7 ] = showeridx;

              // check for duplicate
              std::vector<int> icoord(4);
              for (int i=0; i<4; i++ ) icoord[i] = result[i];
              if ( past_coord.find(icoord)!=past_coord.end() )
                continue;

              past_coord.insert( icoord );

              std::vector<float> adc_vals(3,0);
              for (int p=0; p<3; p++ )
                adc_vals[p] = adc_v[p].pixel( meta.row( result[3], __FILE__, __LINE__ ),
                                              meta.col( result[p], __FILE__, __LINE__ ) );
                                              
              
              vertex_pts.push_back( result );
              std::cout << "[Candidate Vertex Activity: showeridx=" << showeridx << " end=" << ise << "]" << std::endl;
              std::cout << "  shower size: " << shower.points_v.size() << std::endl;
              std::cout << "  imgcoord: (" << (int)result[0] << "," << (int)result[1] << "," << result[2] << ") " << std::endl;
              std::cout << "  tick: " << (int)use_tick << std::endl;
              std::cout << "  pos: (" << result[4] << "," << result[5] << "," << result[6] << ") " << std::endl;
              std::cout << "  tick diff: " << tick_diff << " ticks" << std::endl;
              std::cout << "  pixel radius: " << pixradius << std::endl;
              std::cout << "  adc: (" << adc_vals[0] << "," << adc_vals[1] << "," << adc_vals[2] << ")" << std::endl;

              Candidate_t vtx;
              vtx.type = kShowerWithVertexActivity;
              vtx.pos.resize(3);
              vtx.imgcoord.resize(4);
              for (int i=0; i<3; i++ ) vtx.pos[i] = result[ 4+i ];
              for (int i=0; i<4; i++ ) vtx.imgcoord[i] = result[ i ];
              vtx.cluster_v.push_back( shower );
              vtx.prong_v.push_back( kShowerProng );
              candidate_v.emplace_back( std::move(vtx) );
              
              
            }
          }//end of j plane loop
        }//end of i plane loop
      }//end of loop over ends
      
    }//end of loop over shower


    return candidate_v;
  }

  /**
   * find maximum ADC value near end point
   *
   */
  std::vector<float> VertexReco::_findMaxChargePixel( const std::vector<float>& pt3d,
                                                      const larcv::Image2D& adc,
                                                      const int boxradius,
                                                      const float pixel_threshold ) {
    // project into plane
    double dpos[3] = { pt3d[0], pt3d[1], pt3d[2] };
    int tick = 3200 + pt3d[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5;
    int wire = larutil::Geometry::GetME()->WireCoordinate( dpos, adc.meta().plane() )+0.5;

    if ( tick<=adc.meta().min_y() || tick>=adc.meta().max_y()
         || wire<0 || wire>=adc.meta().max_x() ) {
      std::vector<float> empty;
      return empty;
    }

    int row = adc.meta().row( tick, __FILE__, __LINE__ );
    int col = adc.meta().col( wire, __FILE__, __LINE__  );

    int max_r = -1;
    int max_c = -1;
    float max_pixval = 0;
    for ( int r=row-boxradius; r<=row+boxradius; r++ ) {
      if ( r<0 || r>=(int)adc.meta().rows() ) continue;
      for (int c=col-boxradius; c<=col+boxradius; c++) {
        if ( c<0 || c>=(int)adc.meta().cols() ) continue;

        float pixval = adc.pixel(r,c);
        if ( pixval>max_pixval && pixval>pixel_threshold ) {
          max_pixval = pixval;
          max_r = r;
          max_c = c;
        }
      }
    }

    std::vector<float> result;
    if ( max_r>=0 && max_c>=0 ) {
      result.resize(4);
      result[0] = adc.meta().pos_y(max_r);
      result[1] = adc.meta().pos_x(max_c);
      result[2] = max_pixval;
      result[3] = sqrt( float(max_r-row)*float(max_r-row) + float(max_c-col)*float(max_c-col) );
    }

    // empty if no maximum found
    return result;
  }

  /**
   * merge showers and tracks for shower end search
   *
   */
  std::vector< cluster_t > VertexReco::_mergeShowerClusters( const std::vector<cluster_t>& shower_v,
                                                             const float max_endpt_dist ) {

    std::vector< cluster_t > merged_v;

    // sort by size
    std::vector< std::pair<int,int> > index_size_v;
    index_size_v.reserve( shower_v.size() );
    for ( size_t idx=0; idx<shower_v.size(); idx++ ) {
      index_size_v.push_back( std::pair<int,int>( shower_v[idx].points_v.size(), idx ) );
    }
    std::sort( index_size_v.begin(), index_size_v.end() );

    int nmerged = 0;
    std::clock_t begin = std::clock();
    
    std::vector<int> used_v( shower_v.size(), 0 );    
    for ( auto const& ordinal1 : index_size_v ) {
      int idx  = ordinal1.second;
      int npts = ordinal1.first;
      if (npts<10 ) continue;
      if (used_v[idx]==1) continue;

      cluster_t shower1 = shower_v[ idx ];

      std::cout << "[VertexReco::_mergeShowerClusters] merge into shower[" << idx << "] size=" << npts << std::endl;
      
      for ( auto const& ordinal2 : index_size_v ) {
        //if ( ordinal2.first<10 ) continue;
        if ( used_v[ordinal2.second]==1 ) continue;
        if ( ordinal2.second==idx ) continue;

        auto& shower2 = shower_v[ordinal2.second];

        // // bbox test first
        // if ( !cluster_endpt_in_bbox( shower1, shower2 ) ) {
        //   continue;
        // }

        // test end point dist
        std::vector< std::vector<float> > endpts;
        float endptdist = cluster_closest_endpt_dist( shower1, shower2, endpts );

        if ( endptdist>max_endpt_dist ) continue;

        std::cout << "[VertexReco::_mergeShowerClusters]   merge shower[" << ordinal2.second << "] size=" << ordinal1.second << " endptdist=" << endptdist << std::endl;

        // merge
        cluster_t merge = cluster_merge( shower1, shower2 );
        used_v[idx] = 1;
        used_v[ordinal2.second] = 1;
        nmerged++;
        std::swap(shower1,merge);

      }

      merged_v.emplace_back( std::move(shower1) );
    }
    
    std::clock_t end = std::clock();
    float elapsed = float( end-begin )/CLOCKS_PER_SEC;
    std::cout << "[VertexReco::_mergeShowerClusters] in=" << shower_v.size()
              << " nout=" << merged_v.size()
              << " nmerged=" << nmerged
              << " elapsed=" << elapsed << " secs"
              << std::endl;
    
    
    return merged_v;

  }

  
  /**
   * merge showers and tracks for shower end search
   *
   */
  std::vector< cluster_t > VertexReco::_mergeShowerAndTrackClusters( const std::vector<cluster_t>& shower_v,
                                                                     const std::vector<cluster_t>& track_v ) {

    std::vector< cluster_t > merged_v;

    int idx=-1;
    for ( auto const& shower: shower_v ) {
      idx++;
      if ( shower.points_v.size()<=10 ) continue;

      cluster_t xshower = shower;
      
      for ( auto const& track : track_v ) {

        if ( track.pca_len>20.0 ) continue;
        
        // bbox test first
        if ( !cluster_endpt_in_bbox( xshower, track ) ) {
          continue;
        }

        // test direction
        float cospca = cluster_cospca( xshower, track );
        if ( fabs(cospca)<0.5 ) {
          continue;
        }

        // merge
        cluster_t merge = cluster_merge( xshower, track );
        std::cout << "[VertexReco] merging track into shower[" << idx << "]" << std::endl;
        std::cout << "  original ends: (" << xshower.pca_ends_v[0][0] << "," << xshower.pca_ends_v[0][1] << "," << xshower.pca_ends_v[0][2] << ") "
                  << "--> (" << xshower.pca_ends_v[1][0] << "," << xshower.pca_ends_v[1][1] << "," << xshower.pca_ends_v[1][2] << ") "
                  << std::endl;
        std::cout << "  merged ends: (" << merge.pca_ends_v[0][0] << "," << merge.pca_ends_v[0][1] << "," << merge.pca_ends_v[0][2] << ") "
                  << "--> (" << merge.pca_ends_v[1][0] << "," << merge.pca_ends_v[1][1] << "," << merge.pca_ends_v[1][2] << ") "
                  << std::endl;
        std::swap(merge,xshower);
      }
      std::cout << "  adding merged shower; mergedidx[" << merged_v.size() << "]" << std::endl;

      merged_v.emplace_back( std::move(xshower) );
    }

    return merged_v;

  }
  

  /**
   * dump list of vertex candidates into json file
   *
   */
  void VertexReco::dumpCandidates2json( const std::vector< VertexReco::Candidate_t >& vtx_v, std::string outfile ) {

    nlohmann::json j = dump2json( vtx_v );
    std::ofstream o(outfile.c_str());
    j>>o;
    o.close();
    
  }

  /**
   * convert list into json format
   *
   */
  nlohmann::json VertexReco::dump2json( const std::vector< VertexReco::Candidate_t >& vtx_v ) {
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
      jvtx["type"]     = (int)vtx.type;
      jvtx["clusters"] = jcluster_v;
      jvtx["pos"]      = vtx.pos;
      jvtx["cluster_types"] = ctype_v;

      jvtx_v.emplace_back( std::move(jvtx) );
    }

    j["vertices"] = jvtx_v;
    return j;
  }

}
}
