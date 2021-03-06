#include "PCACluster.h"

#include "cluster_functions.h"

#include "nlohmann/json.hpp"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "ublarcvapp/dbscan/DBScan.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "TRandom3.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <ctime>

namespace larflow {
namespace reco {

  void PCACluster::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {

    larlite::event_larflow3dhit* ev_lfhits
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_lfhit_tree_name );

    larcv::EventImage2D* ev_ssnet_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "ubspurn_plane%d", (int)p );
      ev_ssnet_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();

    // collect track images
    std::vector<larcv::Image2D> ssnet_trackimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_trackimg_v.push_back(ev_ssnet_v[p]->Image2DArray()[1]);

    // collect shower images
    std::vector<larcv::Image2D> ssnet_showerimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_showerimg_v.push_back(ev_ssnet_v[p]->Image2DArray()[0]);

    // containers for track and shower hits
    std::vector<larlite::larflow3dhit> track_hit_v;
    std::vector<larlite::larflow3dhit> shower_hit_v;

    // divide pixels by track and shower
    larflow::reco::cluster_splitbytrackshower( *ev_lfhits, ssnet_trackimg_v, track_hit_v, shower_hit_v, _min_larmatch_score );

    // cluster track hits
    std::vector<int> used_hits_v;
    std::vector<cluster_t> cluster_track_v;
    multipassCluster( track_hit_v, adc_v, cluster_track_v, used_hits_v );

    std::vector<larflow::reco::cluster_t> cluster_shower_v;
    larflow::reco::cluster_sdbscan_larflow3dhits( shower_hit_v, cluster_shower_v, 20.0, 5, 5 );
    larflow::reco::cluster_runpca( cluster_shower_v );

    std::vector<cluster_t> final_v;
    for ( auto& c : cluster_track_v )
      final_v.push_back(c);
    for ( auto& c : cluster_shower_v )
      final_v.push_back(c);

    
    // form clusters of larflow hits for saving
    larlite::event_larflowcluster* evout_lfcluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "pcacluster" );
    larlite::event_pcaxis*         evout_pcaxis    = (larlite::event_pcaxis*)        ioll.get_data( larlite::data::kPCAxis,         "pcacluster" );
    int cidx = 0;
    for ( auto& c : cluster_track_v ) {
      // cluster of hits
      larlite::larflowcluster lfcluster = makeLArFlowCluster( c, ssnet_showerimg_v, ssnet_trackimg_v, adc_v, track_hit_v );
      evout_lfcluster->emplace_back( std::move(lfcluster) );
      // pca-axis
      larlite::pcaxis llpca = cluster_make_pcaxis( c, cidx );
      evout_pcaxis->push_back( llpca );
      cidx++;
    }
    // for ( auto& c : cluster_shower_v ) {
    //   larlite::larflowcluster lfcluster = makeLArFlowCluster( c, ssnet_showerimg_v, ssnet_trackimg_v, adc_v, shower_hit_v );
    //   evout_lfcluster->emplace_back( std::move(lfcluster) );
    //   // pca-axis
    //   larlite::pcaxis llpca = cluster_make_pcaxis( c, cidx );
    //   evout_pcaxis->push_back( llpca );
    //   cidx++;      
    // }

    // form clusters of larflow hits for saving
    larlite::event_larflowcluster* evout_shower_lfcluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfshower" );
    larlite::event_pcaxis*         evout_shower_pcaxis    = (larlite::event_pcaxis*)        ioll.get_data( larlite::data::kPCAxis,         "lfshower" );    
    //evout_shower_lfcluster->reserve( cluster_shower_v.size() );
    cidx = 0;
    for ( auto& cluster : cluster_shower_v ) {
      larlite::larflowcluster lfcluster = makeLArFlowCluster( cluster, ssnet_showerimg_v, ssnet_trackimg_v, adc_v, shower_hit_v );
      evout_shower_lfcluster->emplace_back( std::move(lfcluster) );
      // pca-axis
      larlite::pcaxis llpca = cluster_make_pcaxis( cluster, cidx );
      evout_shower_pcaxis->push_back( llpca );
      cidx++;
    }//end of cluster loop

    // make noise cluster
    larlite::event_larflowcluster* evout_noise_lfcluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfnoise" );    
    larlite::event_pcaxis*         evout_noise_pcaxis    = (larlite::event_pcaxis*)        ioll.get_data( larlite::data::kPCAxis,         "lfnoise" );
    larlite::larflowcluster lfnoise;
    cluster_t noise_cluster;
    for ( size_t i=0; i<track_hit_v.size(); i++ ) {
      auto& hit = track_hit_v[i];
      if ( used_hits_v[i]==0 ) {
        std::vector<float> pt = { hit[0], hit[1], hit[2] };
        std::vector<int> coord_v = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
        noise_cluster.points_v.push_back( pt  );
        noise_cluster.imgcoord_v.push_back( coord_v );
        noise_cluster.hitidx_v.push_back( i );
        lfnoise.push_back( hit );
      }
    }
    cluster_pca( noise_cluster );
    larlite::pcaxis noise_pca = cluster_make_pcaxis( noise_cluster );
    evout_noise_pcaxis->push_back( noise_pca );
    evout_noise_lfcluster->push_back( lfnoise );
    
  }

  /* 
   * split clusters using 2D contours.
   * 
   * Is a static function, allowing other routines to use this algorithm.
   *
   * @param[in] min_second_pca_len Minimum second PC axis eigenvalue to perform split. Else returns cluster as is.
   *
   */
  int PCACluster::split_clusters( std::vector<cluster_t>& cluster_v,
                                  const std::vector<larcv::Image2D>& adc_v,
                                  const float min_second_pca_len ) {

    std::cout << "[PCACluster::split_clusters] start" << std::endl;
    std::clock_t begin = std::clock();
    
    // allocate output vector of clusters
    std::vector<cluster_t> out_v;

    // for debug
    //std::vector<cluster_t> tmp;

    // allocate an array of blank images for 2D contouring purposes
    std::vector<larcv::Image2D> projimg_v;
    for ( auto const& img : adc_v ) {
      larcv::Image2D proj(img.meta());
      proj.paint(0);
      projimg_v.emplace_back( std::move(proj) );
    }

    int nsplit = 0;
    
    // loop over 3d track clusters.
    // we split clusters with a large 2nd PCA-axis    
    for ( size_t i=0; i<cluster_v.size(); i++ ) {
      auto& clust = cluster_v[i];
      std::cout << "  track cluster[" << i << "] pca axis: "
                << " [0]=" << clust.pca_eigenvalues[0] 
                << " [1]=" << clust.pca_eigenvalues[1]
                << " [2]=" << clust.pca_eigenvalues[2]
                << std::endl;
      if ( clust.pca_eigenvalues[1]<min_second_pca_len ) {
        // cluster is line-enough, just pass on
        out_v.emplace_back( std::move(clust) );
        continue;
      }


      // we split this contour (or at least try)
      // populate image with contour
      larflow::reco::cluster_imageprojection( clust, projimg_v );

      // make contours
      ublarcvapp::ContourClusterAlgo contour_algo;
      contour_algo.analyzeImages( projimg_v, 10.0, 2, 5, 10, 10, 2 );

      // we sort the contours by length
      // to do so we need to compile info on them
      struct cdata {
        int index;
        int plane;
        float length;
        bool operator<( const cdata& rhs ) const {
          if ( length>rhs.length )
            return true;
          return false;
        };
      };

      std::vector< cdata > contour_order;
      int ntot = 0;
      for ( size_t p=0; p<3; p++ ) { 
        int ncontours = contour_algo.m_plane_atomics_v[p].size();
        ntot += ncontours;
      }
      contour_order.reserve( ntot );
           
      for ( size_t p=0; p<3; p++ ) {
        int ncontours = contour_algo.m_plane_atomics_v[p].size();
        
        for (int ictr=0; ictr<ncontours; ictr++ ) {
          if ( !contour_algo.m_plane_atomicmeta_v[p][ictr].hasValidPCA() ) {
            //std::cout << "  icontour[" << ictr << "] pca not valid" << std::endl;
            continue;
          }

          // skip contours with a large 2nd pc-axis
          if ( contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAeigenvalue(1)>5.0 ) continue;
          
          // find max-dist from start-end of pc-axis
          float pca_dist = 0;
          float dx = 0;
          for ( size_t i=0; i<2; i++ ) {
            dx = ( contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAEndPos()[i] -
                   contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAStartPos()[i] );
            pca_dist += dx*dx;
          }
          pca_dist = sqrt(pca_dist);
          std::cout << "  icontour[" << ictr << "] plane=" << p 
                    << " pca-dist=" << pca_dist
                    << " eigenvals [0]=" << contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAeigenvalue(0)
                    << " [1]=" << contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAeigenvalue(1)
                    << std::endl;

          cdata c;
          c.index = ictr;
          c.plane = p;
          c.length = pca_dist;

          contour_order.push_back(c);
            
        }//end of contour loop
      }//end of plane loop

      std::sort( contour_order.begin(), contour_order.end() );

      // loop through contours using 'contour_order'
      // for each contour, check if 3d points from cluster is inside it.
      // make 3d cluster for contour. if 3d cluster is straight enough,
      // those points are claimed
      std::vector<int> claimedpts( clust.points_v.size(), 0 );
      int totclaimed = 0;
      int nnewclusters = 0;

      for ( auto const& c : contour_order ) {

        // get contour
        auto const& contour2d    = contour_algo.m_plane_atomics_v[c.plane][c.index];
        auto const& contourmeta = contour_algo.m_plane_atomicmeta_v[c.plane][c.index];
      
        // we now collect the 3d points from this contour
        cluster_t contourcluster;
        contourcluster.points_v.reserve( clust.imgcoord_v.size() );
        contourcluster.imgcoord_v.reserve( clust.imgcoord_v.size() );
        contourcluster.hitidx_v.reserve( clust.imgcoord_v.size() );

        std::vector<int> candidate_idx;
        
        for ( size_t idx=0; idx<clust.imgcoord_v.size(); idx++ ) {

          if ( claimedpts[idx]==1 ) continue; // don't reuse the hits
          
          const std::vector<int>& imgcoord = clust.imgcoord_v[idx];
          int row = projimg_v[c.plane].meta().row( imgcoord[3] );
          bool incontour = false;
          int col = projimg_v[c.plane].meta().col( imgcoord[c.plane] );
          // test if inside contour
          if ( col>=contourmeta.getMinX() && col<=contourmeta.getMaxX()
               && row>=contourmeta.getMinY() && row<=contourmeta.getMaxY() ) {
            // inside bounding box, so test contour
            int result = cv::pointPolygonTest( contour2d, cv::Point2f( (float)col, (float)row ), false );
            if ( result>=0 ) {
              // include in new cluster
              contourcluster.points_v.push_back(   clust.points_v[idx] );
              contourcluster.imgcoord_v.push_back( clust.imgcoord_v[idx] );
              contourcluster.hitidx_v.push_back( clust.hitidx_v[idx] );
              incontour = true;
              candidate_idx.push_back(idx);
            }
          }
        } //end of loop over points in old cluster

        std::cout << "  [ctr=" << c.index << " plane=" << c.plane << " length=" << c.length << " npts=" << contourcluster.points_v.size() << "]" << std::endl;
        
        // check if big enough
        if ( contourcluster.points_v.size()<5 )
          continue;
        
        // get pca of new cluster
        larflow::reco::cluster_pca( contourcluster );

        std::cout << "     pca-eigenvalue[1]=" << contourcluster.pca_eigenvalues[1] << std::endl;

        if ( contourcluster.pca_eigenvalues[1]<5.0 ) {
          // success!
          for ( auto& idx : candidate_idx ) {
            claimedpts[idx] = 1;
            totclaimed+=1;
          }
          //tmp.push_back( contourcluster );          
          out_v.emplace_back( std::move(contourcluster) );
          nnewclusters++;
        }


        if ( totclaimed==claimedpts.size() )
          break;
        
      }//end of loop over contours

      std::cout << "after split: hits claimed=" << totclaimed << " of " << claimedpts.size() << std::endl;

      if ( nnewclusters>0 )
        nsplit++;

      if ( totclaimed+5<claimedpts.size() ) {
        // make unclaimed cluster
        cluster_t unclaimedcluster;
        for ( size_t idx=0; idx<claimedpts.size(); idx++ ) {
          if ( claimedpts[idx]==0 ) {
            unclaimedcluster.points_v.push_back(   clust.points_v[idx] );
            unclaimedcluster.imgcoord_v.push_back( clust.imgcoord_v[idx] );
            unclaimedcluster.hitidx_v.push_back( clust.hitidx_v[idx] );
          }
        }
        larflow::reco::cluster_pca( unclaimedcluster );
        //tmp.push_back( unclaimedcluster );
        out_v.emplace_back( std::move(unclaimedcluster) );
      }

    }//loop over clusters

    //larflow::reco::cluster_dump2jsonfile( tmp, "dump_split.json" );

    std::swap( out_v, cluster_v );
    //end of split cluster

    std::clock_t end = std::clock();
    float elapsed = float( end-begin )/CLOCKS_PER_SEC;
    std::cout << "[PCACluster::split_clusters] end; elapsed=" << elapsed << " secs" << std::endl;


    return nsplit;
  }
    
  int PCACluster::merge_clusters( std::vector<cluster_t>& cluster_v,
                                  const std::vector<larcv::Image2D>& adc_v,
                                  float max_dist_cm, float min_angle_deg, float max_pca2,
                                  bool print_tests ) {

    float min_angle_rad = 3.14159*min_angle_deg/180.0;
    
    // allocate output vector of clusters
    std::vector<cluster_t> out_v;

    std::vector<int> merged( cluster_v.size(), 0);
    
    for (int i=0; i<cluster_v.size(); i++ ) {
      if ( merged[i]==1 ) continue;
      auto& clust_i = cluster_v[i];
      
      for (int j=i+1; j<cluster_v.size(); j++ ) {
        if ( merged[j]==1 ) continue;
        auto& clust_j = cluster_v[j];

        std::vector< std::vector<float> > endpts_v;
        float endptdist = cluster_closest_endpt_dist( clust_i, clust_j, endpts_v );
        float cospca    = cluster_cospca( clust_i, clust_j );
        cospca = 1.0 - fabs(cospca);

        if ( print_tests && endptdist<50.0 )
          std::cout << "merge-test [" << i << "][" << j << "] endptdist=" << endptdist << " cos-pca=" << cospca*180.0/3.14159 << " deg";

        if ( endptdist<max_dist_cm  && fabs(cospca)<min_angle_rad ) {

          // create merger cluster
          cluster_t merge = cluster_merge( clust_i, clust_j );

          if ( print_tests  )
            std::cout << " pca[1]=" << merge.pca_eigenvalues[1] << std::endl;
                  

          // test pca of merged cluster
          if ( merge.pca_eigenvalues[1]<max_pca2 ) {
            // good!
            out_v.emplace_back( std::move(merge) );
            merged[i] = 1;
            merged[j] = 1;
            break; // do not remerge this
          }

        }
        else {
          if ( print_tests && endptdist<50.0 )
            std::cout << std::endl;
        }
      }//end of j cluster
    }//end of i cluster loop

    int nmerged = 0;
    for ( size_t i=0; i<merged.size(); i++ ) {
      if ( merged[i]==0 ) {
        out_v.emplace_back( std::move(cluster_v[i]) );
      }
      else {
        nmerged++;
      }
    }

    std::swap(out_v,cluster_v);
    
    return nmerged;
  }

  /**
   * recluster clusters with large 2nd pca eigenvalues
   *
   * this is meant to clean up clusters that have been split into many pieces which sometimes leaves 
   *  weird clusters behind.
   * 
   */
  void PCACluster::defragment_clusters( std::vector<cluster_t>& cluster_v, const float max_2nd_pca_eigenvalue ) {

    int nsplit = 0;
    
    std::vector<cluster_t> out_v;
    for ( auto& cluster : cluster_v ) {

      if ( cluster.pca_eigenvalues[1]<max_2nd_pca_eigenvalue ) {
        // move on
        out_v.emplace_back( std::move(cluster) );
      }
      else {
        // we re-cluster
        std::vector< ublarcvapp::dbscan::dbCluster > dbcluster_v = ublarcvapp::dbscan::DBScan::makeCluster3f( 5.0, 5, 5, cluster.points_v );

        if ( dbcluster_v.size()==1 ) {
          // didnt find any clusters
          out_v.emplace_back( std::move(cluster) );
          continue;
        }

        // we reclustered
        nsplit++;

        for (int ic=0; ic<(int)dbcluster_v.size()-1; ic++ ) {

          auto& dbclust = dbcluster_v[ic];
          
          cluster_t c;
          c.points_v.reserve( dbclust.size() );
          c.imgcoord_v.reserve( dbclust.size() );
          c.hitidx_v.reserve( dbclust.size() );
          for ( auto const& hitidx : dbclust ) {
            c.points_v.push_back( cluster.points_v[hitidx] );
            c.imgcoord_v.push_back( cluster.imgcoord_v[hitidx] );
            c.hitidx_v.push_back( cluster.hitidx_v[hitidx] );
          }
          
          cluster_pca( c );
          out_v.emplace_back( std::move(c) );
          
        }//end of loop over new clusters
        
      }//end of else recluster
    }//end of loop over clusters
    
    std::swap(out_v,cluster_v);

  }

  larlite::larflowcluster PCACluster::makeLArFlowCluster( cluster_t& cluster,
                                                          const std::vector<larcv::Image2D>& ssnet_showerimg_v,
                                                          const std::vector<larcv::Image2D>& ssnet_trackimg_v,
                                                          const std::vector<larcv::Image2D>& adc_v,
                                                          const std::vector<larlite::larflow3dhit>& source_lfhit_v ) {
    
    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    for ( size_t ii=0; ii<cluster.ordered_idx_v.size(); ii++ ) {

      int ihit = cluster.ordered_idx_v[ii];
        
      larlite::larflow3dhit lfhit;
      lfhit.resize(7,0); // (x,y,z,pixU,pixV,pixY,matchprob)

      int row = adc_v.front().meta().row( cluster.imgcoord_v[ihit][3], __FILE__, __LINE__ );
      
      // transfer 3D points
      for (int i=0; i<3; i++) lfhit[i] = cluster.points_v[ihit][i];

      // transfer charge
      for (int i=0; i<3; i++) {
        int col = adc_v.front().meta().col( cluster.imgcoord_v[ihit][i], __FILE__, __LINE__ );
        lfhit[3+i] = adc_v[i].pixel( row, col );
      }

      // transfer larmatch probability
      lfhit[6] = source_lfhit_v[ cluster.hitidx_v[ihit] ].track_score; // yeah, sorry for ad-hocness

      // transfer image coordates
      lfhit.srcwire = cluster.imgcoord_v[ihit][2];
      lfhit.targetwire.resize(3,0);
      lfhit.targetwire[0] = cluster.imgcoord_v[ihit][0];
      lfhit.targetwire[1] = cluster.imgcoord_v[ihit][1];
      lfhit.targetwire[2] = cluster.imgcoord_v[ihit][2];
      lfhit.tick          = cluster.imgcoord_v[ihit][3];

      // get highest shower and track scores
      float shower_max = 0.;
      float track_max  = 0.;

      const larcv::ImageMeta& meta = ssnet_showerimg_v.front().meta();

      if ( lfhit.tick>meta.min_y() && lfhit.tick<meta.max_y() ) {
        int row = meta.row((float)lfhit.tick,__FILE__,__LINE__);
        for (size_t p=0; p<3; p++) {

          int wire = lfhit.targetwire[p];
          if ( wire<meta.min_x() || wire>=meta.max_x() ) continue;

          int col  = meta.col(wire,__FILE__,__LINE__);

          float sh_score = ssnet_showerimg_v[p].pixel( row, col );
          float tr_score = ssnet_trackimg_v[p].pixel( row, col );

          if ( sh_score>shower_max ) shower_max = sh_score;
          if ( tr_score>track_max  ) track_max  = tr_score;

        }
      }

      lfhit.track_score  = track_max;
      lfhit.shower_score = shower_max;

      lfcluster.emplace_back( std::move(lfhit) );
    }//end of hit loop
    
    return lfcluster;
  }

  cluster_t PCACluster::absorb_nearby_hits( const cluster_t& cluster,
                                            const std::vector<larlite::larflow3dhit>& hit_v,
                                            std::vector<int>& used_hits_v,
                                            float max_dist2line ) {

    cluster_t newcluster;
    int nused = 0;
    for ( size_t ihit=0; ihit<hit_v.size(); ihit++ ) {

      auto const& hit = hit_v[ihit];

      if ( used_hits_v[ ihit ]==1 ) continue;
      
      // apply quick bounding box test
      if ( hit[0] < cluster.bbox_v[0][0] || hit[0]>cluster.bbox_v[0][1]
           || hit[1] < cluster.bbox_v[1][0] || hit[1]>cluster.bbox_v[1][1]
           || hit[2] < cluster.bbox_v[2][0] || hit[2]>cluster.bbox_v[2][1] ) {
        continue;
      }

      // else calculate distance from pca-line
      float dist2line = cluster_dist_from_pcaline( cluster, hit );

      if ( dist2line < max_dist2line ) {
        std::vector<float> pt = { hit[0], hit[1], hit[2] };
        std::vector<int> coord_v = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
        newcluster.points_v.push_back( pt );
        newcluster.imgcoord_v.push_back( coord_v );
        newcluster.hitidx_v.push_back( ihit );
        used_hits_v[ihit] = 1;
        nused++;
      }
      
    }

    if (nused>=10 ) {
      std::cout << "[absorb_nearby_hits] cluster absorbed " << nused << " hits" << std::endl;      
      cluster_pca( newcluster );
    }
    else {
      // throw them back
      std::cout << "[absorb_nearby_hits] cluster hits " << nused << " below threshold" << std::endl;            
      for ( auto& idx : newcluster.hitidx_v )
        used_hits_v[idx] = 0;
      newcluster.points_v.clear();
      newcluster.imgcoord_v.clear();
      newcluster.hitidx_v.clear();
    }

    
    return newcluster;
  }

  void PCACluster::multipassCluster( const std::vector<larlite::larflow3dhit>& inputhits,
                                     const std::vector<larcv::Image2D>& adc_v,
                                     std::vector<cluster_t>& output_cluster_v,
                                     std::vector<int>& used_hits_v ) {
    const int max_passes = 3;
    const int max_pts_to_cluster = 30000;
    
    TRandom3 rand(12345);

    used_hits_v.resize( inputhits.size(), 0 );

    output_cluster_v.clear();
    
    for ( int ipass=0; ipass<max_passes; ipass++ ) {

      // count points remaining      
      int total_pts_remaining = 0;
      if ( ipass==0 ) {
        // on first pass, all points remain
        total_pts_remaining = (int)inputhits.size();
      }
      else {
        // on rest of passes, we count
        for ( auto const& used : used_hits_v ) {
          if ( used==0 ) {
            total_pts_remaining++;
          }
        }
      }

      // downsample points, if needed
      std::vector<larlite::larflow3dhit> downsample_hit_v;
      downsample_hit_v.reserve( max_pts_to_cluster );

      float downsample_fraction = (float)max_pts_to_cluster/(float)total_pts_remaining;
      if ( total_pts_remaining>max_pts_to_cluster ) {
        for ( size_t ihit=0; ihit<inputhits.size(); ihit++ ) {
          if ( used_hits_v[ihit]==0 ) {
            if ( rand.Uniform()<downsample_fraction ) {
              downsample_hit_v.push_back( inputhits[ihit] );
            }
          }
        }
      }
      else {
        for ( size_t ihit=0; ihit<inputhits.size(); ihit++ ) {
          if ( used_hits_v[ihit]==0 ) {
            downsample_hit_v.push_back( inputhits[ihit] );
          }
        }
      }

      std::cout << "[pass " << ipass << "] remaining hits downsampled to " << downsample_hit_v.size() << " of " << total_pts_remaining << std::endl;

      // cluster these hits
      std::vector<larflow::reco::cluster_t> cluster_pass_v;
      larflow::reco::cluster_sdbscan_larflow3dhits( downsample_hit_v, cluster_pass_v, _maxdist, _minsize, _maxkd ); // external implementation, seems best
      larflow::reco::cluster_runpca( cluster_pass_v );

      // we then absorb the hits around these clusters
      std::vector<larflow::reco::cluster_t> dense_cluster_v;
      for ( auto const& ds_cluster : cluster_pass_v ) {
        cluster_t dense_cluster = absorb_nearby_hits( ds_cluster,
                                                      inputhits,
                                                      used_hits_v,
                                                      10.0 );
        if ( dense_cluster.points_v.size()>0 ) 
          dense_cluster_v.emplace_back( std::move(dense_cluster) );
      }
      int nused_tot = 0;
      for ( auto& used : used_hits_v ) {
        nused_tot += used;
      }
      std::cout << "[PCACluster, pass " << ipass << "] after absorbing hits to sparse clusters: " << nused_tot << " of " << used_hits_v.size() << " all hits" << std::endl;
      
      // we perform split functions on the clusters
      int nsplit = 0;
      for (int isplit=0; isplit<3; isplit++ ) {
        nsplit = split_clusters( dense_cluster_v, adc_v, 10.0 );
        std::cout << "[PCACluster, pass " << ipass << "] splitting, pass" << isplit << ": num split=" << nsplit << std::endl;      
        if (nsplit==0 ) break;
      }

      std::cout << "[PCACluster, pass " << ipass << "] defrag clusters" << std::endl;
      defragment_clusters( dense_cluster_v, 10.0 );

      // now split and merge with pass clusters
      for ( auto& dense : dense_cluster_v ) {
        output_cluster_v.emplace_back( std::move(dense) );
      }
      
      // now we merge
      int nmerged = 0;
      nmerged = merge_clusters( output_cluster_v, adc_v, 30.0, 10.0, 10.0 );
      std::cout << "[PCACluster, pass " << ipass << "] merger-0 maxdist=10.0, maxangle=30.0, maxpca=10.0: number merged=" << nmerged << std::endl;

      nmerged = merge_clusters( output_cluster_v, adc_v, 30.0, 30.0, 10.0 );
      std::cout << "[PCACluster, pass " << ipass << "] merger-1 maxdist=10.0, maxangle=30.0, maxpca=10.0: number merged=" << nmerged << std::endl;
      
      nmerged = merge_clusters( output_cluster_v, adc_v, 30.0, 60.0, 20.0, true );
      std::cout << "[PCACluster, pass " << ipass << "] merger-1 maxdist=10.0, maxangle=30.0, maxpca=10.0: number merged=" << nmerged << std::endl;      
      
    }

    int nused_final = 0;
    for ( auto& used : used_hits_v )
      nused_final += used;
    
    std::cout << "[PCACluster, multipass result] nclusters=" << output_cluster_v.size() << " nused=" << nused_final << std::endl;
    
  }
                                     
  
}
}
