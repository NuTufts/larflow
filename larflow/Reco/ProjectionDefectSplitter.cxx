#include "ProjectionDefectSplitter.h"

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

  /**
   * @brief run algorithm on LArCV/larlite data products for current event
   *
   * Expects the following input data products:
   * @verbatim embed:rst:leading-asterisk
   *  * larflow3dhits in ioll from tree name stored in _input_lfhit_tree_name: These are the hits we will cluster
   *  * wire images from larcv tree named "wire": will use these images to do 2D contour analysis
   * @endverbatim
   *
   * Produces the following output products:
   * @verbatim embed:rst:leading-asterisk
   *  * larflowcluster stored in tree _output_cluster_tree_name: the clusters formed by dbscan and split with contour defect analysis
   *  * pcaxis stored in tree named _output_cluster_tree_name: the principle components of the clusters made
   *  * larflow3dhit container named "projsplitnoise": all the hits that were not clustered.
   * @endverbatim

   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void ProjectionDefectSplitter::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {

    // get hits
    larlite::event_larflow3dhit* ev_lfhits
      = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, _input_lfhit_tree_name );

    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();

    // get keypoint event containers to veto hits (if any trees were specified)
    _event_keypoint_for_veto_v.clear();
    if ( _veto_hits_around_keypoints ) {
      for ( auto const& kptreename : _keypoint_veto_trees_v ) {
        larlite::event_larflow3dhit* ev_keypoint
          = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, kptreename );
        _event_keypoint_for_veto_v.push_back( ev_keypoint );
      }
    }


    // cluster track hits
    std::vector<int> used_hits_v( ev_lfhits->size(), 0 );
    std::vector<cluster_t> cluster_track_v;
    _runSplitter( *ev_lfhits, adc_v, used_hits_v, cluster_track_v );
    
    // form clusters of larflow hits for saving
    larlite::event_larflowcluster* evout_lfcluster
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, _output_cluster_tree_name );
    larlite::event_pcaxis*         evout_pcaxis
      = (larlite::event_pcaxis*)        ioll.get_data( larlite::data::kPCAxis, _output_cluster_tree_name );
    
    int cidx = 0;
    for ( auto& c : cluster_track_v ) {
      // cluster of hits
      larlite::larflowcluster lfcluster = _makeLArFlowCluster( c, *ev_lfhits );
      evout_lfcluster->emplace_back( std::move(lfcluster) );
      // pca-axis
      larlite::pcaxis llpca = cluster_make_pcaxis( c, cidx );
      evout_pcaxis->push_back( llpca );
      cidx++;
    }

    // make noise cluster
    larlite::event_larflow3dhit* evout_noise = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "projsplitnoise" );    
    for ( size_t i=0; i<ev_lfhits->size(); i++ ) {
      if ( used_hits_v[i]==0 ) {
        evout_noise->push_back( ev_lfhits->at(i) );
      }
    }
    
  }

  /**
   * @brief split clusters using 2D contours.
   * 
   * Is a static function, allowing other routines to use this algorithm.
   *
   * @param[in] cluster_v Vector of space point clusters
   * @param[in] adc_v Wire plane images we will project space points into. Will do defect analysis on these 2D images.
   * @param[in] min_second_pca_len Minimum second PC axis eigenvalue to perform split. Else returns cluster as is.
   * @return The number of times a cluster was split
   *
   */
  int ProjectionDefectSplitter::split_clusters( std::vector<cluster_t>& cluster_v,
                                                const std::vector<larcv::Image2D>& adc_v,
                                                const float min_second_pca_len ) {

    LARCV_DEBUG() << "[ProjectionDefectSplitter::split_clusters] start" << std::endl;
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
      std::stringstream ss;
      ss << "  track cluster[" << i << "] pca axis: "
                    << " [0]=" << clust.pca_eigenvalues[0] 
                    << " [1]=" << clust.pca_eigenvalues[1]
                    << " [2]=" << clust.pca_eigenvalues[2];

      if ( clust.pca_eigenvalues[1]<min_second_pca_len ) {
        // cluster is line-enough, just pass on
        ss << " 2nd-pca too small. no split" << std::endl;
        LARCV_DEBUG() << ss.str();
        out_v.emplace_back( std::move(clust) );
        continue;
      }

      LARCV_DEBUG() << ss.str() << " do split" << std::endl;
      
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
    std::cout << "[ProjectionDefectSplitter::split_clusters] end; elapsed=" << elapsed << " secs" << std::endl;


    return nsplit;
  }
    
  /**
   * @brief recluster clusters with large 2nd pca eigenvalues
   *
   * this is meant to clean up clusters that have been split into many pieces which sometimes leaves 
   *  weird clusters behind.
   * 
   * @param[in] cluster_v All the current clusters
   * @param[in] max_2nd_pca_eigenvalue The maximum length of the second largest principle component
   */
  void ProjectionDefectSplitter::_defragment_clusters( std::vector<cluster_t>& cluster_v,
                                                       const float max_2nd_pca_eigenvalue ) {

    int nsplit = 0;
    
    std::vector<cluster_t> out_v;
    for ( auto& cluster : cluster_v ) {

      if ( cluster.pca_eigenvalues[1]<max_2nd_pca_eigenvalue ) {
        // move on
        out_v.emplace_back( std::move(cluster) );
      }
      else {
        // we re-cluster -- should move this to simple dbscan algorithm
        //
        std::vector< ublarcvapp::dbscan::dbCluster > dbcluster_v
          = ublarcvapp::dbscan::DBScan::makeCluster3f( _maxdist, _minsize, _maxkd, cluster.points_v );

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

  /**
   * @brief convert cluster_t instance into a larflowcluster instance
   *
   * @param[in] cluster A cluster_t cluster of spacepoints
   * @param[in] source_lfhit_v The original set of hits used to make the given cluster. 
   *                           The info form the hits are copied into the larflowcluster.
   * @return A larflowcluster instance made from the input cluster_t instance
   *
   */
  larlite::larflowcluster
  ProjectionDefectSplitter::_makeLArFlowCluster( cluster_t& cluster,
                                                 const larlite::event_larflow3dhit& source_lfhit_v ) {
    
    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    for ( size_t ii=0; ii<cluster.ordered_idx_v.size(); ii++ ) {
      int iorder = cluster.ordered_idx_v[ii];
      int hitidx = cluster.hitidx_v[iorder];
      auto const& srchit = source_lfhit_v[hitidx];
      lfcluster.push_back( srchit );
    }//end of hit loop
    
    return lfcluster;
  }

  /**
   * @brief Because cluster is down on downsampled points at times, we absorb hits close to the
   *        cluster from the original set of hits
   *
   * @param[in] cluster Cluster made with downsampled hits
   * @param[in] hit_v   The original (not-downsaampled) set of hits used to make the given cluster
   * @param[in] used_hits_v Vector same size as hit_v where value is 1 if hit is already assigned to cluster.
   *                        Hits assigned by this call of the function to the cluster will have their flags set to 1.
   * @param[in] max_dist2line Maximum distance from a point to the first principle component axis
   *                          of the given cluster.
   * @return A new cluster with additional hits added to the given cluster
   */
  cluster_t ProjectionDefectSplitter::_absorb_nearby_hits( const cluster_t& cluster,
                                                           const std::vector<larlite::larflow3dhit>& hit_v,
                                                           std::vector<int>& used_hits_v,
                                                           float max_dist2line ) {

    cluster_t newcluster;
    int nused = 0;
    for ( size_t ihit=0; ihit<hit_v.size(); ihit++ ) {

      auto const& hit = hit_v[ihit];

      if ( used_hits_v[ ihit ]==1 ) continue;
      
      // apply quick bounding box test
      if (    hit[0] < cluster.bbox_v[0][0]-_maxdist || hit[0]>cluster.bbox_v[0][1]+_maxdist
           || hit[1] < cluster.bbox_v[1][0]-_maxdist || hit[1]>cluster.bbox_v[1][1]+_maxdist
           || hit[2] < cluster.bbox_v[2][0]-_maxdist || hit[2]>cluster.bbox_v[2][1]+_maxdist ) {
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
      //std::cout << "[absorb_nearby_hits] cluster absorbed " << nused << " hits" << std::endl;      
      cluster_pca( newcluster );
    }
    else {
      // throw them back
      //std::cout << "[absorb_nearby_hits] cluster hits " << nused << " below threshold" << std::endl;            
      for ( auto& idx : newcluster.hitidx_v )
        used_hits_v[idx] = 0;
      newcluster.points_v.clear();
      newcluster.imgcoord_v.clear();
      newcluster.hitidx_v.clear();
    }

    
    return newcluster;
  }

  /**
   * @brief run 2d contour defect analysis
   * 
   * Our aim is to produce straight clusters that we can later piece
   * together into tracks.
   *
   * Given a set of hits, we first downsample (if needed).
   * The downsampled hits are clustered.
   * Hits for the clusters are projected into the 2D images.
   * Those 2D pixels are used to make 2d contours.
   * The 2D contours are split using defect analysis.
   * Based on which space points are in which contours, the
   * 3D cluster is split if the first second principle component is smaller.
   * 
   * @param[in] inputhits The input spacepoints in the form or larflow3dhit instances
   * @param[in] adc_v The wire plane images
   * @param[in] used_hits_v Vector same length as inputs which indicates if the hit
   *                        has been assigned to a cluster. This function updates this vector.
   * @param[in] output_cluster_v Vector of output clusters
   */
  void ProjectionDefectSplitter::_runSplitter( const larlite::event_larflow3dhit& inputhits,
                                               const std::vector<larcv::Image2D>& adc_v,
                                               std::vector<int>& used_hits_v,
                                               std::vector<cluster_t>& output_cluster_v )
  {

    const int max_pts_to_cluster = 30000;
    int total_pts = inputhits.size();
    
    TRandom3 rand(12345);
    used_hits_v.resize( total_pts, 0 );
    output_cluster_v.clear();

    // veto hits using keypoints
    int nvetoed = _veto_hits_using_keypoints( inputhits, used_hits_v );
    
    // downsample points, if needed
    std::vector<larlite::larflow3dhit> downsample_hit_v;
    downsample_hit_v.reserve( max_pts_to_cluster );

    float downsample_fraction = (float)max_pts_to_cluster/(float)total_pts;
    bool sample = total_pts>max_pts_to_cluster;

    LARCV_INFO() << "Downsample points: " << sample << ", downsample_fraction=" << downsample_fraction << std::endl;
    
    int nremaining = 0;
    for ( size_t ihit=0; ihit<total_pts; ihit++ ) {

      if ( used_hits_v[ihit]==1 )
        continue; // assigned to cluster

      nremaining++;

      if ( used_hits_v[ihit]==2 ) {
        // keypoint veto
        continue;
      }
      
      if ( !sample || rand.Uniform()<downsample_fraction ) {
        downsample_hit_v.push_back( inputhits[ihit] );
      }
      
    }
    LARCV_INFO() << "Remaining hits, " << nremaining << ", downsampled to " << downsample_hit_v.size() << " of " << total_pts << " total" << std::endl;

    // cluster these hits
    std::vector<larflow::reco::cluster_t> cluster_pass_v;
    larflow::reco::cluster_sdbscan_larflow3dhits( downsample_hit_v, cluster_pass_v, _maxdist, _minsize, _maxkd ); // external implementation, seems best
    larflow::reco::cluster_runpca( cluster_pass_v );

    int nused_final = 0;
    std::vector<larflow::reco::cluster_t> dense_cluster_v;    
    if ( sample ) {

      LARCV_INFO() << "Absorb unused hits" << std::endl;
      
      // we then absorb the hits around these clusters
      for ( auto const& ds_cluster : cluster_pass_v ) {
        cluster_t dense_cluster = _absorb_nearby_hits( ds_cluster,
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
      LARCV_DEBUG() << "After absorbing hits to sparse clusters: " << nused_tot << " of " << total_pts << " all hits" << std::endl;
    }
    else {
      LARCV_INFO() << "Do not absorb. Pass clusters." << std::endl;      
      for ( auto& cluster : cluster_pass_v ) {
        dense_cluster_v.emplace_back( std::move(cluster) );
      }
      nused_final = (int)inputhits.size();      
    }

    // if we vetod hits, we assign those veto hits near the ends of found clusters
    // we recalc the pca for these clusters
    // ============= TO DO ==================
    if ( nvetoed>0 ) {
      // track which clusters we modified
      std::vector<int> modded_cluster( dense_cluster_v.size(), 0 );
      int nclaimed = 0;
      
      // add hits to cluster ends
      for ( int ihit=0; ihit<total_pts; ihit++) {
        if ( used_hits_v[ihit]!=2 )
          continue;

        auto const& hit = inputhits[ihit];
        
        // check to add to clusters
        bool claimed = false;
        for ( int ic=0; ic<(int)dense_cluster_v.size(); ic++ ) {
          auto& cluster = dense_cluster_v[ic];
          for ( auto const& endpt : cluster.pca_ends_v ) {
            float dist = 0.;
            for (int v=0; v<3; v++)
              dist += (endpt[v]-hit[v])*(endpt[v]-hit[v]);
            if (dist<_maxdist*_maxdist ) {
              std::vector<float> pt = { hit[0], hit[1], hit[2] };
              std::vector<int>   imgcoord = { hit.targetwire[0], hit.targetwire[1], hit.targetwire[2], hit.tick };
              cluster.points_v.push_back( pt );
              cluster.imgcoord_v.push_back( imgcoord );
              cluster.hitidx_v.push_back( ihit );
              used_hits_v[ihit] = 1;
              modded_cluster[ic] = 1;
              nclaimed++;
              break;
            }
            if ( claimed )
              break;
          }//end of loop over end points
          if ( claimed )
            break;
        }//end of loop over clusters
      }//end of hit loop
      
      // if we modded any clusters, recalc the pca
      int nmodded = 0;
      for ( int ic=0; ic<(int)dense_cluster_v.size(); ic++ ) {
        if ( modded_cluster[ic]==1 ) {
          auto& cluster = dense_cluster_v[ic];
          larflow::reco::cluster_pca(cluster);
            nmodded++;
        }
      }
      LARCV_INFO() << "Absorbed " << nclaimed << " hits and moddified " << nmodded << " clusters"  << std::endl;
    }//end of if we ha veto'd hits, absorb those hits and veto
      
    // we perform split functions on the clusters
    int nsplit = 0;
    for (int isplit=0; isplit<3; isplit++ ) {
      nsplit = split_clusters( dense_cluster_v, adc_v, 2.0 );
      LARCV_DEBUG() << "Splitting, pass" << isplit << ": num split=" << nsplit << std::endl;      
      if (nsplit==0 ) break;
    }
      
    LARCV_DEBUG() << "Defrag clusters" << std::endl;
    _defragment_clusters( dense_cluster_v, 10.0 );
    
    // now split and merge with pass clusters
    for ( auto& dense : dense_cluster_v ) {
      output_cluster_v.emplace_back( std::move(dense) );
    }
      
    for ( auto& used : used_hits_v )
      nused_final += used;
    
    LARCV_DEBUG() << "nclusters=" << output_cluster_v.size() << " nused=" << nused_final << std::endl;
    
  }

  /**
   * @brief Add tree name to get keypoints for vetoing hits
   * 
   * If tree names are provided, the keypoints are used to veto nearby hits.
   * This is done to help break-up particle clusters.
   * When this method is called, _veto_hits_around_keypoints, is set to true.
   * 
   * @param[in] name 
   *
   */
  void ProjectionDefectSplitter::add_input_keypoint_treename_for_hitveto( std::string name )
  {
    _veto_hits_around_keypoints = true;
    _keypoint_veto_trees_v.push_back(name);
  }

  /**
   * @brief veto hits near keypoints
   *
   * we veto hits with _maxdist of a keypoint.
   * we veto by marking hits in the used hits vector.
   *
   * @param[in]  inputhits   Container of input hits
   * @param[out] used_hits_v Vector used to flag/veto hits
   */
  int ProjectionDefectSplitter::_veto_hits_using_keypoints( const larlite::event_larflow3dhit& inputhits,
                                                             std::vector<int>& used_hits_v )
  {

    float max_dist_sq = _maxdist*_maxdist;
    int nhits_vetoed = 0;
    for ( int ihit=0; ihit<(int)inputhits.size(); ihit++ ) {

      if ( used_hits_v[ihit]!=0 )
        continue;
      
      auto const& lfhit = inputhits[ihit];
      
      bool veto_hit = false;
      for ( auto const& pev_keypoint : _event_keypoint_for_veto_v ) {
        for ( auto const& kphit : *pev_keypoint ) {

          float dist = 0.;
          for (int i=0; i<3; i++) {
            dist += (lfhit[i]-kphit[i])*(lfhit[i]-kphit[i]);
          }
          if ( dist<max_dist_sq+0.3 ) {
            // we veto this hit
            veto_hit = true;
          }
          if ( veto_hit )
            break;
        }
        if ( veto_hit )
          break;
      }

      if ( veto_hit ) {
        nhits_vetoed += 1;
        used_hits_v[ihit] = 2;
      }
      
    }
    LARCV_INFO() << "Number of hits veto'd by keypoints: " << nhits_vetoed << std::endl;
    return nhits_vetoed;
  }
  
}
}
