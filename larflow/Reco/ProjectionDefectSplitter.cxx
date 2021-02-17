#include "ProjectionDefectSplitter.h"

#include "cluster_functions.h"

#include "nlohmann/json.hpp"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "ublarcvapp/dbscan/DBScan.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include "TRandom3.h"

#include "larflow/Reco/geofuncs.h"
#include "larflow/Reco/TrackOTFit.h"

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

    // Fit linesegments to clusters
    larlite::event_track* evout_track
      = (larlite::event_track*)ioll.get_data( larlite::data::kTrack, _output_cluster_tree_name );
    if ( _fit_line_segments_to_clusters ) {
      std::clock_t t_fit_start = std::clock();
      LARCV_INFO() << "fit the clusters"  << std::endl;
      fitLineSegmentsToClusters( cluster_track_v, *ev_lfhits, adc_v, *evout_track );
      std::clock_t t_fit_end = std::clock();
      float fit_elapsed = ( t_fit_end-t_fit_start )/CLOCKS_PER_SEC;
      LARCV_INFO() << "fit end; elapsed=" << fit_elapsed << " secs" << std::endl;      
    }
    else {
      // make tracks using pca instead
      
    }
    

    // FORM OUTPUTS
    // =============
    
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

      bool foundnan = false;
      for (int v=0; v<3; v++) {
        if ( std::isnan(clust.pca_eigenvalues[v]) )
          foundnan = true;
      }
      if ( foundnan ) {
        LARCV_DEBUG() << " PCA is NAN. no pass." << std::endl;
        //out_v.emplace_back( std::move(clust) ); //  remove ?
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
          LARCV_DEBUG() << "  icontour[" << ictr << "] plane=" << p 
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

        LARCV_DEBUG() << "  [ctr=" << c.index << " plane=" << c.plane << " length=" << c.length << " npts=" << contourcluster.points_v.size() << "]" << std::endl;
        
        // check if big enough
        if ( contourcluster.points_v.size()<5 )
          continue;
        
        // get pca of new cluster
        larflow::reco::cluster_pca( contourcluster );

        LARCV_DEBUG() << "     pca-eigenvalue[1]=" << contourcluster.pca_eigenvalues[1] << std::endl;

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

      LARCV_DEBUG() << "after split: hits claimed=" << totclaimed << " of " << claimedpts.size() << std::endl;

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
    LARCV_INFO() << "end; elapsed=" << elapsed << " secs" << std::endl;


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
      if ( hitidx<0 || hitidx>=(int)source_lfhit_v.size() ) {
        LARCV_CRITICAL() << "Could not retrieve hit with index=" << hitidx << ". Number of input hits=" << source_lfhit_v.size() << std::endl;
        throw std::runtime_error("Could not retrieve hit index");
      }
      
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
   * @param[in] downsample_hit_v Collection of hits we used to cluster. This can be a fraction of the input hits (in hit_v).
   *                             Downsampling done in order to prevent DBSCAN from taking too long.
   * @param[in] orig_idx_v Map from downsample_hit_v index to hit_v index. The vector is the same length as
   *                       downsample_hit_v and so the position of the vector corresponds to the index of downsample_hit_v.
   *                       The values at each position correspond to the index of hit_v.
   * @param[in] max_dist2line Maximum distance from a point to the first principle component axis
   *                          of the given cluster.
   * @return A new cluster with additional hits added to the given cluster
   */
  cluster_t ProjectionDefectSplitter::_absorb_nearby_hits( cluster_t& cluster,
                                                           const std::vector<larlite::larflow3dhit>& hit_v,
                                                           std::vector<int>& used_hits_v,
                                                           std::vector<larlite::larflow3dhit>& downsample_hit_v,
                                                           std::vector<int>& orig_idx_v,
                                                           float max_dist2line ) {

    cluster_t newcluster;
    int nused = 0;
    std::vector<int> absorbed_orig_index_v;
    
    for ( size_t ihit=0; ihit<hit_v.size(); ihit++ ) {

      auto const& hit = hit_v[ihit];

      if ( used_hits_v[ ihit ]!=0 ) continue;
      
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

        int downsample_index = downsample_hit_v.size();
        downsample_hit_v.push_back( hit );
        orig_idx_v.push_back( ihit );
        
        newcluster.points_v.push_back( pt );
        newcluster.imgcoord_v.push_back( coord_v );
        //newcluster.hitidx_v.push_back( ihit );
        newcluster.hitidx_v.push_back( downsample_index ); // clusters refer to the downsample_hit_v index
        used_hits_v[ihit] = 3;
        absorbed_orig_index_v.push_back( ihit );
        nused++;
      }
      
    }
    
    if (nused>0 ) {
      //std::cout << "[absorb_nearby_hits] cluster absorbed " << nused << " hits" << std::endl;
      for ( size_t iadd=0; iadd<newcluster.points_v.size(); iadd++ ) {
        cluster.points_v.push_back( newcluster.points_v[iadd] );
        cluster.imgcoord_v.push_back( newcluster.imgcoord_v[iadd] );
        cluster.hitidx_v.push_back( newcluster.hitidx_v[iadd] );
      }
      cluster_pca( cluster );
    }
    else {
      // throw them back
      //std::cout << "[absorb_nearby_hits] cluster hits " << nused << " below threshold" << std::endl;            
      for ( auto& idx : absorbed_orig_index_v )
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
    //sample = false; /// for debugging

    LARCV_INFO() << "Downsample points: " << sample << ", downsample_fraction=" << downsample_fraction << std::endl;
    
    int nremaining = 0;
    std::vector<int> orig_idx_v;
    orig_idx_v.reserve( total_pts );
    for ( int ihit=0; ihit<total_pts; ihit++ ) {

      if ( used_hits_v[ihit]==1 )
        continue; // assigned to cluster

      nremaining++;

      if ( used_hits_v[ihit]==2 ) {
        // keypoint veto
        continue;
      }
      
      if ( !sample || rand.Uniform()<downsample_fraction ) {
        downsample_hit_v.push_back( inputhits[ihit] );
        orig_idx_v.push_back( ihit ); ///< map from downsample index to original index
      }
      
    }
    LARCV_INFO() << "Remaining hits, " << nremaining << ", downsampled to " << downsample_hit_v.size() << " of " << total_pts << " total" << std::endl;

    // cluster the hits in the downsample_hit_v vector
    std::vector<larflow::reco::cluster_t> cluster_pass_v;
    larflow::reco::cluster_sdbscan_larflow3dhits( downsample_hit_v, cluster_pass_v, _maxdist, _minsize, _maxkd ); // external implementation, seems best
    larflow::reco::cluster_runpca( cluster_pass_v ); // get pca for each cluster

    // now we want to absorb unsampled hits into the clusters we just made
    int nused_final = 0;
    std::vector<larflow::reco::cluster_t> dense_cluster_v;    
    if ( sample ) {

      LARCV_INFO() << "Absorb unused hits" << std::endl;
      
      // we then absorb the hits around these clusters
      for ( auto& ds_cluster : cluster_pass_v ) {
        cluster_t dense_cluster = _absorb_nearby_hits( ds_cluster,
                                                       inputhits,
                                                       used_hits_v,
                                                       downsample_hit_v,
                                                       orig_idx_v,
                                                       10.0 );
        dense_cluster_v.emplace_back( std::move(ds_cluster) );
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

              int downsample_index = orig_idx_v.size();
              orig_idx_v.push_back( ihit );
              downsample_hit_v.push_back( hit );              
              
              cluster.points_v.push_back( pt );
              cluster.imgcoord_v.push_back( imgcoord );
              cluster.hitidx_v.push_back( downsample_index );

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

      // translate indexing back to original clusters
      for ( size_t ii=0; ii<dense.hitidx_v.size(); ii++ ) {
        int downsample_index = dense.hitidx_v[ii];

        if ( downsample_index<0 || downsample_index>=(int)downsample_hit_v.size() ) {
          std::stringstream ss;
          ss << __FILE__ << ":L" << __LINE__ << " Bad Index=" << downsample_index << ". Number of downsample hits=" << downsample_hit_v.size() << std::endl;
          throw std::runtime_error(ss.str());
        }
        
        int orig_idx = orig_idx_v[downsample_index];
        if ( orig_idx<0 || orig_idx>=total_pts ) {
          std::stringstream ss;
          ss << __FILE__ << ":L" << __LINE__ << " Bad Index=" << orig_idx << std::endl;
          throw std::runtime_error(ss.str());
        }
        dense.hitidx_v[ii] = orig_idx;
      }
      
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
   * @param[in] name Name of tree containing keypoints for vetoing hits
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

  /**
   * @brief fit line segments to clusters
   *
   * @param[in] cluster_v Container of clusters to fit
   * @param[in] lfhit_v Source of hits that were used to make the clusters
   * @param[in] adc_v   Wire plane images
   * @param[out] evout_track Container of tracks made of the fitted line segments to the clusters. 
   *                         Should be same length as cluster_v.
   */
  void ProjectionDefectSplitter::fitLineSegmentsToClusters( const std::vector<larflow::reco::cluster_t>& cluster_v,
                                                             const larlite::event_larflow3dhit& lfhit_v,
                                                             const std::vector<larcv::Image2D>& adc_v,
                                                             larlite::event_track& evout_track )
  {
    for (int icluster=0; icluster<(int)cluster_v.size(); icluster++) {

      try {
        // fit segment
        larlite::track lltrack = fitLineSegmentToCluster( cluster_v[icluster], lfhit_v, adc_v );
        evout_track.emplace_back( std::move(lltrack) );              
      }
      catch (...)  {
        // use pca instead :(
        larlite::track lltrack;
        lltrack.add_vertex( TVector3(cluster_v[icluster].pca_ends_v[0][0],
                                     cluster_v[icluster].pca_ends_v[0][1],
                                     cluster_v[icluster].pca_ends_v[0][2]) );
        lltrack.add_vertex( TVector3(cluster_v[icluster].pca_ends_v[1][0],
                                     cluster_v[icluster].pca_ends_v[1][1],
                                     cluster_v[icluster].pca_ends_v[1][2]) );
        lltrack.add_direction( TVector3(0,0,0) );
        lltrack.add_direction( TVector3(0,0,0) );
        evout_track.emplace_back( std::move(lltrack) );        
      }

    }
  }

  /**
   * @brief fit line segments to clusters
   *
   * We break the pca-line into line segments and use larflow::reco::TrackOTFit to fit each piece.
   * 
   * @param[in] cluster Cluster to fit
   * @param[in] lfhit_v Source of his used to make clusters
   * @param[in] adc_v Wire plane images
   * @param[in] max_line_seg_cm Maximum distance a line segment can be when we break up a cluster before fitting.
   * @return Line segments fitted to cluster in the form of a larlite track object
   */
  larlite::track ProjectionDefectSplitter::fitLineSegmentToCluster( const larflow::reco::cluster_t& cluster,
                                                                    const larlite::event_larflow3dhit& lfhit_v,
                                                                    const std::vector<larcv::Image2D>& adc_v,
                                                                    const float max_line_seg_cm )
  {

    float pca_len = cluster.pca_len;
    if ( pca_len<max_line_seg_cm ) {
      // no need to fit
      larlite::track seg;
      seg.reserve(3);
      TVector3 pts[3];
      for (int i=0; i<3; i++) {
        pts[0][i] = cluster.pca_ends_v[0][i];
        pts[2][i] = cluster.pca_ends_v[1][i];
        pts[1][i] = 0.5*(pts[0][i] + pts[2][i]);
      }
      TVector3 segdir = pts[2]-pts[1];
      float dirlen = segdir.Mag();
      if ( dirlen>0 ) {
        for (int i=0; i<3; i++)
          segdir[i] /= dirlen;
      }
      seg.add_vertex( pts[0] );
      seg.add_vertex( pts[1] );
      seg.add_vertex( pts[2] );
      seg.add_direction( segdir );
      seg.add_direction( segdir );
      seg.add_direction( segdir );      
      return seg;
    }

    int nsegments = pca_len/max_line_seg_cm+1;
    float init_seg_len = pca_len/float(nsegments);
    
    // divide pca-axis
    std::vector< std::vector<float> > init_segments_v;
    init_segments_v.reserve(nsegments+1);
    
    for (int iseg=0; iseg<=nsegments; iseg++) {
      std::vector< float > segpt(3,0);
      for (int i=0; i<3; i++)
        segpt[i] = cluster.pca_ends_v[0][i] + float(iseg)*init_seg_len*cluster.pca_axis_v[0][i];
      init_segments_v.push_back( segpt );
    }

    int nhits = cluster.points_v.size();

    const int nrows = adc_v.front().meta().rows();
    const int nplanes = adc_v.size();
    
    // get the charge of the point
    std::vector<float> q_v( nhits, 0);
    std::vector<int> qplane_v( nhits, -1);
    
    for (int ihit=0; ihit<nhits; ihit++) {
      std::vector<int> imgcoord = { lfhit_v[ihit].targetwire[0],
                                    lfhit_v[ihit].targetwire[1],
                                    lfhit_v[ihit].targetwire[2],
                                    0 };
      imgcoord[3] = adc_v.front().meta().row( lfhit_v[ihit].tick );
      std::vector<float> qpix( nplanes, 0 );
      std::vector<int>   npix( nplanes, 0 );
      for (int dr=-2; dr<=2; dr++) {
        int r = imgcoord[3]+dr;
        if ( r<0 || r>=nrows )
          continue;      
        for (int p=0; p<(int)nplanes; p++) {
          const larcv::Image2D& img = adc_v[p];
	  if ( imgcoord[p]<0 || imgcoord[p]>=(int)img.meta().cols() )
	    continue;
          qpix[p] += img.pixel( r, imgcoord[p], __FILE__, __LINE__ );
          npix[p]++;
        }
      }

      for (int p=0; p<(int)nplanes;p++) {
        if ( npix[p]>0 )
          qpix[p] /= (float)npix[p];
      }

      // get smallest non-zero value
      // this means this wire has the most orthognal projection
      std::sort( qpix.begin(), qpix.end() );
      
      for (int p=0; p<3; p++) {
        if ( qpix[p]>0 ) {
          q_v[ihit] = qpix[p];
          qplane_v[ihit] = p;          
          break;
        }
      }
      
    }//end of hit loop
    
    // get the larmatch score (stored in weird place i know...)
    std::vector<float> lm_v( nhits, 0 );
    for (int ihit=0; ihit<nhits; ihit++) {
      lm_v[ihit] = lfhit_v[ihit].track_score;
    }
    
    // get projection s relative to the start point
    std::vector<float> proj_s( nhits, 0 );
    for (int ihit=0; ihit<nhits; ihit++) {
      proj_s[ihit] = larflow::reco::pointRayProjection3f( cluster.pca_ends_v[0], cluster.pca_axis_v[0], cluster.points_v[ihit] );
    }

    // for the first segment, we could be way off, so we seed by using the first pca-axis
    larflow::reco::cluster_t seg0_cluster;
    seg0_cluster.points_v.reserve( int( nhits*2.0*init_seg_len/pca_len ) );

    // label the segment each point is assigned to
    std::vector< int > segindex_v( nhits, 0 );
    
    for (int ihit=0; ihit<nhits; ihit++) {

      int segidx = proj_s[ihit]/init_seg_len;
      segindex_v[ihit] = segidx;
      
      if ( proj_s[ihit]>=0 && proj_s[ihit]<init_seg_len ) {
        std::vector< float > pos_and_feat(5);
        for (int i=0; i<3; i++)
          pos_and_feat[i] = cluster.points_v[ihit][i];
        pos_and_feat[3] = q_v[ihit];
        pos_and_feat[4] = lm_v[ihit];
        seg0_cluster.points_v.push_back( pos_and_feat );
      }
    }

    // the final segment points we have fitted to
    std::vector< std::vector<float> > final_segment_v = init_segments_v;

    // learning rate for fit
    const float lr = 1.0e-1;
      
    if ( seg0_cluster.points_v.size()>3 ) {
      larflow::reco::cluster_pca( seg0_cluster );
    
      // we minizer over the first segment twice.
      // first we hold the start fixed and vary the end
      // then we hold the end fixed and vary the start.
      std::vector< std::vector<float> > seg0_endpt_pass1;
      seg0_endpt_pass1.push_back( seg0_cluster.pca_ends_v[0] );
      seg0_endpt_pass1.push_back( seg0_cluster.pca_ends_v[1] );
      try {
        TrackOTFit::fit_segment( seg0_endpt_pass1, seg0_cluster.points_v, 100, lr );
      }
      catch  (std::exception& e) {
        // restore
        seg0_endpt_pass1.clear();
        seg0_endpt_pass1.push_back( seg0_cluster.pca_ends_v[0] );
        seg0_endpt_pass1.push_back( seg0_cluster.pca_ends_v[1] );
      }

      // swap
      std::vector< std::vector<float> > seg0_endpt_pass2;
      seg0_endpt_pass2.push_back( seg0_endpt_pass1[1] );
      seg0_endpt_pass2.push_back( seg0_cluster.pca_ends_v[0] );
      try {
        TrackOTFit::fit_segment( seg0_endpt_pass2, seg0_cluster.points_v, 100, lr );
      }
      catch (std::exception& e) {
        // restore
        seg0_endpt_pass2.clear();
        seg0_endpt_pass2.push_back( seg0_endpt_pass1[1] );
        seg0_endpt_pass2.push_back( seg0_cluster.pca_ends_v[0] );
      }

      final_segment_v[0] = seg0_endpt_pass2[1];
      final_segment_v[1] = seg0_endpt_pass2[0];
    }
    
    for (int iseg=0; iseg<nsegments; iseg++) {
      std::vector< std::vector<float> > seg(2);
      seg[0] = final_segment_v[iseg];
      seg[1] = final_segment_v[iseg+1];
      // get points
      std::vector< std::vector<float> > seg_point_v;
      for (int ihit=0; ihit<nhits; ihit++) {
        if ( segindex_v[ihit]==iseg ) {
          std::vector< float > pos_and_feat(5);
          for (int i=0; i<3; i++)
            pos_and_feat[i] = cluster.points_v[ihit][i];
          pos_and_feat[3] = q_v[ihit];
          pos_and_feat[4] = lm_v[ihit];
          seg_point_v.push_back( pos_and_feat );
        }
      }
      
      std::vector< std::vector<float> > seg_end(2);
      seg_end[0] = final_segment_v[iseg];
      seg_end[1] = final_segment_v[iseg+1];
      try {
        TrackOTFit::fit_segment( seg_end, seg_point_v, 100, lr );
      }
      catch ( ... ) {
        continue;
      }

      final_segment_v[iseg+1] = seg_end[1];
    }

    // done fitting, make larlite track object
    larlite::track trackout;
    std::vector<float> seglen_v(final_segment_v.size(),1.0);
    for (int iseg=0; iseg<(int)final_segment_v.size(); iseg++) {
      auto const& pt = final_segment_v[iseg];
      trackout.add_vertex( TVector3( pt[0], pt[1], pt[2] ) );

      TVector3 segdir(0,0,0);
      float seglen = 0.;
      
      if ( iseg+1<final_segment_v.size() ) {
        auto const& nextpt = final_segment_v[iseg+1];
        for (int i=0; i<3; i++) {
          segdir[i] = nextpt[i]-pt[i];
          seglen += segdir[i]*segdir[i];
        }
      }
      else if (iseg-1>=0) {
        auto const& prevpt = final_segment_v[iseg-1];
        for (int i=0; i<3; i++) {
          segdir[i] = pt[i]-prevpt[i];
          seglen += segdir[i]*segdir[i];
        }
      }
      seglen = sqrt(seglen);
      seglen_v[iseg] = seglen;
      if ( seglen>0 ) {
        for (int i=0; i<3; i++)
          segdir[i] /= seglen;
      }
      trackout.add_direction( segdir );
    }
    //std::cout << "Fitted track with " << trackout.NumberTrajectoryPoints() << " points" << std::endl;

    // define average dqdx per segment
    // use the fact that the segments should be in order
    std::map<int,float> segment_q_m;
    std::map<int,int>   segment_nhit_m;
    
    for (int ihit=0; ihit<nhits; ihit++) {

      if ( lm_v[ihit]<0.5 )
        continue;
      
      // end of a segment (or last hit)
      int segidx = segindex_v[ihit];

      auto it_qseg = segment_q_m.find( segidx );
      if ( it_qseg==segment_q_m.end() ) {
        segment_q_m[segidx] = 0.0;
        segment_nhit_m[segidx] = 0;
        it_qseg = segment_q_m.find(segidx);
      }
      
      auto it_nhit = segment_nhit_m.find( segidx );
      
      // continue a segment
      it_qseg->second += q_v[ihit];
      it_nhit->second += 1;
    }

    std::vector<double> dqdx_v;
    for (int iseg=0; iseg<nsegments; iseg++) {
      float dqdx = 0.;
      if ( segment_q_m.find(iseg)!=segment_q_m.end() ) {
        auto it_qseg = segment_q_m.find(iseg);
        auto it_nhit = segment_nhit_m.find(iseg);
        if ( it_nhit->second>0 ) {
          dqdx = it_qseg->second/float(it_nhit->second);
        }
      }
      dqdx_v.push_back(dqdx);
    }
    dqdx_v.push_back(0);
    for (int p=0; p<3; p++) 
      trackout.add_dqdx( dqdx_v );
    
    //std::cout << "Added " << dqdx_v.size() << " dqdx points to track" << std::endl;
    
    return trackout;
  }  
  
}
}
