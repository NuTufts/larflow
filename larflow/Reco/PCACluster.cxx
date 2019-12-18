#include "PCACluster.h"

#include "cluster_functions.h"

#include "nlohmann/json.hpp"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "ublarcvapp/dbscan/DBScan.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace larflow {
namespace reco {

  void PCACluster::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {

    larlite::event_larflow3dhit* ev_lfhits = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "larmatch" );

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

    // cluster all hits
    std::vector<larflow::reco::cluster_t> cluster_all_v;
    larflow::reco::cluster_larflow3dhits( *ev_lfhits, cluster_all_v );
    larflow::reco::cluster_runpca( cluster_all_v );
    larflow::reco::cluster_dump2jsonfile( cluster_all_v, "dump_all.json" );    

    // containers for track and shower hits
    std::vector<larlite::larflow3dhit> track_hit_v;
    std::vector<larlite::larflow3dhit> shower_hit_v;

    // divide pixels by track and shower
    larflow::reco::cluster_splitbytrackshower( *ev_lfhits, ssnet_trackimg_v, track_hit_v, shower_hit_v );

    // cluster each hit type, define pca by the pixels
    std::vector<larflow::reco::cluster_t> cluster_track_v;
    larflow::reco::cluster_larflow3dhits( track_hit_v, cluster_track_v );
    larflow::reco::cluster_runpca( cluster_track_v );

    std::vector<larflow::reco::cluster_t> cluster_shower_v;
    larflow::reco::cluster_larflow3dhits( shower_hit_v, cluster_shower_v, 20.0, 5, 5 );
    larflow::reco::cluster_runpca( cluster_shower_v );

    larflow::reco::cluster_dump2jsonfile( cluster_track_v,  "dump_track.json" );
    larflow::reco::cluster_dump2jsonfile( cluster_shower_v, "dump_shower.json" );    

    // we perform split functions on the track clusters
    int nsplit = 0;
    for (int ipass=0; ipass<3; ipass++ ) {
      nsplit = split_clusters( cluster_track_v, adc_v );
      std::cout << "splitting: pass[" << ipass << "] num split=" << nsplit << std::endl;      
      if (nsplit==0 ) break;
    }

    std::cout << "defrag clusters" << std::endl;
    defragment_clusters( cluster_track_v, 10.0 );
    
    larflow::reco::cluster_dump2jsonfile( cluster_track_v, "dump_split.json" );

    // now we merge
    int nmerged = 0;
    nmerged = merge_clusters( cluster_track_v, adc_v, 10.0, 30.0, 10.0 );
    std::cout << "[merger-0 maxdist=10.0, maxangle=30.0, maxpca=10.0] number merged=" << nmerged << std::endl;

    nmerged = merge_clusters( cluster_track_v, adc_v, 20.0, 15.0, 10.0 );
    std::cout << "[merger-1 maxdist=20.0, maxangle=10.0, maxpca=10.0] number merged=" << nmerged << std::endl;    

    nmerged = merge_clusters( cluster_track_v, adc_v, 20.0, 60.0, 15.0, true );
    std::cout << "[merger-2 maxdist=5.0, maxangle=60.0, maxpca=5.0] number merged=" << nmerged << std::endl;
    
    larflow::reco::cluster_dump2jsonfile( cluster_track_v, "dump_merged.json" );

    std::vector<cluster_t> final_v;
    for ( auto& c : cluster_track_v )
      final_v.push_back(c);
    for ( auto& c : cluster_shower_v )
      final_v.push_back(c);
      
    larflow::reco::cluster_dump2jsonfile( final_v, "dump_final.json" );

    //if (true)
    //return;
    
    // form clusters of larflow hits for saving
    larlite::event_larflowcluster* evout_lfcluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "pcacluster" );
    //evout_lfcluster->reserve( final_v.size() );    
    for ( auto& cluster : final_v ) {
      larlite::larflowcluster lfcluster = makeLArFlowCluster( cluster, ssnet_showerimg_v, ssnet_trackimg_v );
      evout_lfcluster->emplace_back( std::move(lfcluster) );
    }//end of cluster loop

    // form clusters of larflow hits for saving
    larlite::event_larflowcluster* evout_shower_lfcluster = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfshower" );
    //evout_shower_lfcluster->reserve( cluster_shower_v.size() );
    for ( auto& cluster : cluster_shower_v ) {
      larlite::larflowcluster lfcluster = makeLArFlowCluster( cluster, ssnet_showerimg_v, ssnet_trackimg_v );
      evout_shower_lfcluster->emplace_back( std::move(lfcluster) );
    }//end of cluster loop
    
  }

  int PCACluster::split_clusters( std::vector<cluster_t>& cluster_v,
                                  const std::vector<larcv::Image2D>& adc_v ) {

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
      if ( clust.pca_eigenvalues[1]<10.0 ) {
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
          
          if ( contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAeigenvalue(1)>5.0 ) continue;
          
          // find max-dist from start-end of pca-axis
          float pca_dist = 0;
          float dx = 0;
          for ( size_t i=0; i<2; i++ ) {
            dx = ( contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAEndPos()[i] -
                   contour_algo.m_plane_atomicmeta_v[p][ictr].getPCAStartPos()[i] );
            pca_dist += dx*dx;
          }
          pca_dist = sqrt(pca_dist);
          std::cout << "  icontour[" << ictr << "] "
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
          for ( auto const& hitidx : dbclust ) {
            c.points_v.push_back( cluster.points_v[hitidx] );
            c.imgcoord_v.push_back( cluster.imgcoord_v[hitidx] );
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
                                                          const std::vector<larcv::Image2D>& ssnet_trackimg_v ) {
    
    larlite::larflowcluster lfcluster;
    lfcluster.reserve( cluster.points_v.size() );

    for ( size_t ii=0; ii<cluster.ordered_idx_v.size(); ii++ ) {

      int ihit = cluster.ordered_idx_v[ii];
        
      larlite::larflow3dhit lfhit;
      lfhit.resize(3,0);

      // transfer 3D points
      for (int i=0; i<3; i++) lfhit[i] = cluster.points_v[ihit][i];

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
  
  
}
}
