#include "PCACluster.h"

#include "cluster_functions.h"

#include "nlohmann/json.hpp"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace larflow {
namespace reco {

  void PCACluster::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll ) {

    larlite::event_larflow3dhit* ev_lfhits = (larlite::event_larflow3dhit*)ioll.get_data( larlite::data::kLArFlow3DHit, "larmatch" );

    larcv::EventImage2D* ev_ssnet_track_v[3] = {nullptr};
    for ( size_t p=0; p<3; p++ ) {
      char prodname[20];
      sprintf( prodname, "ubspurn_plane%d", (int)p );
      ev_ssnet_track_v[p] = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, prodname );
    }

    larcv::EventImage2D* ev_adc_v = (larcv::EventImage2D*)iolcv.get_data( larcv::kProductImage2D, "wire" );
    const std::vector<larcv::Image2D>& adc_v = ev_adc_v->Image2DArray();

    // collect track images
    std::vector<larcv::Image2D> ssnet_trackimg_v;
    for ( size_t p=0; p<3; p++ )
      ssnet_trackimg_v.push_back(ev_ssnet_track_v[p]->Image2DArray()[1]);

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
    larflow::reco::cluster_larflow3dhits( shower_hit_v, cluster_shower_v );
    larflow::reco::cluster_runpca( cluster_shower_v );

    int nsplit = 0;
    for (int ipass=0; ipass<3; ipass++ ) {
      nsplit = split_clusters( cluster_track_v, adc_v );
      std::cout << "splitting: pass[" << ipass << "] num split=" << nsplit << std::endl;      
      if (nsplit==0 ) break;
    }
    
    larflow::reco::cluster_dump2jsonfile( cluster_track_v, "dump_track.json" );
    
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
    
  
}
}
