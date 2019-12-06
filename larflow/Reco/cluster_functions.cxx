#include "cluster_functions.h"

#include <fstream>

#include "nlohmann/json.hpp"
#include "ublarcvapp/dbscan/DBScan.h"
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include <cilantro/principal_component_analysis.hpp>

namespace larflow {
namespace reco {

  /**
   * we use db scan to make an initial set of clusters
   *
   */
  void cluster_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                              std::vector< cluster_t >& cluster_v ) {

    clock_t start = clock();
    
    // convert points into list of floats
    std::vector< std::vector<float> > points_v;
    points_v.reserve( hit_v.size() );
    
    for ( auto const& lfhit : hit_v ) {
      points_v.push_back( lfhit );
    }
    
    std::vector< ublarcvapp::dbscan::dbCluster > dbcluster_v = ublarcvapp::dbscan::DBScan::makeCluster3f( 5.0, 5, 5, points_v );

    for ( auto const& cluster : dbcluster_v ) {
      cluster_t c;
      c.points_v.reserve(cluster.size());
      c.imgcoord_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster ) {
        // store 3d position and 2D image coordinates
        c.points_v.push_back( points_v.at(hitidx) );
        std::vector<int> coord(4,0);
        coord[3] = hit_v[hitidx].tick;
        coord[0] = (int)hit_v[hitidx].targetwire[0]; // U-plane
        coord[1] = (int)hit_v[hitidx].targetwire[1]; // V-plane
        coord[2] = (int)hit_v[hitidx].srcwire;
        c.imgcoord_v.push_back( coord );
      }
      cluster_v.emplace_back(std::move(c));
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    
    std::cout << "[cluster_larflow3dhits] made clusters: " << dbcluster_v.size() << " elpased=" << elapsed << " secs" << std::endl;    
  }

  void cluster_pca( cluster_t& cluster ) {

    //std::cout << "[cluster_pca]" << std::endl;
    
    std::vector< Eigen::Vector3f > eigen_v;
    eigen_v.reserve( cluster.points_v.size() );
    for ( auto const& hit : cluster.points_v ) {
      eigen_v.push_back( Eigen::Vector3f( hit[0], hit[1], hit[2] ) );
    }
    cilantro::PrincipalComponentAnalysis3f pca( eigen_v );
    cluster.pca_center.resize(3,0);
    cluster.pca_eigenvalues.resize(3,0);    
    for (int i=0; i<3; i++) {
      cluster.pca_center[i] = pca.getDataMean()(i);
      cluster.pca_eigenvalues[i] = pca.getEigenValues()(i);
      std::vector<float> e_v = { pca.getEigenVectors()(0,i),
                                 pca.getEigenVectors()(1,i),
                                 pca.getEigenVectors()(2,i) };
      cluster.pca_axis_v.push_back( e_v );
    }
    // we provide a way to order the points
    struct pca_coord {
      int idx;
      float s;
      bool operator<( const pca_coord& rhs ) const {
        if ( s<rhs.s) return true;
        return false;
      };
    };

    std::vector< pca_coord > ordered_v;
    ordered_v.reserve( cluster.points_v.size() );
    for ( size_t idx=0; idx<cluster.points_v.size(); idx++ ) {
      float s = 0.;
      float lenpc = 0.;
      for (int i=0; i<3; i++ ) {
        s += (cluster.points_v[idx][i]-cluster.pca_center[i])*cluster.pca_axis_v[0][i];
        lenpc += cluster.pca_axis_v[0][i]*cluster.pca_axis_v[0][i];
      }
      if ( lenpc>0 ) {
        s /= sqrt(lenpc);
        for ( int i=0; i<3; i++ )
          cluster.pca_axis_v[0][i] /= sqrt(lenpc);
      }
      //std::cout << "idx[" << idx << "] s=" << s << " lenpc=" << sqrt(lenpc) << std::endl; 
      
      pca_coord coord;
      coord.idx = idx;
      coord.s   = s;
      ordered_v.push_back( coord );
    }
    std::sort( ordered_v.begin(), ordered_v.end() );
    cluster.ordered_idx_v.resize( ordered_v.size() );
    cluster.pca_proj_v.resize( ordered_v.size() );
    for ( size_t i=0; i<ordered_v.size(); i++ ) {
      cluster.ordered_idx_v[i] = ordered_v[i].idx;
      cluster.pca_proj_v[i]    = ordered_v[i].s;
    }

    // define ends
    cluster.pca_ends_v.resize(2);
    cluster.pca_ends_v[0].resize(3,0);
    cluster.pca_ends_v[1].resize(3,0);
    for (int i=0; i<3; i++) {
      cluster.pca_ends_v[0][i] = cluster.pca_center[i] + cluster.pca_proj_v.front()*cluster.pca_axis_v[0][i];
      cluster.pca_ends_v[1][i] = cluster.pca_center[i] + cluster.pca_proj_v.back()*cluster.pca_axis_v[0][i];
    }
    
    
  }

  void cluster_runpca( std::vector<cluster_t>& cluster_v ) {
    clock_t start = clock();
    for ( auto& cluster : cluster_v ) {
      cluster_pca( cluster );
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    std::cout << "[ cluster_runpca ] elapsed=" << elapsed << " secs" << std::endl;
  }
  
  void cluster_dump2jsonfile( const std::vector<cluster_t>& cluster_v, std::string outfilename ) {
    nlohmann::json j;
    std::vector< nlohmann::json > jcluster_v;
    j["clusters"] = jcluster_v;

    for ( auto const& cluster : cluster_v ) {
      //std::cout << "cluster: nhits=" << cluster.points_v.size() << std::endl;
      nlohmann::json jcluster;
      jcluster["hits"] = cluster.points_v;

      // save start, center, end of pca to make line
      std::vector< std::vector<float> > pca_points(3);
      for (int i=0; i<3; i++ )
        pca_points[i].resize(3,0);

      //std::cout << "cluster: s_min=" << cluster.pca_proj_v.front() << " s_max=" << cluster.pca_proj_v.back() << std::endl;
      
      for (int i=0; i<3; i++ ) {
        pca_points[0][i] = cluster.pca_ends_v[0][i] - 5.0*cluster.pca_axis_v[0][i];
        pca_points[1][i] = cluster.pca_center[i];
        pca_points[2][i] = cluster.pca_ends_v[1][i] + 5.0*cluster.pca_axis_v[0][i];
      }
      jcluster["pca"] = pca_points;

      j["clusters"].emplace_back( std::move(jcluster) );      
    }

    std::ofstream o(outfilename.c_str());
    j >> o;
    o.close();
  }

  /**
   * we split larflow points by projecting into planes and getting scores
   *
   */
  void cluster_splitbytrackshower( const std::vector<larlite::larflow3dhit>& hit_v,
                                   const std::vector<larcv::Image2D>& ssnettrack_image_v,
                                   std::vector<larlite::larflow3dhit>& track_hit_v,
                                   std::vector<larlite::larflow3dhit>& shower_hit_v ) {

    clock_t begin = clock();
    
    track_hit_v.clear();
    shower_hit_v.clear();
    track_hit_v.reserve(  hit_v.size() );
    shower_hit_v.reserve( hit_v.size() );

    std::vector< const larcv::ImageMeta* > meta_v( ssnettrack_image_v.size(),0);
    for ( size_t p=0; p<ssnettrack_image_v.size(); p++ )
      meta_v[p] = &(ssnettrack_image_v[p].meta());
    
    for ( auto const & hit : hit_v ) {
      std::vector<float> scores(3,0);
      scores[0] = ssnettrack_image_v[0].pixel( meta_v[0]->row( hit.tick, __FILE__, __LINE__ ), hit.targetwire[0], __FILE__, __LINE__ );
      scores[1] = ssnettrack_image_v[1].pixel( meta_v[1]->row( hit.tick, __FILE__, __LINE__ ), hit.targetwire[1], __FILE__, __LINE__ );
      scores[2] = ssnettrack_image_v[2].pixel( meta_v[2]->row( hit.tick, __FILE__, __LINE__ ), hit.srcwire,       __FILE__, __LINE__ );

      // condition ... gather metrics
      int n_w_score = 0;
      float tot_score = 0.;
      float max_score = 0.;
      float min_non_zero = 1.;
      for ( auto s : scores ) {
        if ( s>0 ) n_w_score++;
        tot_score += s;
        if ( max_score<s )
          max_score = s;
        if ( s>1 && s<min_non_zero )
          min_non_zero = 0;
      }

      if ( n_w_score>0 && tot_score/float(n_w_score)>0.5 ) {
        track_hit_v.push_back( hit );
      }
      else
        shower_hit_v.push_back( hit );
    }

    clock_t end = clock();
    double elapsed = double(end-begin)/CLOCKS_PER_SEC;
    
    std::cout << "[cluster_split_by_trackshower]: "
              << "original=" << hit_v.size()
              << " into track=" << track_hit_v.size()
              << " and shower=" << shower_hit_v.size()
              << " elasped=" << elapsed << " secs"
              << std::endl;
  }

  /**
   * project the 3D cluster points back into the 2D image2d
   *
   */
  void cluster_imageprojection( const cluster_t& cluster, std::vector<larcv::Image2D>& clust2d_images_v ) {

    // zero the images
    for ( auto& img : clust2d_images_v )
      img.paint(0.0);
    
    for ( auto const& imgcoord : cluster.imgcoord_v ) {
      for ( auto& img : clust2d_images_v ) {
        img.set_pixel( img.meta().row( imgcoord[3], __FILE__, __LINE__ ),
                       img.meta().col( imgcoord[img.meta().plane()], __FILE__, __LINE__ ),
                       50.0 /* dummy value */ );
      }
    }
    
  }

  /**
   * project the 3D cluster points back into the 2D image2d
   *
   */
  void cluster_getcontours( std::vector<larcv::Image2D>& clust2d_images_v ) {
    ublarcvapp::ContourClusterAlgo contour_algo;
    contour_algo.analyzeImages( clust2d_images_v, 10.0, 2, 5, 10, 10, 2 );

    // contours made. now what.
    for ( size_t p=0; p<3; p++ ) {
      int ncontours = contour_algo.m_plane_atomics_v[p].size();
      std::cout << "[cluster_getcontours] ncontours plane[" << p << "] = " << ncontours << std::endl;
      for (int ictr=0; ictr<ncontours; ictr++ ) {

        if ( !contour_algo.m_plane_atomicmeta_v[p][ictr].hasValidPCA() ) {
          std::cout << "  icontour[" << ictr << "] pca not valid" << std::endl;
          continue;
        }
        
        // dist from start-end of pca-axis
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
      }
    }
    
  }

  
  

  
}
}
