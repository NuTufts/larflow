#include "cluster_functions.h"

#include <fstream>

//#include "TVector.h"
#include "ublarcvapp/dbscan/DBScan.h"  ///< hand-written
#include "ublarcvapp/dbscan/sDBScan.h" ///< external
#include "ublarcvapp/ContourTools/ContourClusterAlgo.h"
#include <cilantro/principal_component_analysis.hpp>

namespace larflow {
namespace reco {

  ClusterFunctions::ClusterFunctions() {}
  
  /**
   * @brief use DB scan to cluster vector of larflow3dhit
   *
   * @param[in]  hit_v     Vector of larflow3dhit
   * @param[out] cluster_v Container of larflow::reco::cluster_t objects made
   * @param[in]  maxdist   maximum distance two points can be connected
   * @param[in]  minsize   minimum size of cluster
   * @param[in]  maxkd     maximum number of connections a node can have
   * 
   */
  void cluster_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                              std::vector< cluster_t >& cluster_v,
                              const float maxdist, const int minsize, const int maxkd )
  {
    
    // convert points into list of floats
    std::vector< std::vector<float> > points_v;
    points_v.reserve( hit_v.size() );
    
    for ( auto const& lfhit : hit_v ) {
      std::vector<float > hit = { (float)lfhit[0], (float)lfhit[1], (float)lfhit[2] };
      points_v.push_back( hit );
    }


    clock_t start = clock();  
    std::vector< ublarcvapp::dbscan::dbCluster > dbcluster_v = ublarcvapp::dbscan::DBScan::makeCluster3f( maxdist, minsize, maxkd, points_v );

    
    for (int ic=0; ic<(int)dbcluster_v.size()-1;ic++) {
      // skip the last cluster, which are noise points
      auto const& cluster = dbcluster_v[ic];
      cluster_t c;
      c.points_v.reserve(cluster.size());
      c.imgcoord_v.reserve(cluster.size());
      c.hitidx_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster ) {
        // store 3d position and 2D image coordinates
        c.points_v.push_back( points_v.at(hitidx) );
        std::vector<int> coord(4,0);
        coord[3] = hit_v[hitidx].tick;
        coord[0] = (int)hit_v[hitidx].targetwire[0]; // U-plane
        coord[1] = (int)hit_v[hitidx].targetwire[1]; // V-plane
        coord[2] = (int)hit_v[hitidx].srcwire;
        c.imgcoord_v.push_back( coord );
        c.hitidx_v.push_back(hitidx);
      }
      cluster_v.emplace_back(std::move(c));
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    
    //std::cout << "[cluster_larflow3dhit] made clusters: " << dbcluster_v.size() << " elpased=" << elapsed << " secs" << std::endl;    
  }

  /**
   * @brief make clusters from vector of floats using DB scan
   * 
   * @param[in]  points_v  vector of 3D space points represented as a vector<float>
   * @param[out] cluster_v Container of larflow::reco::cluster_t objects made
   * @param[in]  maxdist   maximum distance two points can be connected
   * @param[in]  minsize   minimum size of cluster
   * @param[in]  maxkd     maximum number of connections a node can have
   * 
   */
  void cluster_spacepoint_v( const std::vector< std::vector<float> >& points_v,
                             std::vector< cluster_t >& cluster_v,
                             const float maxdist, const int minsize, const int maxkd )
  {

    clock_t start = clock();  
    std::vector< ublarcvapp::dbscan::dbCluster > dbcluster_v
      = ublarcvapp::dbscan::DBScan::makeCluster3f( maxdist, minsize, maxkd, points_v );

    
    for (int ic=0; ic<(int)dbcluster_v.size()-1;ic++) {
      // skip the last cluster, which are noise points
      auto const& cluster = dbcluster_v[ic];
      cluster_t c;
      c.points_v.reserve(cluster.size());
      //c.imgcoord_v.reserve(cluster.size());
      c.hitidx_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster ) {
        // store 3d position and 2D image coordinates
        c.points_v.push_back( points_v.at(hitidx) );
        c.hitidx_v.push_back(hitidx);
      }
      cluster_v.emplace_back(std::move(c));
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    
    //std::cout << "[cluster_spacepoint] made clusters: " << dbcluster_v.size() << " elpased=" << elapsed << " secs" << std::endl;    
      
  }

  
  /**
   * @brief cluster larflow3dhit with simple DBScan external library
   * 
   * This is preferred to above.
   *
   * @param[in]  hit_v     Vector of larflow3dhit
   * @param[out] cluster_v Container of larflow::reco::cluster_t objects made
   * @param[in]  maxdist   maximum distance two points can be connected
   * @param[in]  minsize   minimum size of cluster
   * @param[in]  maxkd     maximum number of connections a node can have
   *
   */
  void cluster_sdbscan_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                                      std::vector< cluster_t >& cluster_v,
                                      const float maxdist, const int minsize, const int maxkd ) {
    
    clock_t start = clock();
    
    // convert points into list of floats
    std::vector< std::vector<float> > points_v;
    points_v.reserve( hit_v.size() );
    
    for ( auto const& lfhit : hit_v ) {
      std::vector<float> hit = { lfhit[0], lfhit[1], lfhit[2] };
      points_v.push_back( hit );
    }
    
    auto sdbscan = ublarcvapp::dbscan::SDBSCAN< std::vector<float>, float >();
    sdbscan.Run( &points_v, 3, maxdist, minsize );
    
    auto noise = sdbscan.Noise;
    auto dbcluster_v = sdbscan.Clusters;
    
    for (int ic=0; ic<(int)dbcluster_v.size();ic++) {
      // skip the last cluster, which are noise points
      auto const& cluster = dbcluster_v[ic];
      cluster_t c;
      c.points_v.reserve(cluster.size());
      c.imgcoord_v.reserve(cluster.size());
      c.hitidx_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster ) {
        // store 3d position and 2D image coordinates
        c.points_v.push_back( points_v.at(hitidx) );
        std::vector<int> coord(4,0);
        coord[3] = hit_v[hitidx].tick;
        coord[0] = (int)hit_v[hitidx].targetwire[0]; // U-plane
        coord[1] = (int)hit_v[hitidx].targetwire[1]; // V-plane
        coord[2] = (int)hit_v[hitidx].targetwire[2]; // Y-plane
        c.imgcoord_v.push_back( coord );
        c.hitidx_v.push_back(hitidx);
      }
      cluster_v.emplace_back(std::move(c));
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    
    //std::cout << "[cluster_simple_larflow3dhits] made clusters: " << dbcluster_v.size() << " elpased=" << elapsed << " secs" << std::endl;    
  }

  /**
   * @brief cluster spacepoints represented as vector<float> with simple DBScan external library
   * 
   * This is preferred to above.
   *
   * @param[in]  hit_v     Vector of larflow3dhit
   * @param[out] cluster_v Container of larflow::reco::cluster_t objects made
   * @param[in]  maxdist   maximum distance two points can be connected
   * @param[in]  minsize   minimum size of cluster
   * @param[in]  maxkd     maximum number of connections a node can have
   *
   */
  void cluster_sdbscan_spacepoints( const std::vector< std::vector<float> >& hit_v,
                                    std::vector< cluster_t >& cluster_v,
                                    const float maxdist, const int minsize, const int maxkd )
  {
    
    clock_t start = clock();
    
    // convert points into list of floats
    std::vector< std::vector<float> > points_v;
    points_v.reserve( hit_v.size() );
    
    for ( auto const& lfhit : hit_v ) {
      std::vector<float> hit = { lfhit[0], lfhit[1], lfhit[2] };
      points_v.push_back( hit );
    }
    
    auto sdbscan = ublarcvapp::dbscan::SDBSCAN< std::vector<float>, float >();
    sdbscan.Run( &points_v, 3, maxdist, minsize );
    
    auto noise = sdbscan.Noise;
    auto dbcluster_v = sdbscan.Clusters;
    
    for (int ic=0; ic<(int)dbcluster_v.size();ic++) {
      // skip the last cluster, which are noise points
      auto const& cluster = dbcluster_v[ic];
      cluster_t c;
      c.points_v.reserve(cluster.size());
      c.imgcoord_v.reserve(cluster.size());
      c.hitidx_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster ) {
        // store 3d position and 2D image coordinates
        c.points_v.push_back( points_v.at(hitidx) );
        c.hitidx_v.push_back(hitidx);        
      }
      cluster_v.emplace_back(std::move(c));
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    
    //std::cout << "[cluster_sdbscan_spacepoints] made clusters: " << dbcluster_v.size() << " elpased=" << elapsed << " secs" << std::endl;    
  }
  
  /**
   * @brief run PCA on the hits in the cluster
   *
   * points_v must be filled.
   *
   * the following members are then filled:
   * \verbatim embed:rst:leading-asterisk
   *  * larflow::reco::cluster_t::bbox_v
   *  * larflow::reco::cluster_t::pca_center
   *  * larflow::reco::cluster_t::pca_eigenvalues
   *  * larflow::reco::cluster_t::pca_axis_v
   *  * larflow::reco::cluster_t::ordered_idx_v
   *  * larflow::reco::cluster_t::pca_proj_v
   *  * larflow::reco::cluster_t::pca_ends_v
   *  * larflow::reco::cluster_t::pca_len
   *  * larflow::reco::cluster_t::pca_radius_v
   *  * larflow::reco::cluster_t::pca_max_r
   *  * larflow::reco::cluster_t::pca_ave_r2
   * \endverbatim
   * 
   * @param[out] cluster   Container of larflow::reco::cluster_t objects made
   * 
   */  
  void cluster_pca( cluster_t& cluster ) {

    //std::cout << "[cluster_pca]" << std::endl;
    cluster.bbox_v.resize(3);
    for (size_t i=0; i<3; i++ ) {
      cluster.bbox_v[i].resize(2,0);
      cluster.bbox_v[i][0] = 1.0e9;  // initial min value
      cluster.bbox_v[i][1] = -1.0e9; // initial max value
    }
    
    std::vector< Eigen::Vector3f > eigen_v;
    eigen_v.reserve( cluster.points_v.size() );
    for ( auto const& hit : cluster.points_v ) {
      // store values for PCA 
      eigen_v.push_back( Eigen::Vector3f( hit[0], hit[1], hit[2] ) );
      // also determine axis-aligned bounding box values
      for (int i=0; i<3; i++ ) {
        if ( hit[i]<cluster.bbox_v[i][0] ) cluster.bbox_v[i][0] = hit[i];
        if ( hit[i]>cluster.bbox_v[i][1] ) cluster.bbox_v[i][1] = hit[i];
      }
    }

    if ( eigen_v.size()==0 ) {
      throw std::runtime_error("cluster_functions.cc:L271 not enough points to take PCA");
    }
    
    cilantro::PrincipalComponentAnalysis3f pca( eigen_v );
    cluster.pca_center.resize(3,0);
    cluster.pca_eigenvalues.resize(3,0);
    cluster.pca_axis_v.clear();
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

    // normalize first axis (should be though)
    for ( int ipca=0; ipca<3; ipca++) {
      float lenpc = 0.;
      for (int i=0; i<3; i++ ) {
        lenpc += cluster.pca_axis_v[ipca][i]*cluster.pca_axis_v[ipca][i];
      }
      if ( lenpc>0 ) {
        lenpc = sqrt(lenpc);
        for ( int i=0; i<3; i++ )
          cluster.pca_axis_v[ipca][i] /= lenpc;
      }
    }
    
    std::vector< pca_coord > ordered_v;
    ordered_v.reserve( cluster.points_v.size() );
    for ( size_t idx=0; idx<cluster.points_v.size(); idx++ ) {
      float s = 0.;
      for (int i=0; i<3; i++ ) {
        s += (cluster.points_v[idx][i]-cluster.pca_center[i])*cluster.pca_axis_v[0][i];
      }
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
      //cluster.pca_proj_v[i]    = ordered_v[i].s - ordered_v.front().s; // set first point to zero
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

    float len3 = 0;
    std::vector<float> d3(3);    
    for (int i=0; i<3; i++ ) {
      d3[i] = cluster.pca_ends_v[1][i] - cluster.pca_ends_v[0][i];
      len3 += d3[i]*d3[i];      
    }
    len3 = sqrt(len3);    
    cluster.pca_len = len3;
      
    // define radius of each point to the pca-axis
    float max_r = 0.;
    float ave_r2 = 0.;
    cluster.pca_radius_v.resize( ordered_v.size(), 0 );
    for ( size_t i=0; i<ordered_v.size(); i++ ) {
      // get distance of point from pca-axis
      // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
      std::vector<float>& pt = cluster.points_v[ ordered_v[i].idx ];

      std::vector<float> d1(3);
      std::vector<float> d2(3);

      float len1 = 0.;
      for (int i=0; i<3; i++ ) {
        d1[i] = pt[i] - cluster.pca_ends_v[0][i];
        d2[i] = pt[i] - cluster.pca_ends_v[1][i];
        len1 += d1[i]*d1[i];
      }
      len1 = sqrt(len1);

      if ( len3<1.0e-4 ) {
        cluster.pca_radius_v[i] = len1;
      }
      else {

        // cross-product
        std::vector<float> d1xd2(3);
        d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
        d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
        d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
        float len1x2 = 0.;
        for ( int i=0; i<3; i++ ) {
          len1x2 += d1xd2[i]*d1xd2[i];
        }
        len1x2 = sqrt(len1x2);
        cluster.pca_radius_v[i]  = len1x2/len3; // distance of point from PCA-axis
      }
      ave_r2 += cluster.pca_radius_v[i]*cluster.pca_radius_v[i];
      if ( max_r < cluster.pca_radius_v[i] )
        max_r = cluster.pca_radius_v[i];
      
    }//end of loop over ordered points

    ave_r2 = sqrt( ave_r2/cluster.pca_radius_v.size() );

    cluster.pca_max_r  = max_r;
    cluster.pca_ave_r2 = ave_r2;
    
    
  }

  /**
   * @brief run larflow::cluster_pca on a vector of cluster_t
   *
   * @param[out] cluster_v A vector of cluster_t
   *
   */
  void cluster_runpca( std::vector<cluster_t>& cluster_v ) {
    clock_t start = clock();
    for ( auto& cluster : cluster_v ) {
      cluster_pca( cluster );
    }
    clock_t end = clock();
    double elapsed = double(end-start)/CLOCKS_PER_SEC;
    //std::cout << "[ cluster_runpca ] elapsed=" << elapsed << " secs" << std::endl;
  }

  /**
   * @brief dump cluster_t contents into a json file for debugging
   *
   * @param[in] cluster_v Clusters to dumpout.
   * @param[in] outfilename Path to the file we write to.
   *
   */
  void cluster_dump2jsonfile( const std::vector<cluster_t>& cluster_v, std::string outfilename ) {
    nlohmann::json j;
    std::vector< nlohmann::json > jcluster_v;
    j["clusters"] = jcluster_v;

    for ( auto const& cluster : cluster_v ) {
      //std::cout << "cluster: nhits=" << cluster.points_v.size() << std::endl;
      nlohmann::json jcluster = cluster_json(cluster);
      j["clusters"].emplace_back( std::move(jcluster) );      
    }
    
    std::ofstream o(outfilename.c_str());
    j >> o;
    o.close();
  }

  /**
   * @brief convert cluster into json object
   * 
   * for debugging
   *
   * @param[in] cluster Cluster whose info we will convert into json
   *
   */
  nlohmann::json cluster_json( const cluster_t& cluster ) {

    nlohmann::json jcluster;
    jcluster["hits"] = cluster.points_v;

    // save start, center, end of pca to make line
    std::vector< std::vector<float> > pca_points(3);
    for (int i=0; i<3; i++ )
      pca_points[i].resize(3,0);

    for (int i=0; i<3; i++ ) {
      pca_points[0][i] = cluster.pca_ends_v[0][i];// - 5.0*cluster.pca_axis_v[0][i];
      pca_points[1][i] = cluster.pca_center[i];
      pca_points[2][i] = cluster.pca_ends_v[1][i];// + 5.0*cluster.pca_axis_v[0][i];
    }
    jcluster["pca"] = pca_points;
    return jcluster;
  }
    

  /**
   * @brief Split larflow points by projecting into planes and getting scores
   *
   * @param[in] hit_v               A vector of hits in larflow3dhit form.
   * @param[in] ssnettrack_image_v  Images with the track score for each pixel, one for each plane.
   * @param[out] track_hit_v        larflow3dhits from hit_v which land on majority track-like pixels.
   * @param[out] shower_hit_v       larflow3dhits from hit_v which land on majority shower-like pixels.
   * @param[in]  min_larmatch_score minimum larmatch threshold for hit to be passed into track_hit_v or shower_hit_v.
   *
   */
  void cluster_splitbytrackshower( const std::vector<larlite::larflow3dhit>& hit_v,
                                   const std::vector<larcv::Image2D>& ssnettrack_image_v,
                                   std::vector<larlite::larflow3dhit>& track_hit_v,
                                   std::vector<larlite::larflow3dhit>& shower_hit_v,
                                   float min_larmatch_score ) {

    clock_t begin = clock();
    
    track_hit_v.clear();
    shower_hit_v.clear();
    track_hit_v.reserve(  hit_v.size() );
    shower_hit_v.reserve( hit_v.size() );

    std::vector< const larcv::ImageMeta* > meta_v( ssnettrack_image_v.size(),0);
    for ( size_t p=0; p<ssnettrack_image_v.size(); p++ )
      meta_v[p] = &(ssnettrack_image_v[p].meta());

    int below_threshold = 0.;
    
    for ( auto const & hit : hit_v ) {

      //std::cout << "hit[9]=" << hit[9] << std::endl;
      if ( min_larmatch_score>0 && hit.size()>=10 && hit[9]<min_larmatch_score ) {
        below_threshold++;
        continue;
      }
      
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
              << " below-threshold=" << below_threshold
              << " elasped=" << elapsed << " secs"
              << std::endl;
  }

  /**
   * @brief project the 3D cluster points back into the 2D image2d
   *
   * @param[in]  cluster          Cluster whose hits in cluster_t::points_v we project into images.
   * @param[out] clust2d_images_v Image2D whose pixels are set to 50.0 if cluster hit projects onto it.
   *
   */
  void cluster_imageprojection( const cluster_t& cluster,
                                std::vector<larcv::Image2D>& clust2d_images_v ) {

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
   * @brief project the 3D cluster points back into the 2D image2d
   *
   * unfinished. 
   *
   * @brief clust2d_images_v Wire plane images for which we will find 2D contours.
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

  /**
   * @brief calculate closest distance from cluster pca end points
   *
   * uses larflow::reco::cluster_t::pca_ends_v
   *
   * @param[in] clust_a First cluster.
   * @param[in] clust_b Second cluster.
   * @param[out] endpts End point on clust_a and end point on clust_b are returned
   * @return distance between closest end points in cm
   *
   */
  float cluster_closest_endpt_dist( const cluster_t& clust_a, const cluster_t& clust_b,
                                    std::vector< std::vector<float> >& endpts ) {
    // 2x2 calculation

    float dist[4] = {0};
    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        for (int k=0; k<3; k++) {
          dist[ 2*i+j ] += (clust_a.pca_ends_v[i][k] - clust_b.pca_ends_v[j][k])*(clust_a.pca_ends_v[i][k] - clust_b.pca_ends_v[j][k]);
        }
      }
    }

    float mindist = -1.0;
    int mini = 0;
    int minj = 0;
    for (int k=0; k<4; k++ ) {
      if ( dist[k]<mindist || mindist<0 ) {
        mindist = dist[k];
        mini = k/2;
        minj = k%2;
      }
    }

    endpts.resize(2);
    endpts[0].resize(3,0);
    endpts[1].resize(3,0);
    for (int k=0; k<3; k++ ) {
      endpts[0][k] = clust_a.pca_ends_v[mini][k];
      endpts[1][k] = clust_b.pca_ends_v[minj][k];
    }
    
    if ( mindist>0 )
      mindist = sqrt(mindist);
    
    return mindist;
  }

  /**
   * @brief check if one of the end pts are in the bounding box of the cluster
   *
   * @param[in] clust_a First cluster.
   * @param[in] clust_b Second cluster.
   * @return True if there are bounding box overlap.
   */
  bool cluster_endpt_in_bbox( const cluster_t& clust_a, const cluster_t& clust_b ) {

    bool a_in_b = false;
    bool b_in_a = false;
    for (int iend=0; iend<2; iend++) {
      int dims_inside = 0;
      for (int i=0; i<3; i++ ) {
        if ( clust_a.pca_ends_v[iend][i]>=clust_b.bbox_v[i][0] && clust_a.pca_ends_v[iend][i]<=clust_b.bbox_v[i][1] )
          dims_inside++;
      }
      if ( dims_inside==3 )
        a_in_b = true;
      
      dims_inside = 0;
      for (int i=0; i<3; i++ ) {
        if ( clust_b.pca_ends_v[iend][i]>=clust_a.bbox_v[i][0] && clust_b.pca_ends_v[iend][i]<=clust_a.bbox_v[i][1] )
          dims_inside++;
      }
      if ( dims_inside==3 )
        b_in_a = true;
    }

    return a_in_b || b_in_a;
  }

  /** 
   * @brief calculate cosine between first PCA axis of two clusters
   * @param[in] clust_a First cluster.
   * @param[in] clust_b Second cluster.
   * @return inner product of first PC axis
   */
  float cluster_cospca( const cluster_t& clust_a, const cluster_t& clust_b ) {
    float cospca = 0.;
    float lena = 0.;
    float lenb = 0.;
    for ( int k=0; k<3; k++ ) {
      cospca += clust_a.pca_axis_v[0][k]*clust_b.pca_axis_v[0][k];
    }
    return cospca;
  }

  /**
   * @brief make a new cluster that is a combination of two clusters
   *
   * This creates a new cluster based on the union of two clusters
   * 
   * @param[in] clust_a First cluster.
   * @param[in] clust_b Second cluster.
   * @return Union cluster.
   *
   */
  cluster_t cluster_merge( const cluster_t& clust_a, const cluster_t& clust_b ) {
    // copy first
    cluster_t merge = clust_a;

    merge.points_v.reserve(   merge.points_v.size()+clust_b.points_v.size() );
    merge.imgcoord_v.reserve( merge.imgcoord_v.size()+clust_b.imgcoord_v.size() );
    merge.hitidx_v.reserve( merge.imgcoord_v.size()+clust_b.imgcoord_v.size() );
    for ( size_t i=0; i<clust_b.points_v.size(); i++ ) {
      merge.points_v.push_back(   clust_b.points_v[i] );
      merge.imgcoord_v.push_back( clust_b.imgcoord_v[i] );
      merge.hitidx_v.push_back( clust_b.hitidx_v[i] );
    }

    cluster_pca( merge );
    return merge;
  }

  /**
   * @brief convert PC axis information in cluster into larlite::pcaxis object
   *
   * Note, pcaxis objects typically have only 3 vectors.
   * Here we add an additional 2 vectors (total 5) in the larlite::pcaxis object.
   * These include info on the start and end pca line points: cluster_t.pca_ends_v
   * 
   * @param[in] c     The cluster to convert.
   * @param[in] cidx  An ID number the user is free to set.
   * @return larlite::pcaxis containing cluster PCA info.
   */
  larlite::pcaxis cluster_make_pcaxis( const cluster_t& c, int cidx ) {

    // pca-axis
    larlite::pcaxis::EigenVectors e_v;
    // just std::vector< std::vector<double> >
    // we store axes (3) and then the 1st axis end points. So five vectors.
    for ( auto const& a_v : c.pca_axis_v ) {
      std::vector<double> da_v = { (double)a_v[0], (double)a_v[1], (double) a_v[2] };
        e_v.push_back( da_v );
    }
    // start and end points
    for ( auto const& p_v : c.pca_ends_v ) {
      std::vector<double> dp_v = { (double)p_v[0], (double)p_v[1], (double)p_v[2] };
      e_v.push_back( dp_v );
    }
    double eigenval[3] = { c.pca_eigenvalues[0], c.pca_eigenvalues[1], c.pca_eigenvalues[2] };
    double centroid[3] = { c.pca_center[0], c.pca_center[1], c.pca_center[2] };
    larlite::pcaxis llpca( true, c.points_v.size(), eigenval, e_v, centroid, 0, cidx );

    return llpca;
  }

  /**
   * @brief convert PC axis information in cluster into larlite::pcaxis object
   *
   * Note, pcaxis objects typically have only 3 vectors.
   * Here we add an additional 2 vectors (total 5) in the larlite::pcaxis object.
   * These include info on the start and end pca line points: cluster_t.pca_ends_v
   * 
   * @param[in] c     The cluster to convert.
   * @param[in] cidx  An ID number the user is free to set.
   * @return larlite::pcaxis containing cluster PCA info.
   */
  larlite::pcaxis cluster_make_pcaxis_wrt_point( const cluster_t& c,
                                                 const std::vector<float>& refpt,
                                                 int cidx ) {

    // pca-axis
    larlite::pcaxis::EigenVectors e_v;
    // just std::vector< std::vector<double> >
    // we store axes (3) and then the 1st axis end points. So five vectors.
    for ( auto const& a_v : c.pca_axis_v ) {
      std::vector<double> da_v = { (double)a_v[0], (double)a_v[1], (double) a_v[2] };
        e_v.push_back( da_v );
    }
    // start and end points
    int closest_end = larflow::reco::cluster_closest_pcaend( c, refpt );
    int other_end = (closest_end==0) ? 1 : 0;
    std::vector<double> start_v(3,0);
    std::vector<double> end_v(3,0);
    for (int i=0; i<3; i++) {
      start_v[i] = c.pca_ends_v[closest_end][i];
      end_v[i]   = c.pca_ends_v[other_end][i];
    }
    e_v.push_back( start_v );
    e_v.push_back( end_v );

    double eigenval[3] = { c.pca_eigenvalues[0], c.pca_eigenvalues[1], c.pca_eigenvalues[2] };
    double centroid[3] = { c.pca_center[0], c.pca_center[1], c.pca_center[2] };
    larlite::pcaxis llpca( true, c.points_v.size(), eigenval, e_v, centroid, 0, cidx );

    return llpca;
  }
  

  /**
   * @brief convert larflow cluster back into a cluster_t object.
   *
   * we assume that the larflowcluster was made using the tools in this module.
   *
   * @param[in] lfcluster larflow3dhit cluster to convert.
   * @return Cluster_t copy of the input larflow3dhit cluster
   *
   */
  cluster_t cluster_from_larflowcluster( const larlite::larflowcluster& lfcluster ) {

    //std::cout << "[cluster_from_larflowcluster] input cluster size=" << lfcluster.size() << std::endl;
    cluster_t c;
    c.points_v.reserve( 2*lfcluster.size() );
    c.imgcoord_v.reserve( 2*lfcluster.size() );
    c.hitidx_v.reserve( 2*lfcluster.size() );
    for ( auto const& lfhit : lfcluster ) {
      //if ( lfhit.size()<3 ) continue;
      //if ( lfhit.targetwire.size()<3 ) continue;
      // std::cout << "lfhit: " << lfhit[0] << "," << lfhit[1] << "," << lfhit[2] << std::endl;
      // std::cout << "   coord=(" << lfhit.targetwire[0] << "," << lfhit.targetwire[1] << "," << lfhit.targetwire[2] << ")" << std::endl;
      // std::cout << "   tick=" << lfhit.tick << std::endl;
      std::vector<float> pos   = { lfhit[0], lfhit[1], lfhit[2] };
      std::vector<int>   coord = { lfhit.targetwire[0],
                                   lfhit.targetwire[1],
                                   lfhit.targetwire[2],
                                   lfhit.tick };
      c.points_v.push_back( pos );
      c.imgcoord_v.push_back( coord );
      c.hitidx_v.push_back( (int)c.points_v.size()-1 );
    }

    cluster_pca( c );

    return c;
  }

  /*
   * @brief project hits into the ADC image and get the total pixel sum for each plane
   *
   * @param[in] cluster Cluster whose hits we project. (Note we use cluster_t::imgcoord).
   * @param[in] img_v   Wire plane image to get the charge sum from.
   * @return charge sum on each plane.
   *
   */
  std::vector<float> cluster_pixelsum( const cluster_t& cluster,
                                       const std::vector<larcv::Image2D>& img_v  ) {
    
    std::vector<float> cluster_adc(3,0.0);

    std::vector<larcv::Image2D> blank_v;
    for (auto const& img : img_v ) {
      larcv::Image2D blank(img.meta());
      blank.paint(0);
      blank_v.emplace_back( std::move(blank) );
    }

    for ( size_t ihit=0; ihit<cluster.imgcoord_v.size(); ihit++ ) {
      int row = img_v[0].meta().row( cluster.imgcoord_v[ihit][3], __FILE__, __LINE__ );
      for ( size_t p=0; p<img_v.size(); p++ ) {
        int col = img_v[p].meta().col( cluster.imgcoord_v[ihit][p], __FILE__, __LINE__ );
        float used   = blank_v[p].pixel(row,col,__FILE__,__LINE__);
        if ( used<1.0 ) {
          float pixval = img_v[p].pixel(row,col,__FILE__,__LINE__);
          cluster_adc[p] += pixval;
          // mark it to not be reused
          blank_v[p].set_pixel(row,col,10.0);
        }
      }
    }
    return cluster_adc;
  }

  /**
   * @brief get distance of test point from pca-axis
   *
   * @param[in] cluster Cluster to test. Using end points saved in clust.pca_ends_v to perform calculation
   * @param[in] pt test point
   * @return distnce from first PCA line
   */
  float cluster_dist_from_pcaline( const cluster_t& cluster, const std::vector<float>& pt ) {

    // 
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    std::vector<float> d1(3);
    std::vector<float> d2(3);

    float len1 = 0.;
    for (int i=0; i<3; i++ ) {
      d1[i] = pt[i] - cluster.pca_ends_v[0][i];
      d2[i] = pt[i] - cluster.pca_ends_v[1][i];
      len1 += d1[i]*d1[i];
    }
    len1 = sqrt(len1);

    if ( cluster.pca_len<1.0e-4 ) {
      // short cluster, use distance to end point
      return len1;
    }

    // cross-product
    std::vector<float> d1xd2(3);
    d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
    d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
    d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
    float len1x2 = 0.;
    for ( int i=0; i<3; i++ ) {
      len1x2 += d1xd2[i]*d1xd2[i];
    }
    len1x2 = sqrt(len1x2);
    float r = len1x2/cluster.pca_len;
    return r;
  }

  /**
   * @brief is the projection of the point on the 1st pca axis
   *
   * @param[in] cluster Cluster to test. Using end points saved in clust.pca_ends_v to perform calculation
   * @param[in] pt test point
   * @return true if projection on line
   */
  bool cluster_is_point_within_seg( const cluster_t& cluster, const std::vector<float>& pt ) {

    std::vector<float> pca_dir(3,0);
    float len = 0.;
    for (int i=0; i<3; i++) {
      pca_dir[i] = cluster.pca_ends_v[1][i]-cluster.pca_ends_v[0][i];
      len += pca_dir[i]*pca_dir[i];
    }
    if (len==0)
      return false;
    
    for (int i=0; i<3; i++)
      pca_dir[i] /= len;

    float s = larflow::reco::pointRayProjection3f( cluster.pca_ends_v[0], pca_dir, pt );
    if ( s>=0 && s<=len )
      return true;
    return false;
  }
  
  /**
   * @brief append hits of cluster to another
   *
   * This modifies one cluster by copying contents of other cluster into it
   * 
   * @param[inout] merge    cluster that will be appended to
   * @param[in]    clust_b  cluster we copy data from
   */
  void cluster_append( cluster_t& merge, const cluster_t& clust_b ) {

    merge.points_v.reserve(   merge.points_v.size()+clust_b.points_v.size() );
    if ( merge.imgcoord_v.size()+clust_b.imgcoord_v.size()>0 )
      merge.imgcoord_v.reserve( merge.imgcoord_v.size()+clust_b.imgcoord_v.size() );
    if ( merge.imgcoord_v.size()+clust_b.imgcoord_v.size() >0 )
      merge.hitidx_v.reserve( merge.imgcoord_v.size()+clust_b.imgcoord_v.size() );
    
    for ( size_t i=0; i<clust_b.points_v.size(); i++ ) {
      merge.points_v.push_back(   clust_b.points_v[i] );
      if ( clust_b.imgcoord_v.size()>0 )
        merge.imgcoord_v.push_back( clust_b.imgcoord_v[i] );
      if ( clust_b.hitidx_v.size()>0 )
        merge.hitidx_v.push_back( clust_b.hitidx_v[i] );
    }

    cluster_pca( merge );
  }
  
  /**
   * @brief populate cluster bbox info
   *
   * @param[inout] cluster cluster to modify
   *
   */
  void cluster_bbox( cluster_t& cluster )
  {
    //std::cout << "[cluster_pca]" << std::endl;
    cluster.bbox_v.resize(3);
    for (size_t i=0; i<3; i++ ) {
      cluster.bbox_v[i].resize(2,0);
      cluster.bbox_v[i][0] = 1.0e9;  // initial min value
      cluster.bbox_v[i][1] = -1.0e9; // initial max value
    }

    // determine axis-aligned bounding box values    
    for ( auto const& hit : cluster.points_v ) {
      for (int i=0; i<3; i++ ) {
        if ( hit[i]<cluster.bbox_v[i][0] ) cluster.bbox_v[i][0] = hit[i];
        if ( hit[i]>cluster.bbox_v[i][1] ) cluster.bbox_v[i][1] = hit[i];
      }
    }
  }

  /** 
   *@brief distance to bounding box
   *
   */
  float cluster_dist_to_bbox( const cluster_t& cluster, const std::vector<float>& testpt  )
  {
    bool isinside = true;
    float mindist2plane[3] = { 1.0e9, 1.0e9, 1.0e9 };

    for ( int i=0; i<3; i++ ) {
      float lowdist  = testpt[i]-cluster.bbox_v[i][0];
      float highdist = testpt[i]-cluster.bbox_v[i][1];
      if ( lowdist>0 && highdist<0 ) {
        mindist2plane[i] = 0;
      }
      else if ( lowdist<0 ) {
        mindist2plane[i] = fabs(lowdist);
        isinside = false;
      }
      else if ( highdist>0 ) {
        mindist2plane[i] = fabs(highdist);
        isinside = false;        
      }
    }

    if ( isinside )
      return 0.0;
    
    float maxdist = 0;
    for (int i=0; i<3; i++ ) {
      if ( maxdist<mindist2plane[i] )
        maxdist = mindist2plane[i];
    }

    return maxdist;
  }
  
  /**
   * @brief give index of pca end that is closest
   *
   */
  int cluster_closest_pcaend( const cluster_t& cluster, const std::vector<float>& testpt )
  {
    if ( cluster.pca_ends_v.size()!=2 ) {
      throw std::runtime_error( "[larflow::reco::cluster_closest_pcaend] pca_end_v in cluster is not size 2");
    }
    
    float enddist[2] = {0};
    for (int iend=0; iend<2; iend++) {
      for (int v=0; v<3; v++) {
        enddist[iend] += ( cluster.pca_ends_v[iend][v] - testpt[v] )*( cluster.pca_ends_v[iend][v] - testpt[v] );
      }
    }

    if ( enddist[0]<enddist[1] )
      return 0;
    else
      return 1;

    // should never get here
    return 0;
  }

  larlite::track cluster_make_trunk( const cluster_t& cluster, const std::vector<float>& vtxpos )
  {

    float dist[2] = { 0, 0 };
    for (int iend=0; iend<2; iend++) {
      for (int i=0; i<3; i++) {
        dist[iend] += ( cluster.pca_ends_v[iend][i] - vtxpos[i] )*( cluster.pca_ends_v[iend][i] - vtxpos[i] );
      }
    }        
    
    int istart=0;
    int iend = (int)cluster.points_v.size();
    int dii = 1;
    TVector3 all_pca_dir;
    TVector3 all_pca_start;
    TVector3 all_pca_end;    
    if ( dist[1]<dist[0] ) {
      istart = (int)cluster.points_v.size() - 1;
      iend = 0;
      dii = -1;
      float pcalen = 0.;
      for (int v=0; v<3; v++) {
	all_pca_dir[v] = cluster.pca_ends_v[0][v]-cluster.pca_ends_v[1][v];
	pcalen += all_pca_dir[v]*all_pca_dir[v];
	all_pca_start[v] = cluster.pca_ends_v[1][v];
      }
      pcalen = sqrt(pcalen);
      if ( pcalen>0 ) {
	for (int v=0; v<3; v++) {
	  all_pca_dir[v] /= pcalen;
	  all_pca_end[v] = all_pca_start[v] + 10.0*all_pca_dir[v];
	}
      }
    }
    else {
      float pcalen = 0.;
      for (int v=0; v<3; v++) {
	all_pca_dir[v] = cluster.pca_ends_v[1][v]-cluster.pca_ends_v[0][v];
	pcalen += all_pca_dir[v]*all_pca_dir[v];
	all_pca_start[v] = cluster.pca_ends_v[0][v];
      }
      pcalen = sqrt(pcalen);
      if (pcalen>0 ) {
	for (int v=0; v<3; v++) {
	  all_pca_dir[v] /= pcalen;
	  all_pca_end[v] = all_pca_start[v] + 10.0*all_pca_dir[v];	  
	}
      }
    }

    larflow::reco::cluster_t newtrunk_cluster;
    newtrunk_cluster.points_v.reserve(200);
    std::vector< std::vector<float> > newtrunk_hits_v;
    newtrunk_hits_v.reserve( 200 );
    for (int ii=istart; ii!=iend; ii += dii ) {
      int iidx = cluster.ordered_idx_v[ii];
      int ipt  = cluster.hitidx_v[iidx];
      
      if ( fabs(cluster.pca_proj_v[istart]-cluster.pca_proj_v[ii]) < 10.0
	   && cluster.pca_radius_v[ii] < 3.0 ) {
	std::vector<float> pt = cluster.points_v.at(iidx);
	newtrunk_hits_v.push_back(pt);
	newtrunk_cluster.points_v.push_back(pt);
      }
    }//end of loop over pca order
    //LARCV_DEBUG() << newtrunk_cluster.points_v.size() << " hits of " << cluster.points_v.size() << " used to form trunk" << std::endl;

    bool use_all_pca = false;
    if ( cluster.points_v.size()<10 ) {
      use_all_pca = true;
    }
    else {
      try {
	larflow::reco::cluster_pca( newtrunk_cluster );
      }
      catch ( std::exception& e ) {
	//LARCV_DEBUG() << "Error calculating PCA: " << e.what() << std::endl;
	// we make the trunk using the overall pca
	use_all_pca = true;
      }
    }

    if ( newtrunk_cluster.pca_len<3.0 )
      use_all_pca = true;

    larlite::track shower_newtrunk;
    shower_newtrunk.reserve(2);
    
    if ( use_all_pca ) {
      shower_newtrunk.add_vertex( all_pca_start );
      shower_newtrunk.add_vertex( all_pca_end );
      shower_newtrunk.add_direction( all_pca_dir );
      shower_newtrunk.add_direction( all_pca_dir );
      return shower_newtrunk;
    }
    
    float trunkdist[2] = {0,0};
    for (int iend=0; iend<2; iend++) {
      for (int v=0; v<3; v++) {
	float dx = newtrunk_cluster.pca_ends_v[iend][v]-vtxpos[v];
	trunkdist[iend] += dx*dx;
      }
      trunkdist[iend] = sqrt(trunkdist[iend]);
    }
    
    if ( trunkdist[0]<trunkdist[1] ) {
      TVector3 trunkstart;
      TVector3 trunkend;      
      TVector3 trunkdir;
      for (int v=0; v<3; v++) {
	trunkstart[v] = newtrunk_cluster.pca_ends_v[0][v];
	float dx = newtrunk_cluster.pca_ends_v[1][v]-newtrunk_cluster.pca_ends_v[0][v];
	trunkdir[v] = dx/newtrunk_cluster.pca_len;
	trunkend[v] = trunkstart[v] + 10.0*trunkdir[v];
      }
      shower_newtrunk.add_vertex( trunkstart );
      shower_newtrunk.add_vertex( trunkend );
      shower_newtrunk.add_direction( trunkdir );
      shower_newtrunk.add_direction( trunkdir );            
    }
    else {
      TVector3 trunkstart;
      TVector3 trunkend;      
      TVector3 trunkdir;
      for (int v=0; v<3; v++) {
	trunkstart[v] = newtrunk_cluster.pca_ends_v[1][v];
	float dx = newtrunk_cluster.pca_ends_v[0][v]-newtrunk_cluster.pca_ends_v[1][v];
	trunkdir[v] = dx/newtrunk_cluster.pca_len;
	trunkend[v] = trunkstart[v] + 10.0*trunkdir[v];
      }
      shower_newtrunk.add_vertex( trunkstart );
      shower_newtrunk.add_vertex( trunkend );
      shower_newtrunk.add_direction( trunkdir );
      shower_newtrunk.add_direction( trunkdir );            
    }      

    return shower_newtrunk;
    
  }
}
}
