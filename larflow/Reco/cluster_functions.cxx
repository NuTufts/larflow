#include "cluster_functions.h"

#include <fstream>

#include "nlohmann/json.hpp"
#include "ublarcvapp/dbscan/DBScan.h"
#include <cilantro/principal_component_analysis.hpp>

namespace larflow {
namespace reco {

  /**
   * we use db scan to make an initial set of clusters
   *
   */
  void cluster_larflow3dhits( const std::vector<larlite::larflow3dhit>& hit_v,
                              std::vector< cluster_t >& cluster_v ) {

    // convert points into list of floats
    std::vector< std::vector<float> > points_v;
    points_v.reserve( hit_v.size() );
    
    for ( auto const& lfhit : hit_v ) {
      points_v.push_back( lfhit );
    }
    
    std::vector< ublarcvapp::dbscan::dbCluster > dbcluster_v = ublarcvapp::dbscan::DBScan::makeCluster3f( 5.0, 5, 5, points_v );
    std::cout << "Made clusters: " << dbcluster_v.size() << std::endl;

    for ( auto const& cluster : dbcluster_v ) {
      cluster_t c;
      c.points_v.reserve(cluster.size());
      for ( auto const& hitidx : cluster )
        c.points_v.push_back( points_v.at(hitidx) );
      cluster_v.emplace_back(std::move(c));
    }
  }

  void cluster_pca( cluster_t& cluster ) {

    std::cout << "[cluster_pca]" << std::endl;
    
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
    
  }

  void cluster_runpca( std::vector<cluster_t>& cluster_v ) {
    for ( auto& cluster : cluster_v ) {
      cluster_pca( cluster );
    }
  }
  
  void cluster_dump2jsonfile( const std::vector<cluster_t>& cluster_v, std::string outfilename ) {
    nlohmann::json j;
    std::vector< nlohmann::json > jcluster_v;
    j["clusters"] = jcluster_v;

    for ( auto const& cluster : cluster_v ) {
      std::cout << "cluster: nhits=" << cluster.points_v.size() << std::endl;
      nlohmann::json jcluster;
      jcluster["hits"] = cluster.points_v;

      // save start, center, end of pca to make line
      std::vector< std::vector<float> > pca_points(3);
      for (int i=0; i<3; i++ )
        pca_points[i].resize(3,0);

      std::cout << "cluster: s_min=" << cluster.pca_proj_v.front() << " s_max=" << cluster.pca_proj_v.back() << std::endl;
      
      for (int i=0; i<3; i++ ) {
        pca_points[0][i] = cluster.pca_center[i] + (cluster.pca_proj_v.front()-5)*cluster.pca_axis_v[0][i];
        pca_points[1][i] = cluster.pca_center[i];
        pca_points[2][i] = cluster.pca_center[i] + (cluster.pca_proj_v.back()+5)*cluster.pca_axis_v[0][i];
        //pca_points[0][i] = cluster.pca_center[i] - 10*cluster.pca_axis_v[0][i];
        //pca_points[1][i] = cluster.pca_center[i];
        //pca_points[2][i] = cluster.pca_center[i] + 10*cluster.pca_axis_v[0][i];        
      }
      jcluster["pca"] = pca_points;

      j["clusters"].emplace_back( std::move(jcluster) );      
    }

    std::ofstream o(outfilename.c_str());
    j >> o;
    o.close();
  }

  
}
}
