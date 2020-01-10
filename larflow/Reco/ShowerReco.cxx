#include "ShowerReco.h"

#include "larcv/core/DataFormat/EventImage2D.h"
#include "PCACluster.h"
#include "nlohmann/json.hpp"

#include <ctime>

namespace larflow {
namespace reco {

  void ShowerReco::process( larcv::IOManager& iolc, larlite::storage_manager& ioll ) {

    // we process shower clusters produced by PCA cluster algo
    // steps:
    // (1) find trunk candidates:
    //      - we contour the shower pixels and look for straight segments
    //      - we gather track clusters that are connected to the shower clusters
    // (2) we build a shower hypothesis from the trunks:
    //      - we add points along the pca-axis of the cluster
    //      - does one end of the trunk correspond to the end of the assembly? define as start point
    //      - shower envelope expands from start
    //      - trunk pca and assembly pca are aligned
    // (3) choose the best shower hypothesis that has been formed
    
    // first get the ingredients

    larlite::event_larflowcluster* shower_lfcluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "lfshower" );

    larlite::event_larflowcluster* pca_lfcluster_v
      = (larlite::event_larflowcluster*)ioll.get_data( larlite::data::kLArFlowCluster, "pcacluster" );

    larcv::EventImage2D* ev_adc = (larcv::EventImage2D*)iolc.get_data( larcv::kProductImage2D, "wire" );

    std::clock_t begin_process = std::clock();
    std::cout << "[ShowerReco::process] start" << std::endl;
    std::cout << "  num larflow clusters: " << pca_lfcluster_v->size() << std::endl;
    std::cout << "  num shower clusters:  " << shower_lfcluster_v->size() << std::endl;    

    // convert back to cluster objects
    std::clock_t begin_convert = std::clock();
    std::vector< cluster_t > shower_cluster_v;
    for ( auto const& lfcluster : *shower_lfcluster_v ) {
      cluster_t c = cluster_from_larflowcluster( lfcluster );
      shower_cluster_v.emplace_back( std::move(c) );
    }
    std::clock_t end_convert = std::clock();
    std::cout << "  convert lf clusters back to cluster_t: "
              << float(end_convert-begin_convert)/CLOCKS_PER_SEC << " secs" << std::endl;

    // now for each shower cluster, we find some trunk candidates.
    // can have any number of such candidates per shower cluster
    // we only analyze clusters with a first pc-axis length > 1.0 cm
    std::vector< std::vector<cluster_t> > trunk_candidates_vv;
    // pointer to the shower larflow clusters we bother to split
    std::vector< const larlite::larflowcluster* > analyzed_shower_lfcluster_v;
    std::vector< int > analyzed_index_v;
    int idx = -1;
    for ( auto const& showercluster : shower_cluster_v ) {
      idx++;
      if ( showercluster.pca_len<1.0 ) continue;

      std::cout << "[" << idx << "] ShowerReco::process -- clusteridx=[" << idx << "]" << std::endl;
      std::vector<cluster_t> candidates_v = findTrunkCandidates( showercluster, ev_adc->Image2DArray() );
      trunk_candidates_vv.emplace_back( std::move(candidates_v) );
      analyzed_shower_lfcluster_v.push_back( &(shower_lfcluster_v->at(idx)) );
      analyzed_index_v.push_back( idx );
    }

    std::cout << "  number of analyze shower clusters: " << analyzed_shower_lfcluster_v.size() << std::endl;

    std::clock_t end_process = std::clock();
    std::cout << "[ShowerReco::process] end; elapsed = "
              << float(end_process-begin_process)/CLOCKS_PER_SEC << " secs"      
              << std::endl;
  }

  /*
   * search for trunk candidates using atomic contours
   *
   */
  std::vector<cluster_t> ShowerReco::findTrunkCandidates( const cluster_t& showerclust,
                                                          const std::vector<larcv::Image2D>& adc_v ) {
    std::vector<cluster_t> candidates_v;

    std::vector<cluster_t> split_v;
    split_v.push_back( showerclust );

    // we set a second pca length of zero to force contour split
    //larflow::reco::PCACluster::split_clusters( split_v, adc_v, 0.0 );

    std::cout << "[ShowerReco::findTrunkCandidates] " << std::endl;
    std::cout << "  shower len=" << showerclust.pca_len
              << " max_r=" << showerclust.pca_max_r << " ave_r^2=" << showerclust.pca_ave_r2 << std::endl;
    std::cout << "  shower pc eigenvalues: [0]=" << showerclust.pca_eigenvalues[0] << " [1]=" << showerclust.pca_eigenvalues[1] << std::endl;
    std::cout << "  shower cluster split into " << split_v.size() << " pieces." << std::endl;
              
    
    return candidates_v;
  }

  void ShowerReco::dump2json( const std::vector<cluster_t>& shower_v, std::string outfile ) {

  }
  
}
}
