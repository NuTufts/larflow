#ifndef __LARFLOW_RECO_CLUSTER__
#define __LARFLOW_RECO_CLUSTER__

#include <vector>

#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "CilantroSpectral.h"
#include "CilantroPCA.h"
#include "DBSCAN.h"

namespace larflow {


  class RecoCluster {

  public:

    RecoCluster() {};
    virtual ~RecoCluster() {};

    void filter_hits(const std::vector<larlite::larflow3dhit>& hits, std::vector<larlite::larflow3dhit>& fhits, int min_nn, float nn_dist, float fraction_kept=1.0);
    void filterLineClusters(std::vector< std::vector<larlite::larflow3dhit> >& flowclusters,
			    std::vector<larlite::pcaxis>& pcainfos,
			    std::vector< std::vector< std::pair<float,larlite::larflow3dhit> > >& sorted_flowclusters,
			    std::vector<int>& isline);
    std::vector< std::vector<larlite::larflow3dhit> > clusterHits( const std::vector<larlite::larflow3dhit>& hits, std::string algo, bool return_unassigned=true );
    void selectClustToConnect(std::vector< std::vector< std::pair<float,larlite::larflow3dhit> > >& sorted_flowclusters,
			      std::vector<larlite::pcaxis>& pcainfos,
			      std::vector<int>& isline,
			      std::vector<std::vector<int> >& parallel_idx,
			      std::vector<std::vector<int> >& stitch_idx,
			      const double parallel_thresh,
			      const double enddist_thresh);
    void ConnectClusters(std::vector<std::vector<larlite::larflow3dhit> >& flowclusters,
			 std::vector<std::vector<int> >& stich_idx,
			 std::vector<std::vector<larlite::larflow3dhit> >& newclusters);

    void EvaluateClusters(std::vector<std::vector<larlite::larflow3dhit> >& flowclusters,
			  std::vector<int>& purity,
			  std::vector<float>& efficiency);
    
    void set_dbscan_param(float maxdist, float minhits, int maxkdn){_dbscan_param.maxdist=maxdist; _dbscan_param.minhits=minhits; _dbscan_param.maxkdneighbors=maxkdn;};
    void set_spectral_param(int NC, int MaxNN, float MaxDist, float Sigma){
      _spectral_param.NC=NC;
      _spectral_param.MaxNN=MaxNN;
      _spectral_param.MaxDist=MaxDist;
      _spectral_param.Sigma=Sigma;};
    
  protected:
    struct Cluster_t {
      std::vector<larlite::larflow3dhit> phits; 
      float aabbox[3][2]; // axis-aligned bounding box (for faster nn tests)
    };

    struct dbscan_param {
    dbscan_param() : maxdist(-1),
	minhits(-1),
	maxkdneighbors(0){};
      float maxdist; //eps
      float minhits;
      int maxkdneighbors;
    };

    struct spectral_param {
    spectral_param() : NC(0),
	MaxNN(0),
	MaxDist(-1),
	Sigma(-1){};
      int NC; //num cluster
      int MaxNN; //num neighbors
      float MaxDist; 
      float Sigma; // for kernel
    };
    
    std::vector<Cluster_t> createClusters( const std::vector<larlite::larflow3dhit>& hits, std::string algo );
    std::vector<Cluster_t> createClustersPy( const std::vector<larlite::larflow3dhit>& hits, std::string algo);
    Cluster_t assignUnmatchedToClusters( const std::vector<larlite::larflow3dhit>& unmatchedhit_v, std::vector<Cluster_t>& cluster_v );

    dbscan_param _dbscan_param;
    spectral_param _spectral_param;
    
  };

}

#endif
