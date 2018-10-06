#ifndef __LARFLOW_FLASHMATCH__
#define __LARFLOW_FLASHMATCH__

#include <vector>

// larlite
#include "DataFormat/opflash.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"

// larcv
#include "larcv/core/DataFormat/Image2D.h"

namespace larflow {


  class LArFlowFlashMatch {

  public:

    LArFlowFlashMatch();
    virtual ~LArFlowFlashMatch() {};

    struct Results_t {
      /* std::vector<larlite::opflash*> flash_v; */
      /* std::vector<larlite::larflowcluster*> cluster_v; */
      /* std::vector< std::vector<float> > flash2cluster_weights_v; */
      /* std::vector< std::vector<int> > forbidden_matches_v; */
      /* std::vector< std::vector<float> > chi2; */
      float global_chi2;
    };
    
    
    Results_t match( const std::vector<larlite::opflash>& beam_flashes,
		     const std::vector<larlite::opflash>& cosmic_flashes,
		     const std::vector<larlite::larflowcluster>& clusters,
		     const std::vector<larcv::Image2D>& img_v );
    

    // internal data members
    // ---------------------
    struct QPoint_t {
      QPoint_t() {
	xyz[0] = xyz[1] = xyz[2] = 0.0;
	tick = 0;
	pixeladc = 0.0;
	fromplaneid = -1;
      };
      float xyz[3]; // (tick,y,z) coordinates
      float tick;
      float pixeladc;
      int   fromplaneid; // { 0:U, 1:V, 2:Y, 3:UV-ave }
    };

    struct FlashData_t : public std::vector<float> {
      int tpc_tick;
      int tpc_trigx;
      bool isbeam;
      float tot;
    };

    struct QCluster_t : public std::vector<QPoint_t> {
      float min_tyz[3];
      float max_tyz[3];
    };
    
    struct FlashHypo_t : public std::vector<float> {
      int clusteridx;
      int flashidx;
      float tot;
    };

  protected:
    
    // functions to produce spatial charge estimates

    // 1) first naive adc sum on pixel intensities
    void buildInitialQClusters( const std::vector<larlite::larflowcluster>&,
				std::vector<QCluster_t>&,
				const std::vector<larcv::Image2D>&, int src_plane );

    // 2) replace charge from dead regions in y-wires: connect points in 3d near dead region end.
    //    project into u,v. collect hits there.

    // 3) hits/pixels not part of cluster (pixel got flowed to wrong place), but clearly inside neighborhood of pixels
    //    we collect that as charge as well

    // Collection of the opflash data
    // -------------------------------
    std::vector< FlashData_t > collectFlashInfo( const std::vector<larlite::opflash>& beam_flashes,
						 const std::vector<larlite::opflash>& cosmic_flashes );
    
    // fitting strategies
    // 1) simple metropolist hastings using global-chi2 as function
    // 2) gradient descent

    // flash-cluster compatability matrix
    // (asks if a cluster is physically possible to be the source of the flash)
    // -----------------------------------    
    int* m_compatibility;
    int  _nflashes;   // row
    int  _nqclusters; // col
    int  _nelements;  // row*col
    int* _nclusters_compat_wflash; // [nflashes]
    bool _compatibility_defined;
    void buildFullCompatibilityMatrix( const std::vector<FlashData_t>&, const std::vector<QCluster_t>& );
    void resetCompatibiltyMatrix();
    inline int  getCompat( int iflash, int iqcluster )  { return *(m_compatibility + _nqclusters*iflash + iqcluster); };
    inline void setCompat( int iflash, int iqcluster, int compat ) { *(m_compatibility + _nqclusters*iflash + iqcluster) = compat; };

    // Flash Hypothesis Building
    // -------------------------
    std::vector< std::vector<LArFlowFlashMatch::FlashHypo_t> >  buildFlashHypotheses( const std::vector<FlashData_t>& flashdata_v,
										      const std::vector<QCluster_t>& qcluster_v );

    // Build Fit Parameter matrices
    // ----------------------------
    int _nmatches;         // {(flash,cluster) ordered pairs, unrolled
    int _nflashes_red;     // (flash w/ matches)
    int _nclusters_red;    // (clusters w/ matches)
    bool _reindexed;       // have we reindexed the compatible flashes/clusters
    std::vector<int> _match_clustidx;
    std::vector<int> _match_flashidx;
    std::vector<int> _match_flashidx_orig;
    std::map<int,int> _flash_reindex;
    std::map<int,int> _clust_reindex;
    float* m_flash_hypo;   // [nclusters_red][npmts]
    float* m_flash_data;   // [nflashes_red][npmts]
    float* m_flashhypo_norm;
    float* m_flashdata_norm;
    int* _pair2index;      // [flashreindex][cluster-reindex], value is match-index
    void buildFittingData(const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>&  qcluster_v );
    void clearFittingData();

    // Calculate Initial Fit Point
    // ----------------------------
    void calcInitialFitPoint();
			       


  };


}

#endif
