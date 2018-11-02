#ifndef __LARFLOW_FLASHMATCH__
#define __LARFLOW_FLASHMATCH__

#include <vector>
#include <set>

// ROOT

// larlite
#include "DataFormat/opflash.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/mctrack.h"
#include "DataFormat/pcaxis.h"

// larcv
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"

// larflowflashmatch
#include "FlashMatchTypes.h"
#include "FlashMatchCandidate.h"
#include "QClusterCore.h"

class TRandom3;
class TFile;
class TTree;
namespace larutil {
  class SpaceChargeMicroBooNE;
}

namespace larflow {


  class LArFlowFlashMatch {

  public:

    LArFlowFlashMatch();
    virtual ~LArFlowFlashMatch();

    struct Results_t {
      /* std::vector<larlite::opflash*> flash_v; */
      /* std::vector<larlite::larflowcluster*> cluster_v; */
      /* std::vector< std::vector<float> > flash2cluster_weights_v; */
      /* std::vector< std::vector<int> > forbidden_matches_v; */
      /* std::vector< std::vector<float> > chi2; */
      float global_chi2;
    };
    
    // Functions meant for users
    // --------------------------
    
    Results_t match( const larlite::event_opflash& beam_flashes,
		     const larlite::event_opflash& cosmic_flashes,
		     const std::vector<larlite::larflowcluster>& clusters,
		     const std::vector<larcv::Image2D>& img_v,
		     const bool ignorelast=true);

    void loadMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v, bool do_truth_matching=true );
    void loadChStatus( const larcv::EventChStatus* evstatus ) { _evstatus = evstatus; };
    
    std::vector<larlite::larflowcluster> exportMatchedTracks();
    void saveAnaVariables( std::string anafilename="out_larflow_flashmatch_ana.root" );
    void writeAnaFile();
    void clearEvent();
    
  protected:

    // vectors for storing the events data flashes and reconstructed qclusters
    std::vector<FlashData_t> _flashdata_v;
    std::vector<QCluster_t>  _qcluster_v;
    std::vector<QClusterComposite> _qcomposite_v;
    //std::vector<larlite::pcaxis>  _pca_qcluster_v; // pca, useful
    
    // QCluster Tools
    // ----------------------------------------------
    // 1) first naive adc sum on pixel intensities
    void buildInitialQClusters( const std::vector<larlite::larflowcluster>&,
				std::vector<QCluster_t>&,
				const std::vector<larcv::Image2D>&, int src_plane,
				bool ignorelast=true );
    
    // 2) replace charge from dead regions in y-wires: connect points in 3d near dead region end.
    //    project into u,v. collect hits there. (offloaded to qclustercore)
    /* void fillClusterGapsUsingCorePCA( QCluster_t& cluster ); */
    /* void applyGapFill( std::vector<QCluster_t>& qcluster_v ); */
    
    // 3) hits/pixels not part of cluster (pixel got flowed to wrong place), but clearly inside neighborhood of pixels
    //    we collect that as charge as well

    // Collection of the opflash data
    // -------------------------------
    std::vector< FlashData_t > collectFlashInfo( const larlite::event_opflash& beam_flashes,
						 const larlite::event_opflash& cosmic_flashes );
    
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
    void resetCompatibilityMatrix();
    inline int  getCompat( int iflash, int iqcluster )  { return *(m_compatibility + _nqclusters*iflash + iqcluster); };
    inline void setCompat( int iflash, int iqcluster, int compat ) { *(m_compatibility + _nqclusters*iflash + iqcluster) = compat; };
    void printCompatInfo( const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>& qcluster_v );

    // shape-only comparison code
    // --------------------------
    // need to define bins in z-dimension and assign pmt channels to them
    // this is for shape fit
    std::vector< std::vector<int> > _zbinned_pmtchs;
    //float shapeComparison( const FlashHypo_t& hypo, const FlashData_t& data, float data_norm=1.0, float hypo_norm=1.0 );
    float chi2Comparison( const FlashHypo_t& hypo, const FlashData_t& data, float data_norm=1.0, float hypo_norm=1.0 );    
    void dumpMatchImages( const std::vector<FlashData_t>& flashdata_v, bool shapeonly, bool usefmatch );
    void dumpQClusterImages();
    void dumpQCompositeImages();

    // Flash Hypothesis Building
    // -------------------------
    struct flashclusterpair_t {
      int flashidx;
      int clustidx;
      flashclusterpair_t( int fid, int cid )
      : flashidx(fid), clustidx(cid)
      {};
      bool operator==(const flashclusterpair_t &rhs) const {
        return flashidx == rhs.flashidx && clustidx == rhs.clustidx;
      };
      bool operator<(const flashclusterpair_t& rhs) const
      {
	if ( flashidx<rhs.flashidx) return true;
	else if ( flashidx>rhs.flashidx) return false;
	else {
	  if ( clustidx<rhs.clustidx ) return true;
	  else return false;
	}
	return false; // should never get here
      };
    };
    std::map<flashclusterpair_t,int> m_flash_hypo_map;   // using orig index
    std::vector< FlashHypo_t > m_flash_hypo_v; // DEPRECATED
    std::vector< FlashMatchCandidate > m_matchcandidate_hypo_v; // builflashhypotheses builds this
    void  buildFlashHypotheses( const std::vector<FlashData_t>& flashdata_v,
				const std::vector<QCluster_t>& qcluster_v );
    FlashHypo_t& getHypothesisWithOrigIndex( int flashidx, int clustidx );
    bool hasHypothesis( int flashidx, int clustidx );
    int getMatchIndexFromOrigIndices( int flashidx, int clustidx );
    void getFlashClusterIndexFromMatchIndex( int matchidx, int& flashidx, int& clustidx );
    void clearMatchHypotheses();
    
    // Match refinement
    // ----------------------------
    std::vector<int> _flashdata_best_hypo_chi2_idx;
    std::vector<float> _flashdata_best_hypo_chi2;    
    std::vector<int> _flashdata_best_hypo_maxdist_idx;
    std::vector<float> _flashdata_best_hypo_maxdist;    
    std::vector<int> _clustdata_best_hypo_chi2_idx;
    std::vector<int> _clustdata_best_hypo_maxdist_idx;
    float _fMaxDistCut;
    float _fCosmicDiscThreshold;
    float _fweighted_scalefactor_mean;
    float _fweighted_scalefactor_var;
    float _fweighted_scalefactor_sig;
    float _ly_neg_prob;
    void reduceMatchesWithShapeAnalysis( const std::vector<FlashData_t>& flashdata_v,
					 const std::vector<QCluster_t>&  qcluster_v,
					 bool adjust_pe_for_cosmic_disc );
    void clearFirstRefinementVariables();
    
    // Build Fit Parameter matrices
    // ----------------------------
    int _nmatches;         // {(flash,cluster) ordered pairs, unrolled
    int _nflashes_red;     // (flash w/ matches)
    int _nclusters_red;    // (clusters w/ matches)
    bool _reindexed;       // have we reindexed the compatible flashes/clusters
    std::vector<int> _match_flashidx;    
    std::vector<int> _match_clustidx;
    std::vector<int> _match_flashidx_orig;
    std::vector<int> _match_clustidx_orig;    
    std::map<int,int> _flash_reindex; // original -> reindex
    std::map<int,int> _clust_reindex; // original -> reindex
    std::map<flashclusterpair_t,int> m_flash_hypo_remap; // using reduced indexing
    float* m_flash_hypo;   // [nclusters_red][npmts]
    float* m_flash_data;   // [nflashes_red][npmts]
    float* m_flashhypo_norm;
    float* m_flashdata_norm;
    int*   m_iscosmic;
    int* _pair2index;      // [flashreindex][cluster-reindex], value is match-index
    int getMatchIndex( int reflashidx, int reclustidx ) { return *(_pair2index + reflashidx*_nclusters_red + reclustidx); };
    bool doOrigIndexPairHaveMatch( int flashidx_orig, int clustidx_orig );
    void buildFittingData(const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>&  qcluster_v );
    void clearFittingData();

    // Define fit parameters
    // ---------------------
    // possible matches unrolled into 1D vector with _nmatches elements
    float* flightyield;
    float* fmatch;         // [_nmatches]
    float* fmatch_nll;     // [_nmatches]
    float* fmatch_maxdist; // [_nmatches]
    float* fpmtweight;     // [_nflashes_red*32]
    bool _parsdefined;
    void defineFitParameters();
    void clearFitParameters();
    void zeroMatchVector();
    std::vector<float> getMatchScoresForCluster( int icluster );    
					 
    // Set Initial Fit Point
    // ----------------------------
    void setInitialFitPoint(const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>&  qcluster_v );

    // Calc NLL (given state)
    // ----------------------
    float _fclustsum_weight;
    float _fflashsum_weight;    
    float _fl1norm_weight;
    float _flightyield_weight;
    float calcNLL(bool print=false);

    // Proposal Generation
    // -------------------
    TRandom3* _rand;
    float generateProposal( const float hamdist_mean, const float lydist_mean, const float lydist_sigma,
			    std::vector<float>& match_v, float& ly  );

    // MCTrack Info
    // ------------
    const std::vector<larlite::mctrack>* _mctrack_v;
    std::map<int,int> _mctrackid2index;
    std::set<int>     _nu_mctrackid;
    std::vector<int> _flash_truthid;
    std::vector<int> _cluster_truthid;
    std::vector<int> _flash2truecluster;
    std::vector<int> _cluster2trueflash;
    larutil::SpaceChargeMicroBooNE* _psce;
    bool kDoTruthMatching;
    bool kFlashMatchedDone;
    void doFlash2MCTrackMatching( std::vector<FlashData_t>& flashdata_v ); // matches _mctrack_v
    void doTruthCluster2FlashTruthMatching( std::vector<FlashData_t>& flashdata_v, std::vector<QCluster_t>& qcluster_v );
    void buildClusterExtensionsWithMCTrack( bool appendtoclusters, std::vector<QCluster_t>& qcluster_v );
    void clearMCTruthInfo();
    void setFitParsWithTruthMatch();    

    // ChStatus Info
    // ----------------------------------------------
    const larcv::EventChStatus* _evstatus;


    // secondMatchRefinement
    // ---------------------
    void secondMatchRefinement();


    // analysis variable tree
    // ----------------------
    std::string _ana_filename;
    TFile* _fanafile;
    TTree* _anatree;
    int   _redstep;
    int   _truthmatch;
    int   _isneutrino;
    int   _intime;
    int   _isbeam;
    float _hypope;
    float _datape;
    float _maxdist_orig;
    float _peratio_orig;
    float _maxdist_wext;
    float _peratio_wext;
    float _maxdist_red2;
    float _peratio_red2;
    bool  _save_ana_tree;
    bool  _anafile_written;
    void setupAnaTree();
    void clearAnaVariables();

  };
    
    
}

#endif
  
