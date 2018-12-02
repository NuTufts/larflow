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
#include "DataFormat/mcshower.h"
#include "DataFormat/pcaxis.h"

// larcv
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"

// larflowflashmatch
#include "FlashMatchTypes.h"
#include "FlashMatchCandidate.h"
#include "QClusterCore.h"
#include "LassoFlashMatch.h"

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
    
    // Functions meant for users
    // --------------------------
    
    void match( const larlite::event_opflash& beam_flashes,
		const larlite::event_opflash& cosmic_flashes,
		const std::vector<larlite::larflowcluster>& clusters,
		const std::vector<larcv::Image2D>& img_v,
		const bool ignorelast=true);
    void setRSE( int run, int subrun, int event );    
    void clearEvent();
    void dumpPrefitImages(bool dump)  { _fDumpPrefit  = dump; };
    void dumpPostfitImages(bool dump) { _fDumpPostfit = dump; };    

    void loadMCTrackInfo(  const std::vector<larlite::mctrack>&  mctrack_v, const std::vector<larlite::mcshower>& mcshower_v, bool do_truth_matching=true );
    void loadChStatus( const larcv::EventChStatus* evstatus ) { _has_chstatus=true; _evstatus = evstatus; };

    void saveAnaMatchData(); // call per event
    void saveAnaVariables( std::string anafilename="out_larflow_flashmatch_ana.root" ); // call at begining
    void writeAnaFile(); // call at end


    std::vector<larlite::larflowcluster> exportMatchedClusters(); // returns larflowclusters using best match
    std::vector<larlite::larflowcluster> exportIntimeClusters();  // returns larflowclusters selected
    
  protected:

    // define enum to track rejection reason
    typedef enum { kUncut=0, kWrongTime, kFirstShapeCut, kFirstPERatio, kEnterLength, kFirstFit, kFinalFit } CutReason_t;
    
    // vectors for storing the events data flashes and reconstructed qclusters
    std::vector<FlashData_t>       _flashdata_v;
    std::vector<QCluster_t>        _qcluster_v;
    std::vector<QClusterComposite> _qcomposite_v;
    
    // QCluster Tools
    // ----------------------------------------------
    // 1) first naive adc sum on pixel intensities
    bool _clusters_defined;
    void buildInitialQClusters( const std::vector<larlite::larflowcluster>&,
				std::vector<QCluster_t>&,
				const std::vector<larcv::Image2D>&, int src_plane,
				bool ignorelast=true );
    void clearClusterData();
    
    // 2) replace charge from dead regions in y-wires: connect points in 3d near dead region end.
    //    project into u,v. collect hits there. (offloaded to qclustercore)
    /* void fillClusterGapsUsingCorePCA( QCluster_t& cluster ); */
    /* void applyGapFill( std::vector<QCluster_t>& qcluster_v ); */
    
    // 3) hits/pixels not part of cluster (pixel got flowed to wrong place), but clearly inside neighborhood of pixels
    //    we collect that as charge as well

    // Collection of the opflash data
    // -------------------------------
    bool _flashes_defined;
    std::vector< FlashData_t > collectFlashInfo( const larlite::event_opflash& beam_flashes,
						 const larlite::event_opflash& cosmic_flashes );
    void clearFlashData();
    
    // fitting strategies
    // 1) simple metropolist hastings using global-chi2 as function
    // 2) gradient descent

    // Cut Variable for track-flash matches for dev
    // ---------------------------------------------
    struct CutVars_t {
      CutVars_t()
      : cutfailed(kUncut),
	dtick_window(0),
	maxdist_wext(-1),
	maxdist_noext(-1),
	pe_hypo(-1),
	pe_data(-1),
	peratio_wext(-1),
	peratio_noext(-1),
	enterlen(-1),
	fit1fmatch(-1),
	truthfmatch(-1)
      {};
      CutReason_t cutfailed;

      // timing window
      float dtick_window;
      
      // shape comparison
      float maxdist_wext;
      float maxdist_noext;

      // peratio comparison
      float pe_hypo;
      float pe_data;
      float peratio_wext;
      float peratio_noext;

      // entering length
      float enterlen;

      // first fit
      float fit1fmatch;

      // truth
      float truthfmatch;
    };

    // flash-cluster compatability matrix
    // (asks if a cluster is physically possible to be the source of the flash)
    // -----------------------------------    
    int* m_compatibility;
    int  _nflashes;   // row
    int  _nqclusters; // col
    int  _nelements;  // row*col
    bool _compatibility_defined;
    std::vector< CutVars_t > _compat_cutvars;
    void buildFullCompatibilityMatrix( const std::vector<FlashData_t>&, const std::vector<QCluster_t>& );
    void resetCompatibilityMatrix();
    inline int  getCompat( int iflash, int iqcluster )  { return *(m_compatibility + _nqclusters*iflash + iqcluster); };
    inline void setCompat( int iflash, int iqcluster, int compat ) { *(m_compatibility + _nqclusters*iflash + iqcluster) = compat; };
    CutVars_t& getCutVars( int iflash, int iqcluster ) { return _compat_cutvars.at( _nqclusters*iflash + iqcluster ); };
    void printCompatInfo( const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>& qcluster_v );
    void printCompatSummary();
    void clearMatchVariables();

    
    // Timing Based Match rejection
    // ----------------------------
    void reduceUsingTiming();
    
    // shape-only comparison code
    // --------------------------
    // need to define bins in z-dimension and assign pmt channels to them
    // this is for shape fit
    std::vector< std::vector<int> > _zbinned_pmtchs;
    bool _fDumpPrefit;
    bool _fDumpPostfit;
    float chi2Comparison( const FlashHypo_t& hypo, const FlashData_t& data, float data_norm=1.0, float hypo_norm=1.0 );    
    void dumpQCompositeImages( std::string prefix );

    
    // Match refinement
    // ----------------------------
    float _fMaxDistCut;
    float _fPERatioCut;
    float _fMaxEnterExt;
    float _fCosmicDiscThreshold;    
    void reduceMatchesWithShapeAnalysis( const std::vector<FlashData_t>& flashdata_v,
					 const std::vector<QCluster_t>&  qcluster_v );

    // Entering Extension reductino
    // ----------------------------
    void reduceUsingEnteringLength();
    

    // lasso fitter
    // ---------------
    TRandom3* _rand;
    bool _parsdefined;
    LassoFlashMatch _fitter;
    struct MatchPair_t {
      int flashidx;
      int clusteridx;
      MatchPair_t()
      : flashidx(-1),
	clusteridx(-1)
      {};
      MatchPair_t( int fidx, int clidx )
      : flashidx(fidx),
	clusteridx(clidx)
      {};
      bool operator< ( const MatchPair_t& rhs ) const {
	if ( flashidx<rhs.flashidx ) return true;
	else if ( flashidx==rhs.flashidx && clusteridx<rhs.clusteridx ) return true;
	return false;
      };
    };
    std::map< MatchPair_t, int > _pair2matchidx;
    std::map< int, MatchPair_t > _matchidx2pair;
    void prepareFitter();
    void setInitialFlashMatchVector();
    void reduceUsingFitResults();
    void clearFitter();
					 
    // MCTrack Info
    // ------------
    const std::vector<larlite::mctrack>*  _mctrack_v;
    const std::vector<larlite::mcshower>* _mcshower_v;    
    std::map<int,int> _mctrackid2index;
    std::set<int>     _nu_mctrackid;
    std::vector<int> _flash_truthid;
    std::vector<int> _cluster_truthid;
    std::vector<int> _flash2truecluster;
    std::vector<int> _cluster2trueflash;
    std::vector< std::vector<const larlite::mctrack*> >  _flash_matched_mctracks_v;
    std::vector< std::vector<const larlite::mcshower*> > _flash_matched_mcshowers_v;
    larutil::SpaceChargeMicroBooNE* _psce;
    bool _kDoTruthMatching;
    bool _kFlashMatchedDone;
    void doFlash2MCTrackMatching( std::vector<FlashData_t>& flashdata_v ); // matches _mctrack_v
    void doTruthCluster2FlashTruthMatching( std::vector<FlashData_t>& flashdata_v, std::vector<QCluster_t>& qcluster_v );
    void buildClusterExtensionsWithMCTrack( bool appendtoclusters, std::vector<QCluster_t>& qcluster_v );
    void clearMCTruthInfo();
    void setFitParsWithTruthMatch();    

    // ChStatus Info
    // ----------------------------------------------
    bool  _has_chstatus;
    const larcv::EventChStatus* _evstatus;

    // analysis variable tree
    // ----------------------
    std::string _ana_filename;
    TFile* _fanafile;
    TTree* _anatree;   // results per potential flash-cluster match
    TTree* _flashtree; // results per flash
    int   _run;
    int   _subrun;
    int   _event;
    int   _cutfailed;
    int   _flash_isneutrino;
    int   _flash_intime;
    int   _flash_isbeam;
    int   _flash_isvisible;
    int   _flash_crosstpc;
    int   _flash_mcid;
    float _flash_bestfmatch;
    float _flash_truthfmatch;
    int   _flash_bestclustidx;
    float _flash_tick;
    int   _truthmatched; // flash-cluster    
    int   _usedext; // cluster
    float _hypope;
    float _datape;

    // cut vars
    float _dtick_window;
    float _maxdist_best;
    float _maxdist_wext;
    float _maxdist_noext;    
    float _peratio_best;
    float _peratio_wext;
    float _peratio_noext;    
    float _enterlen;
    float _fmatch;
    float _fmatch_truth;
    
    bool  _save_ana_tree;
    bool  _anafile_written;
    void setupAnaTree();
    void clearAnaVariables();

  };
    
    
}

#endif
  
