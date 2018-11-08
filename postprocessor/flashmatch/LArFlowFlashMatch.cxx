#include "LArFlowFlashMatch.h"
#include <sstream>

// ROOT
#include "TCanvas.h"
#include "TH1D.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2D.h"
#include "TEllipse.h"
#include "TBox.h"
#include "TGraph.h"
#include "TText.h"
#include "TStyle.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"

// larflow postprocessor
#include "cluster/CoreFilter.h"
#include "cluster/CilantroPCA.h"

// flashmatch code
#include "FlashMatchCandidate.h"

namespace larflow {

  LArFlowFlashMatch::LArFlowFlashMatch()
    : _clusters_defined(false),
      _flashes_defined(false),
      m_compatibility(nullptr),
      _nflashes(0),
      _nqclusters(0),
      _nelements(0),
      _compatibility_defined(false),
      _fMaxDistCut(0),
      _fPERatioCut(0),
      _fMaxEnterExt(0),
      _fCosmicDiscThreshold(0),
      _rand(nullptr),
      _parsdefined(false),
      _mctrack_v(nullptr),
      _psce(nullptr),
      _kDoTruthMatching(false),
      _kFlashMatchedDone(false),
      _fitter(32),
      _fanafile(nullptr),
      _anatree(nullptr),
      _save_ana_tree(false),
      _anafile_written(false)
  {

    // define bins in z-dimension and assign pmt channels to them
    // this is for shape fit

    if ( true ) {
      // 10-bin in z      
       const larutil::Geometry* geo = larutil::Geometry::GetME();
      _zbinned_pmtchs.resize(10);
      for (int ich=0; ich<32; ich++) {
	int opdet = geo->OpDetFromOpChannel(ich);
	double xyz[3];
	geo->GetOpChannelPosition( ich, xyz );
	int bin = xyz[2]/100.0;
	_zbinned_pmtchs[bin].push_back( ich );
      }
    }
    else {
      // use all pmt-channels
      _zbinned_pmtchs.resize(32);
      for (int ich=0; ich<32; ich++) {
	_zbinned_pmtchs[ich].push_back( ich );
      }
    }

    // more default parameters
    _fMaxDistCut = 0.5;
    _fPERatioCut = 2.0;
    _fMaxEnterExt = 30.0;
    _fCosmicDiscThreshold = 10.0;

    // random generators
    _rand = new TRandom3( 4357 );
  }

  LArFlowFlashMatch::~LArFlowFlashMatch() {
    delete _rand;
    clearEvent();
    if ( _fanafile ) {
      if ( !_anafile_written ) writeAnaFile();
      _fanafile->Close();
    }
  }

  void LArFlowFlashMatch::clearEvent() {
    clearFitter();
    resetCompatibilityMatrix();
    clearMCTruthInfo();
    clearClusterData();
    clearFlashData();    
    _evstatus = nullptr;
    _has_chstatus = false;
  }
  
  void LArFlowFlashMatch::match( const larlite::event_opflash& beam_flashes,
				 const larlite::event_opflash& cosmic_flashes,
				 const std::vector<larlite::larflowcluster>& clusters,
				 const std::vector<larcv::Image2D>& img_v,
				 const bool ignorelast ) {

    // first is to build the charge points for each cluster
    _qcluster_v.clear();
    // we ignore last cluster because sometimes we put store unclustered hits in that entry
    if ( ignorelast ) {
      _qcluster_v.resize( clusters.size()-1 );
    }
    else {
      _qcluster_v.resize( clusters.size() );
    }

    // we build up the charge clusters that are easy to grab
    buildInitialQClusters( clusters, _qcluster_v, img_v, 2, ignorelast );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] InitialQClusters and cores are built" << std::endl;
    //std::cin.get();

    // collect the flashes
    _flashdata_v.clear();
    _flashdata_v = collectFlashInfo( beam_flashes, cosmic_flashes );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Flash Data objects created" << std::endl;
    //std::cin.get();
    
    
    // MC matching: for origina flash and cluster
    if ( _kDoTruthMatching && _mctrack_v!=nullptr ) {
      std::cout << "[LArFlowFlashMatch::match][INFO] Doing MCTrack truth-reco matching" << std::endl;
      doFlash2MCTrackMatching( _flashdata_v );
      doTruthCluster2FlashTruthMatching( _flashdata_v, _qcluster_v );
      bool appendtoclusters = true;
      _kFlashMatchedDone = true;
    }
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Flash-Cluster-MCTrack Truth-Matching Performed" << std::endl;    

    // we have to build up charge in dead regions in the Y-plane
    // [TODO]
    
    // also collect charge from pixels within the track
    //  that might have been flowed to the wrong cluster and not apart of the correct cluster.
    //  we do this using a neighbor fill (if # of pixels around pixel belong to given cluster
    //  make a qpoint
    // [TODO]


    std::cout << "Number of data flashes: " << _flashdata_v.size() << std::endl;
    std::cout << "Number of clusters: " << _qcluster_v.size() << std::endl;
    
    // build initial compbatility matrix
    buildFullCompatibilityMatrix( _flashdata_v, _qcluster_v );

    // reduce using time compatibility
    reduceUsingTiming();
    std::cout << "[LArFlowFlashMatch::match][DEBUG] reduce using timing" << std::endl;        
    printCompatSummary();

    // refined compabtibility: incompatible-z
    reduceMatchesWithShapeAnalysis( _flashdata_v, _qcluster_v );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] reduce matches using shape analysis" << std::endl;
    printCompatSummary();

    // refined compabtibility: incompatible-z
    reduceMatchesWithShapeAnalysis( _flashdata_v, _qcluster_v );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] reduce matches using shape analysis" << std::endl;
    printCompatSummary();

    // refined compabtibility: entering length
    reduceUsingEnteringLength();
    std::cout << "[LArFlowFlashMatch::match][DEBUG] reduce matches using entering length" << std::endl;
    printCompatSummary();
    
    std::cout << "[LArFlowFlashMatch::match][DEBUG] PREFIT COMPATIBILITY" << std::endl;
    printCompatInfo( _flashdata_v, _qcluster_v );

    _fitter.setUseBterms(true);
    prepareFitter();
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Fitter Loaded" << std::endl;
    //dumpQCompositeImages( "prefit" );

    if ( _kDoTruthMatching && _kFlashMatchedDone ) {
      setFitParsWithTruthMatch();
    }

    _fitter.setLossFunction( LassoFlashMatch::kNLL );
    //_fitter.setLossFunction( LassoFlashMatch::kMaxDist );    
    _fitter.printState(false);
    _fitter.printClusterGroups();
    _fitter.printFlashBundles( false );

    std::cout << "-------------------------------------------------------------------------------" << std::endl;    
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Set initial match vector using best maxdist" << std::endl;    
    setInitialFlashMatchVector();
    _fitter.printState(false);
    _fitter.printClusterGroups();
    _fitter.printFlashBundles( false );

    std::cout << "About to start fit. [ENTER] to continue" << std::endl;
    std::cin.get();
    
    // setup learning schedule
    // two stages: first use SGD to minimize with noise
    LassoFlashMatch::LearningConfig_t epoch1;
    epoch1.iter_start = 0;
    epoch1.iter_end   = 20000;
    epoch1.lr = 1.0e-5;
    epoch1.use_sgd = true;
    epoch1.matchfrac = 0.5; // introduce noise
    LassoFlashMatch::LearningConfig_t epoch2;
    epoch2.iter_start = 20001;
    epoch2.iter_end   = 30000;
    epoch2.lr = 1.0e-5;
    epoch2.use_sgd = false;
    epoch2.matchfrac = 1.0; // introduce noise
    // last stage, setting to global min
    LassoFlashMatch::LearningConfig_t epoch3;
    epoch3.iter_start = 20001;
    epoch3.iter_end   = 30000;
    epoch3.lr = 1.0e-4;
    epoch3.use_sgd = false;
    epoch3.matchfrac = 1.0; // introduce noise
    _fitter.addLearningScheduleConfig( epoch1 );
    _fitter.addLearningScheduleConfig( epoch2 );
    //_fitter.addLearningScheduleConfig( epoch3 );    
    _fitter.fitSGD( 30000,-1, true, 0.5 );
    
    // set compat from fit
    reduceUsingFitResults();
    std::cout << "[LArFlowFlashMatch::match][DEBUG] POSTFIT COMPATIBILITY" << std::endl;
    printCompatInfo( _flashdata_v, _qcluster_v );
    _fitter.printState(false);      
    _fitter.printClusterGroups();
    _fitter.printFlashBundles( false );
    _fitter.printBterms();    
    
    dumpQCompositeImages( "postfit" );
    assert(false);

    
    return;
  }

  // ==============================================================================
  // CHARGE CLUSTER TOOLS
  // -------------------------------------------------------------------------------

  void LArFlowFlashMatch::buildInitialQClusters( const std::vector<larlite::larflowcluster>& lfclusters, std::vector<QCluster_t>& qclusters,
						 const std::vector<larcv::Image2D>& img_v, const int src_plane, bool ignorelast ) {

    // we take the larflow 3dhit clusters and convert them into QCluster_t objects
    // these are then used to build QClusterComposite objects, which
    //  are able to generate flash-hypotheses in a number of configurations

    if ( _clusters_defined )
      throw std::runtime_error("[LArFlowFlashMatch::buildInitialQClusters][ERROR] Clusters redefined. Must clear first.");
    
    if ( ignorelast ) {
      if ( qclusters.size()!=lfclusters.size()-1 )
	qclusters.resize( lfclusters.size()-1 );
    }
    else {
      if ( qclusters.size()!=lfclusters.size() )
	qclusters.resize( lfclusters.size() );
    }

    // build the qclusters -- associating 3D position and grabbing charge pixel value from 2D image
    const larcv::ImageMeta& src_meta = img_v[src_plane].meta();

    int nclusters = lfclusters.size();
    if ( ignorelast ) nclusters -= 1;
    for ( size_t icluster=0; icluster<nclusters; icluster++ ) {

      const larlite::larflowcluster& lfcluster = lfclusters[icluster];
      
      QCluster_t& qcluster = qclusters[icluster];
      qcluster.idx = icluster;
      qcluster.reserve( lfcluster.size() );
      for ( size_t i=0; i<3; i++) {
	qcluster.min_tyz[i] =  1.0e9;
	qcluster.max_tyz[i] = -1.0e9;
      }

      // store mctrackids
      std::map<int,int> mctrackid_counts;

      std::vector< std::vector<float> > clusterpts;
      clusterpts.reserve( lfcluster.size() );
      for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
	QPoint_t qhit;
	qhit.xyz.resize(3,0);
	for (size_t i=0; i<3; i++) {
	  qhit.xyz[i] = lfcluster[ihit][i];
	}
	qhit.tick = lfcluster[ihit].tick;
	qhit.type = kNonCore; // default
	
	// clean up the hits
	if ( qhit.tick<src_meta.min_y() || qhit.tick>src_meta.max_y() )
	  continue;

	bool isok = true;
	for (int i=0; i<3; i++) {
	  if ( std::isnan(qhit.xyz[i]) )
	    isok = false;
	}
	if ( qhit.xyz[1]<-118.0 || qhit.xyz[1]>118 ) isok = false;
	if ( qhit.xyz[2]<0 || qhit.xyz[2]>1050 ) isok = false;
	if ( std::isnan(qhit.tick) || qhit.tick<0 )
	  isok = false;
	if (!isok )
	  continue;

	if ( qhit.tick > qcluster.max_tyz[0] )
	  qcluster.max_tyz[0] = qhit.tick;
	if ( qhit.tick < qcluster.min_tyz[0] )
	  qcluster.min_tyz[0] = qhit.tick;
	
	for (size_t i=1; i<3; i++) {
	  if ( qhit.xyz[i] > qcluster.max_tyz[i] )
	    qcluster.max_tyz[i] = qhit.xyz[i];
	  if ( qhit.xyz[i] < qcluster.min_tyz[i] )
	    qcluster.min_tyz[i] = qhit.xyz[i];
	}

	int row = img_v[src_plane].meta().row( lfcluster[ihit].tick );
	int col = img_v[src_plane].meta().col( lfcluster[ihit].srcwire );
	qhit.pixeladc    = img_v[src_plane].pixel( row, col );
	qhit.fromplaneid = src_plane;

	// mc track id
	auto it=mctrackid_counts.find( lfcluster[ihit].trackid );
	if ( it==mctrackid_counts.end() ) {
	  mctrackid_counts[ lfcluster[ihit].trackid ] = 0;
	}
	mctrackid_counts[ lfcluster[ihit].trackid ] += 1;

	clusterpts.push_back( qhit.xyz );
	qcluster.emplace_back( std::move(qhit) );
      }//end of hit loop

      // assign mctrackid based on majority
      int maxcounts = 0;
      int maxid = 0;
      for (auto& it : mctrackid_counts ) {
	if ( maxcounts < it.second ) {
	  maxcounts = it.second;
	  maxid = it.first;
	}
      }
      qcluster.mctrackid = maxid;
      std::cout << "qcluster[" << icluster <<  "] mctrackid=" << maxid << std::endl;
      // Define the QClusterCore -- this tries to identify, using DBSCAN, a core cluster
      // Removes small, floating clusters (that come from truth clustering, probably less so for reco-clustering)
      // Also defines the central PCA, and orders the core hits as a function of projection
      //  along the pca line. Useful for later operations
      QClusterComposite qcomp( qcluster );
      _qcomposite_v.emplace_back( std::move(qcomp) );
      
    }//end of cluster loop

    _clusters_defined = true;
  }

  void LArFlowFlashMatch::clearClusterData() {
    _qcomposite_v.clear();
    _qcluster_v.clear();
    _clusters_defined = false;
  }


  // ==============================================================================
  // DATA FLASH
  // -------------------------------------------------------------------------------
  
  std::vector<FlashData_t> LArFlowFlashMatch::collectFlashInfo( const larlite::event_opflash& beam_flashes,
								const larlite::event_opflash& cosmic_flashes ) {

    if ( _flashes_defined )
      throw std::runtime_error("[LArFlowFlashMatch::collectFlashInfo][ERROR] flash data redefined. must clear first.");
    
    const larutil::Geometry* geo       = larutil::Geometry::GetME();
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    //const size_t npmts = geo->NOpDets();
    const size_t npmts = 32;
    const float  usec_per_tick = 0.5; // usec per tick
    const float  tpc_trigger_tick = 3200;
    const float  driftv = larp->DriftVelocity();
    
    std::vector< FlashData_t > flashdata;
    flashdata.reserve( beam_flashes.size()+cosmic_flashes.size() );
    int iflash = 0;
    const larlite::event_opflash* flashtypes[2] = { &beam_flashes, &cosmic_flashes };
    
    for (int n=0; n<2; n++) {
      for ( auto const& flash : *flashtypes[n] ) {

	if ( n==1 && flash.Time()<22.5 ) {
	  // cosmic disc within beam window, skip
	  continue;
	}
	
	FlashData_t newflash;
	
	newflash.resize( npmts, 0.0 );
	newflash.tot = 0.;
	float maxpmtpe = 0.;
	int maxpmtch = 0;
	
	//int choffset = (n==1 && flash.nOpDets()>npmts) ? 200 : 0;
	int choffset = 0;
	
	for (size_t ich=0; ich<npmts; ich++) {
	  float pe = flash.PE( choffset + ich );
	  newflash[ich] = pe;
	  newflash.tot += pe;
	  if ( pe > maxpmtpe ) {
	    maxpmtpe = pe;
	    maxpmtch = ich;
	  }
	}
	newflash.tpc_tick  = tpc_trigger_tick + flash.Time()/0.5;
	newflash.tpc_trigx = flash.Time()*driftv; // x-assuming x=0 occurs when t=trigger
	newflash.maxch     = maxpmtch;
	newflash.idx       = iflash;
	newflash.isbeam    = ( n==0 ) ? true : false;
	if ( n==0 && flash.Time()>1.0 && flash.Time()<3.8 )
	  newflash.intime = true;
	else
	  newflash.intime = false;
	
	Double_t pmtpos[3];
	geo->GetOpChannelPosition( maxpmtch, pmtpos );	
	newflash.maxchposz   = pmtpos[2];
	
	// normalize
	if (newflash.tot>0) {
	  for (size_t ich=0; ich<npmts; ich++)
	    newflash[ich] /= newflash.tot;
	}

	std::cout << "dataflash[" << iflash << "] "
		  << " tick=" << newflash.tpc_tick
		  << " totpe=" << newflash.tot
		  << " dT=" << flash.Time() << " us (intime=" << newflash.intime << ") "
		  << std::endl;
	
	flashdata.emplace_back( std::move(newflash) );	
	iflash++;	
      }//end of flash loop
    }//end of container loop

    _flashes_defined = true;
    
    return flashdata;
  }

  void LArFlowFlashMatch::clearFlashData() {
    _flashdata_v.clear();
    _flashes_defined = false;
  }

  // ==============================================================
  // COMPATIBILITY MATRIX
  // ==============================================================
  
  void LArFlowFlashMatch::buildFullCompatibilityMatrix( const std::vector< FlashData_t >& flash_v,
							const std::vector< QCluster_t>& qcluster_v ) {

    // asserts to remind myself what depends on what
    assert( _flashes_defined && _clusters_defined );

    if ( _compatibility_defined )
      throw std::runtime_error("[LArFlowFlashMatch::buildFullCompatibilityMatrix][ERROR] compatibility matrix redefined. clear first.");
    
    if ( m_compatibility )
      delete [] m_compatibility;

    _nflashes   = flash_v.size();
    _nqclusters = qcluster_v.size();
    _nelements  = _nflashes*_nqclusters;
    m_compatibility = new int[ _nelements ];
    _compat_cutvars.resize( _nelements );
    _compatibility_defined = true;
  }

  void LArFlowFlashMatch::printCompatSummary() {
    int ncompatmatches = 0;
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {
      for (size_t iclust=0; iclust<_qcluster_v.size(); iclust++) {
	if ( getCompat(iflash,iclust)==kUncut ) ncompatmatches++;
      }
    }
    std::cout << "===========================================" << std::endl;
    std::cout << "FLASH-CLUSTER COMPATIBILITY SUMMARY" << std::endl;
    std::cout << "number of compat flash-cluster matches: " << ncompatmatches << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "NUM COMPAT CLUSTERS PER FLASH" << std::endl;
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {
      int nclustmatches = 0;
      for (size_t iclust=0; iclust<_qcluster_v.size(); iclust++)
	if ( getCompat(iflash,iclust)==kUncut ) nclustmatches++;
      std::cout << "[dataflash " << iflash << "] " << nclustmatches << std::endl;
    }
    std::cout << "------------------------------------------" << std::endl;      
  }

  void LArFlowFlashMatch::resetCompatibilityMatrix() {
    _nflashes = 0;
    _nqclusters = 0;
    _nelements = 0;
    _compatibility_defined = false;
    delete [] m_compatibility;    
    m_compatibility = nullptr;
    _compat_cutvars.clear();
  }

  // =================================================================================================
  // MATCH REJECTION METHODS
  // =================================================================================================

  void LArFlowFlashMatch::reduceUsingTiming() {

    // check for required methods run
    assert( _compatibility_defined );
    
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float max_drifttime_ticks = (256.0+20.0)/larp->DriftVelocity()/0.5; // allow some slop
    
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {
      const FlashData_t& flash = _flashdata_v[iflash];
      
      for ( size_t iq=0; iq<_qcluster_v.size(); iq++) {
	const QCluster_t& qcluster = _qcluster_v[iq];
	
	float dtick_min = qcluster.min_tyz[0] - flash.tpc_tick;
	float dtick_max = qcluster.max_tyz[0] - flash.tpc_tick;
	CutVars_t& cutvar = getCutVars(iflash,iq);

	// must happen after (allow for some slop)
	if ( dtick_min < -10 || dtick_max < -10 ) {
	  cutvar.dtick_window = ( dtick_max<0 ) ? dtick_max : dtick_min;
	  cutvar.cutfailed = kWrongTime;
	  setCompat( iflash, iq, kWrongTime ); // too early
	}
	else if ( dtick_min > max_drifttime_ticks ) {
	  cutvar.dtick_window = dtick_min-max_drifttime_ticks;
	  cutvar.cutfailed = kWrongTime;	  
	  setCompat( iflash, iq, kWrongTime ); // too late
	}
	else {
	  cutvar.dtick_window = dtick_min;
	  setCompat( iflash, iq, kUncut ); // ok
	}
      }
    }
  }

  void LArFlowFlashMatch::reduceUsingEnteringLength() {
    // we remove obviously incorrect matches
    // for those with a very linear shape, assume its muon-like
    // ask if entering extensin makes sense
    //   -- are we traversing too much non-dead regions?
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {
      const FlashData_t& flash = _flashdata_v[iflash];
      float xoffset = (flash.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
      
      for ( size_t iq=0; iq<_qcluster_v.size(); iq++) {
	if ( getCompat(iflash,iq)!=kUncut ) continue;

	
	// what are we doing? if we have to use the entering extension to traverse
	// a large part of the detector (not in a dead region)
	// to get to the visible core, its probably not correct.
	// also, we require that the w/ extension portion is needed.

	CutVars_t& cutvar = getCutVars( iflash,iq );

	// we only want to do this kind of analysis if the cut passes because of the
	// the use of extensions

	bool used_maxdist_wext = ( cutvar.maxdist_wext  < cutvar.maxdist_noext );
	bool used_peratio_wext = ( cutvar.maxdist_noext < cutvar.maxdist_noext );

	if ( !used_maxdist_wext && !used_peratio_wext )
	  continue;

	// have to consider the edge of the detector image cutting off muon
	float maxtick = flash.tpc_tick + 256.0/(0.5*larutil::LArProperties::GetME()->DriftVelocity());
	float mintick = flash.tpc_tick;

	// how much in the x-direction, do I expect to have been cut off?
	float adjusted_xmin = 0;
	float adjusted_xmax = 250;
	if ( maxtick> 2400+1008*6 ) {
	  adjusted_xmax = (maxtick-(2400+1008*6))*(0.5*larutil::LArProperties::GetME()->DriftVelocity());
	}
	else if ( mintick<2400 ) {
	  adjusted_xmin = (2400-mintick)*(0.5*larutil::LArProperties::GetME()->DriftVelocity());
	}
	
	// how long is the extension throught the detector?
	const QCluster_t& qenter = _qcomposite_v[ iq ]._entering_qcluster;
	cutvar.enterlen = 0;
	for ( size_t ipt=0; ipt<qenter.size(); ipt++ ) {
	  const QPoint_t& qpt = qenter[ipt];
	  float xyz[3] = { qpt.xyz[0]-xoffset, qpt.xyz[1], qpt.xyz[2] };
	  
	  if ( xyz[0]>adjusted_xmin && xyz[0]<adjusted_xmax && xyz[1]>-117 && xyz[1]<117 && xyz[2]>0 && xyz[2]<1036 )
	    cutvar.enterlen += qpt.pixeladc;
	}
	
	if ( cutvar.enterlen>_fMaxEnterExt ) {
	  cutvar.cutfailed = kEnterLength;
	  setCompat(iflash,iq,kEnterLength);
	}

      }
    }
  }
  
  void LArFlowFlashMatch::reduceMatchesWithShapeAnalysis( const std::vector<FlashData_t>& flashdata_v,
							  const std::vector<QCluster_t>&  qcluster_v ) {
    // FIRST STAGE REDUCTION
    // from this function we reduced number of possible flash-cluster matches
    // this is done by
    // (1) comparing shape (CDF maxdist) and chi2
    // (2) for chi2, if adjust_pe_for_cosmic_disc, we try to account for pe lost due to cosmic disc threshold
        
    for (int iflash=0; iflash<flashdata_v.size(); iflash++) {
      
      const FlashData_t& flashdata = flashdata_v[iflash];
      float xoffset = (flashdata.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
            
      for (int iclust=0; iclust<qcluster_v.size(); iclust++) {

	if ( getCompat( iflash, iclust )!=kUncut )
	  continue;
       
	const QClusterComposite& qcomposite = _qcomposite_v[iclust];
	CutVars_t& cutvar = getCutVars( iflash, iclust );
	
	// build flash hypothesis for qcluster-iflash pair
	FlashCompositeHypo_t comphypo_wext  = qcomposite.generateFlashCompositeHypo( flashdata, true );
	FlashCompositeHypo_t comphypo_noext = qcomposite.generateFlashCompositeHypo( flashdata, false );

	FlashHypo_t hypo_wext = comphypo_wext.makeHypo();
	FlashHypo_t hypo_noext = comphypo_noext.makeHypo();	

	cutvar.maxdist_wext  = FlashMatchCandidate::getMaxDist( flashdata, hypo_wext, false );
	cutvar.maxdist_noext = FlashMatchCandidate::getMaxDist( flashdata, hypo_noext, false );
	
	
	// remove clearly bad matches
	float maxdist = ( cutvar.maxdist_wext<cutvar.maxdist_noext ) ? cutvar.maxdist_wext : cutvar.maxdist_noext;
	if ( maxdist > _fMaxDistCut ) {
	  cutvar.cutfailed = kFirstShapeCut;
	  setCompat(iflash,iclust,kFirstShapeCut); // fails shape match
	}
	
	// also do pe cut, since we have hypotheses
	cutvar.peratio_wext  = (hypo_wext.tot  - flashdata.tot)/flashdata.tot;
	cutvar.peratio_noext = (hypo_noext.tot - flashdata.tot)/flashdata.tot;
	float peratio = ( cutvar.peratio_wext < cutvar.peratio_noext ) ? cutvar.peratio_wext : cutvar.peratio_noext;

	if ( fabs(peratio) > _fPERatioCut ) {
	  cutvar.cutfailed = kFirstPERatio;
	  setCompat(iflash,iclust,kFirstPERatio);
	}
	
      }//end of cluster loop

    }//end of flash loop
    
  }

  void LArFlowFlashMatch::reduceUsingFitResults() {
    for ( int imatch=0; imatch<_fitter._nmatches; imatch++ ) {
      int iflash = _matchidx2pair[imatch].flashidx;
      int iclust = _matchidx2pair[imatch].clusteridx;

      CutVars_t& cutvars = getCutVars( iflash, iclust );

      float fmatch = _fitter._fmatch_v[imatch];
      if ( fmatch<0.2 ) {
	cutvars.cutfailed = kFirstFit;
	cutvars.fit1fmatch = fmatch;
	setCompat( iflash, iclust, kFirstFit );
      }
    }
  }

  // ======================================================================
  // PRINTING
  // --------

  void LArFlowFlashMatch::printCompatInfo( const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>& qcluster_v ) {
    std::cout << "----------------------" << std::endl;
    std::cout << "COMPAT MATRIX" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << "{}=bestchi2  ()=bestmaxdist" << std::endl;
    int totcompat = 0;
    for (int iflash=0; iflash<_nflashes; iflash++) {
      int ncompat = 0;
      std::vector<int> compatidx;
      for (int iclust=0; iclust<_nqclusters; iclust++) {
	if ( getCompat(iflash,iclust)==0 ) {
	  compatidx.push_back( iclust );
	  ncompat ++;
	}
      }
      std::cout << "flash[" << iflash << "] [Tot: " << ncompat << "] ";
      if ( flashdata_v[iflash].mctrackid>=0 )
	std::cout << "[truthclust=" << flashdata_v[iflash].truthmatched_clusteridx << "] ";
      for ( auto& idx : compatidx ) {
	bool bestmaxdist = false;
	bool bestchi2 = false;

	std::cout << " ";
	if ( bestchi2 )
	  std::cout << "{";
	if ( bestmaxdist )
	  std::cout << "(";
	std::cout << idx;
	if ( bestmaxdist )
	  std::cout << ")";
	if ( bestchi2 )
	  std::cout << "}";	
      }
      std::cout << std::endl;
      totcompat += ncompat;
    }
    std::cout << "[Total " << totcompat << " compatible matches]" << std::endl;
  }

  // ==========================================================================
  // FIT PARAMETERS
  // ---------------------------

  void LArFlowFlashMatch::prepareFitter() {
    _fitter.clear();
    _pair2matchidx.clear();
    _matchidx2pair.clear();
    
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {
      const FlashData_t& flash = _flashdata_v[iflash];
      FlashData_t flashrenorm =  flash;
      for ( size_t ich=0; ich<flash.size(); ich++ )
	flashrenorm[ich] = flash.tot*flash[ich];
      
      for (size_t iclust=0; iclust<_qcluster_v.size(); iclust++) {
	if ( getCompat(iflash,iclust)!=kUncut )
	  continue;

	QClusterComposite& qcomposite = _qcomposite_v[iclust];
	CutVars_t& cutvars = getCutVars( iflash, iclust );

	int imatchidx = -1;
	if ( cutvars.maxdist_wext < cutvars.maxdist_noext ) {
	  FlashHypo_t hypo = qcomposite.generateFlashCompositeHypo( flash, true ).makeHypo();
	  imatchidx = _fitter.addMatchPair( iflash, iclust, flashrenorm, hypo );
	}
	else {
	  FlashHypo_t hypo = qcomposite.generateFlashCompositeHypo( flash, false ).makeHypo();
	  imatchidx = _fitter.addMatchPair( iflash, iclust, flashrenorm, hypo );
	}
	
	_pair2matchidx[ MatchPair_t( iflash, iclust ) ] = imatchidx;
	_matchidx2pair[ imatchidx ] = MatchPair_t( iflash, iclust );
      }
      std::cout << "after flash=" << iflash << std::endl;
    }


  }

  // clear and reset fitter
  void LArFlowFlashMatch::clearFitter() {
    _parsdefined = false;
    _pair2matchidx.clear();
    _matchidx2pair.clear();
    _fitter.clear();
  }

  // Set Initial fmatch vector using best fit
  void LArFlowFlashMatch::setInitialFlashMatchVector() {

    std::map<int,float> clust_bestmaxdist;
    std::map<int,int>   clust_bestidx;
    
    for ( auto& it_match : _matchidx2pair ) {
      int imatch = it_match.first;
      int iflash = it_match.second.flashidx;
      int iclust = it_match.second.clusteridx;

      CutVars_t& cutvars = getCutVars(iflash,iclust);
      
      if ( clust_bestmaxdist.find( iclust )==clust_bestmaxdist.end() ) {
	clust_bestmaxdist[iclust] = 1.0;
	clust_bestidx[iclust]     = -1;
      }
      
      float maxdist = ( cutvars.maxdist_wext < cutvars.maxdist_noext ) ? cutvars.maxdist_wext : cutvars.maxdist_noext;
      if ( maxdist < clust_bestmaxdist[iclust] ) {
	clust_bestmaxdist[iclust] = maxdist;
	clust_bestidx[iclust] = imatch;
      }
    }

    std::vector<float> fmatch_init( _matchidx2pair.size() );
    for ( auto& it_match : _matchidx2pair ) {
      int imatch = it_match.first;
      int iflash = it_match.second.flashidx;
      int iclust = it_match.second.clusteridx;

      if ( imatch==clust_bestidx[iclust] )
	//fmatch_init[imatch] = 0.9;
	fmatch_init[imatch] = 1.0;
      else
	//fmatch_init[imatch] = 0.1;
	fmatch_init[imatch] = 0.0;
    }
    _fitter.setFMatch( fmatch_init );
    
  }
  
  // ==========================================================================
  // DEBUG: DumpQCompositeImage
  // ---------------------------
  
  void LArFlowFlashMatch::dumpQCompositeImages( std::string prefix ) {

    gStyle->SetOptStat(0);
    
    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();    
    const float  driftv = larp->DriftVelocity();    

    TCanvas c2d("c2d","pmt flash", 1500, 800);
    TPad datayz("pad1", "",0.0,0.7,0.8,1.0);
    TPad hypoyz("pad2", "",0.0,0.4,0.8,0.7);
    TPad dataxy("pad3", "",0.8,0.7,1.0,1.0);
    TPad hypoxy("pad4", "",0.8,0.4,1.0,0.7);
    TPad histpad("pad5","",0.0,0.0,1.0,0.4);
    datayz.SetRightMargin(0.05);
    datayz.SetLeftMargin(0.05);    
    hypoyz.SetRightMargin(0.05);
    hypoyz.SetLeftMargin(0.05);
    

    // shapes/hists used for each plot
    TH2D bg("hyz","",100,-50,1080, 120, -130, 130);
    TH2D bgxy("hxy","",25,-300,550, 120, -130, 130);
    TBox boxzy( 0, -116.5, 1036, 116.5 );
    boxzy.SetFillStyle(0);
    boxzy.SetLineColor(kBlack);
    boxzy.SetLineWidth(1);
    TBox boxxy( 0, -116.5, 256, 116.5 );
    boxxy.SetFillStyle(0);
    boxxy.SetLineColor(kBlack);
    boxxy.SetLineWidth(1);

    // badch indicator
    std::vector< TBox* > zy_deadregions;
    for (int p=2; p<3; p++) {
      const larcv::ChStatus& status = _evstatus->status( p );
      int maxchs = ( p<=1 ) ? 2400 : 3456;
      bool inregion = false;
      int regionstart = -1;
      int currentregionwire = -1;
      for (int ich=0; ich<maxchs; ich++) {
	
	if ( !inregion && status.status(ich)!=4 ) {
	  inregion = true;
	  regionstart = ich;
	  currentregionwire = ich;
	}
	else if ( inregion && status.status(ich)!=4 ) {
	  currentregionwire = ich;
	}
	else if ( inregion && status.status(ich)==4 ) {
	  // end a region, make a box!
	  TBox* badchs = new TBox( (float)(0.3*regionstart), -115, (float)(0.3*currentregionwire), 115 );
	  badchs->SetFillColor( 19 );
	  badchs->SetLineColor( 0 );
	  zy_deadregions.push_back( badchs );
	  inregion = false;
	}
      }
    }//end of plane loop for dead channels

    // pmt markers
    std::vector<TEllipse*> pmtmarkers_v(32,0);
    std::vector<TText*>    chmarkers_v(32,0);
    
    for (int ich=0; ich<32; ich++) {
      int opdet = geo->OpDetFromOpChannel(ich);
      double xyz[3];
      geo->GetOpChannelPosition( ich, xyz );
      
      pmtmarkers_v[ich] = new TEllipse(xyz[2],xyz[1], 10.0, 10.0);
      pmtmarkers_v[ich]->SetLineColor(kBlack);

      char pmtname[10];
      sprintf(pmtname,"%02d",ich);
      chmarkers_v[ich] = new TText(xyz[2]-10.0,xyz[1]-5.0,pmtname);
      chmarkers_v[ich]->SetTextSize(0.04);

    }
    
    // make charge graphs
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {

      // refresh the plot
      c2d.Clear();
      c2d.Draw();
      c2d.cd();
      
      // Draw the pads
      datayz.Draw();
      hypoyz.Draw();
      dataxy.Draw();
      hypoxy.Draw();
      histpad.Draw();
      
      const FlashData_t& flash = _flashdata_v[iflash];
      float xoffset = (flash.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
      std::vector< TGraph > graphs_zy_v;
      std::vector< TGraph > graphs_xy_v;
      int nclusters_drawn = 0;
      char histname_data[50];
      sprintf(histname_data,"hflashdata_%d",(int)iflash);
      TH1D hflashdata(histname_data,"",32,0,32);
      hflashdata.SetLineWidth(3);
      hflashdata.SetLineColor(kBlack);

      std::vector<TEllipse*> datamarkers_v(32,0);
      float norm = flash.tot;
      for (size_t ich=0; ich<32; ich++) {
	
	int opdet = geo->OpDetFromOpChannel(ich);
	double xyz[3];
	geo->GetOpChannelPosition( ich, xyz );
	
	float pe   = (flash[ich]*norm);
	hflashdata.SetBinContent( ich+1, pe );
	
	if ( pe>10 )
	  pe = 10 + (pe-10)*0.10;
	float radius = ( pe>50 ) ? 50 : pe;
	datamarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
	datamarkers_v[ich]->SetFillColor(29);
      }

      // truth-matched track for flash
      TGraph* mctrack_data_zy = nullptr;
      TGraph* mctrack_data_xy = nullptr;
      bool mctrack_match = false;
      if ( flash.mctrackid>=0 )
	mctrack_match = true;      
      if ( mctrack_match ) {
	const larlite::mctrack& mct = (*_mctrack_v)[ _mctrackid2index[flash.mctrackid] ];
	mctrack_data_zy = new TGraph( mct.size() );
	mctrack_data_xy = new TGraph( mct.size() );	
	for (int istep=0; istep<(int)mct.size(); istep++) {
	  mctrack_data_zy->SetPoint(istep, mct[istep].Z(), mct[istep].Y() );
	  mctrack_data_xy->SetPoint(istep, mct[istep].X(), mct[istep].Y() );
	}
	mctrack_data_zy->SetLineColor(kBlue);
	mctrack_data_zy->SetLineWidth(1);
	mctrack_data_xy->SetLineColor(kBlue);
	mctrack_data_xy->SetLineWidth(1);
      }

      // BEGIN CLUSTER LOOP
      std::vector< TGraph* > mctrack_clustertruth_xy_v;
      std::vector< TGraph* > mctrack_clustertruth_zy_v;
      std::vector< TH1D* > flash_hypotheses_v;
      for (size_t iclust=0; iclust<_qcluster_v.size(); iclust++) {

	int compat = getCompat( iflash, iclust );
	if ( compat!=0 )
	  continue;

	const QCluster_t& qclust = _qcluster_v[iclust];
	CutVars_t& cutvars = getCutVars(iflash,iclust);

	// truth matched to track cluster
	TGraph* mctrack_hypo_zy = nullptr;
	TGraph* mctrack_hypo_xy = nullptr;
	bool has_clust_truthmatch = false;
	if ( qclust.mctrackid>=0 ) {
	  has_clust_truthmatch = true;
	  const larlite::mctrack& mct = (*_mctrack_v)[ _mctrackid2index[qclust.mctrackid] ];
	  mctrack_hypo_zy = new TGraph( mct.size() );
	  mctrack_hypo_xy = new TGraph( mct.size() );	
	  for (int istep=0; istep<(int)mct.size(); istep++) {
	    mctrack_hypo_zy->SetPoint(istep, mct[istep].Z(), mct[istep].Y() );
	    mctrack_hypo_xy->SetPoint(istep, mct[istep].X(), mct[istep].Y() );
	  }
	  mctrack_hypo_zy->SetLineColor(kMagenta);
	  mctrack_hypo_zy->SetLineWidth(1);
	  mctrack_hypo_xy->SetLineColor(kMagenta);
	  mctrack_hypo_xy->SetLineWidth(1);
	  mctrack_clustertruth_xy_v.push_back( mctrack_hypo_xy );
	  mctrack_clustertruth_zy_v.push_back( mctrack_hypo_zy );	  
	}
	
	//std::cout << "[larflow::LArFlowFlashMatch::dumpQClusterImages][INFO] Cluster " << iclust << std::endl;
	const QClusterComposite& qcomposite  = _qcomposite_v[iclust];
	std::vector<TGraph> gtrack_v = qcomposite.getTGraphs( xoffset );

	Int_t colors[4] = { kRed, kOrange+2, kCyan, kMagenta };
	for ( size_t ig=0; ig<gtrack_v.size(); ig++) {
	  auto& g = gtrack_v[ig];
	  g.SetMarkerStyle(20);
	  g.SetMarkerSize(0.3);
	  g.SetMarkerColor(colors[ig/2]);
	  if ( g.GetN()>0 ) {
	    if ( ig%2==0 )
	      graphs_zy_v.emplace_back( std::move(g) );
	    else
	      graphs_xy_v.emplace_back( std::move(g) );
	  }
	}


	FlashCompositeHypo_t comphypo_wext  = qcomposite.generateFlashCompositeHypo( flash, true );
	FlashCompositeHypo_t comphypo_noext = qcomposite.generateFlashCompositeHypo( flash, false );

	FlashHypo_t hypo_wext = comphypo_wext.makeHypo();
	FlashHypo_t hypo_noext = comphypo_noext.makeHypo();	

	char histname1[100];
	sprintf(histname1, "hhypo_flash%d_clust%d", (int)iflash, (int)iclust );
	TH1D* hhypo = new TH1D(histname1,"",32,0,32);
	if ( cutvars.maxdist_wext > cutvars.maxdist_noext )
	  hhypo->SetLineStyle( 2 );
	if ( iclust==flash.truthmatched_clusteridx ) {
	  hhypo->SetLineColor(kGreen+2);
	  hhypo->SetLineWidth( 2 );
	}
	
	for (int ich=0; ich<32; ich++) {
	  if ( cutvars.maxdist_wext < cutvars.maxdist_noext )
	    hhypo->SetBinContent( ich+1, hypo_wext[ich] );
	  else
	    hhypo->SetBinContent( ich+1, hypo_noext[ich] ); 
	}
	flash_hypotheses_v.push_back( hhypo );
	nclusters_drawn++;
      }//end of cluster loop

      // ----------------------------------------------
      // TOP YZ-2D PLOT: FLASH-DATA/FLASH TRUTH TRACK
      // ----------------------------------------------
      datayz.cd();
      bg.Draw();
      boxzy.Draw();
      for ( auto& pbadchbox : zy_deadregions ) {
	pbadchbox->Draw();
      }
      
      for (int ich=0; ich<32; ich++) {
	pmtmarkers_v[ich]->Draw();	  
	datamarkers_v[ich]->Draw();
      }
      
      for (size_t ig=0; ig<graphs_zy_v.size(); ig++ ) {
	graphs_zy_v[ig].Draw("P");
      }	
      
      for (int ich=0; ich<32; ich++)
	chmarkers_v[ich]->Draw();

      if ( mctrack_match )
	mctrack_data_zy->Draw("L");
      
      // ---------------------------------------------
      // TOP XY-2D PLOT: FLASH-DATA/FLASH TRUTH TRACK
      // ---------------------------------------------
      dataxy.cd();
      bgxy.Draw();
      boxxy.Draw();
      
      for (size_t ig=0; ig<graphs_xy_v.size(); ig++ ) {
	graphs_xy_v[ig].Draw("P");
      }

      if ( mctrack_match )
	mctrack_data_xy->Draw("L");
      
      // --------------------------------------------------
      // BOTTOM YZ-2D PLOT: FLASH-HYPO/CLUSTER TRUTH TRACK
      // --------------------------------------------------
      hypoyz.cd();
      bg.Draw();
      boxzy.Draw();
      for ( auto& pbadchbox : zy_deadregions ) {
	pbadchbox->Draw();
      }
      
      for (int ich=0; ich<32; ich++) {
	pmtmarkers_v[ich]->Draw();	  
	// hypo
      }

      for ( auto& ptrack : mctrack_clustertruth_zy_v )
	ptrack->Draw("L");
      
      for (int ich=0; ich<32; ich++)
	chmarkers_v[ich]->Draw();
      
      // -----------------------------------------------------
      // BOTTOM XY-2D PLOT: FLASH-HYPO/TRUTH TRACK FOR CLUSTER
      // -----------------------------------------------------
      hypoxy.cd();
      bgxy.Draw();
      boxxy.Draw();
            
      for ( auto& ptrack : mctrack_clustertruth_xy_v )
	ptrack->Draw("L");

      // -------------------------------
      // FLASH PE
      // -------------------------------
      histpad.cd();
      hflashdata.Draw("hist");
      for ( auto& phist : flash_hypotheses_v )
	phist->Draw("hist same");

      c2d.Update();
      c2d.Draw();

      char cname[100];
      sprintf(cname,"%s_qcomposite_flash%02d.png",prefix.c_str(),(int)iflash);
      c2d.SaveAs(cname);

      std::cout << "number of clusters draw: " << nclusters_drawn << std::endl;
      std::cout << "graph_zy: " << graphs_zy_v.size() << std::endl;
      std::cout << "graph_xy: " << graphs_xy_v.size() << std::endl;            
      std::cout << "[enter to continue]" << std::endl;
      //std::cin.get();

      for (int ich=0; ich<32; ich++) {
	delete datamarkers_v[ich];
      }

      if ( mctrack_match ) {
	delete mctrack_data_zy;
	delete mctrack_data_xy;
      }

      for ( auto& ptrack : mctrack_clustertruth_zy_v )
	delete ptrack;
      for ( auto& ptrack : mctrack_clustertruth_xy_v )
	delete ptrack;
      for ( auto& phist : flash_hypotheses_v )
	delete phist;
      
    }//end of flash loop


    // clean up vis items
    for (int ich=0; ich<32; ich++) {
      delete pmtmarkers_v[ich];
      delete chmarkers_v[ich];
    }
    for (int i=0; i<(int)zy_deadregions.size(); i++) {
      delete zy_deadregions[i];
    }
  }
  

  /*
  float LArFlowFlashMatch::generateProposal( const float hamdist_mean, const float lydist_mean, const float lydist_sigma,
					     std::vector<float>& match_v, float& ly  ) {
    // generate number of flips
    //int nflips = _rand->Poisson( hamdist_mean );
    int nflips = (int)fabs(_rand->Exp(hamdist_mean));
    
    // draw index
    std::set<int> flip_idx;
    while ( nflips>0 && flip_idx.size()!=nflips ) {
      int idx = _rand->Integer( _nmatches );
      flip_idx.insert(idx);
    }
    
    // generate ly change
    float proposal_ly = -1.;
    float dly = 0.;
    while (proposal_ly<0) {
      dly = _rand->Gaus( 0, lydist_sigma );
      proposal_ly = ly + dly;
    }
    ly = proposal_ly;


    // copy state vector
    //match_v.resize( _nmatches );
    //memcpy( match_v.data(), fmatch, sizeof(float)*_nmatches );
    // do the flips
    for (auto& idx : flip_idx ) {
      match_v[idx] = (match_v[idx]>0.5 ) ? 0.0 : 1.0;
    }

    // return prob of jump
    float prob_hamdist = TMath::Poisson( hamdist_mean, nflips );
    float prob_ly      = TMath::Gaus( dly/lydist_sigma );
    
    return float(flip_idx.size());
  }
  */
  
  // ==============================================================================================
  // MC TRUTH FUNCTIONS
  // ==============================================================================================
  
  void LArFlowFlashMatch::loadMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v, bool do_truth_matching ) {
    _mctrack_v = &mctrack_v;
    _kDoTruthMatching = do_truth_matching;
    _kFlashMatchedDone = false;
    std::cout << "[LArFlowFlashMatch::loadMCTrackInfo][INFO] Loaded MC tracks." << std::endl;
  }

  void LArFlowFlashMatch::doFlash2MCTrackMatching( std::vector<FlashData_t>& flashdata_v ) {

    //space charge and time service; ideally initialized in algo constructor
    if ( _psce==nullptr ){
      _psce = new ::larutil::SpaceChargeMicroBooNE;
    }
    // note on TimeService: if not initialized from root file via GetME(true)
    // needs trig_offset hack in tufts_larflow branch head
    const ::larutil::TimeService* tsv = ::larutil::TimeService::GetME(false);

    std::vector<std::vector<int>>   track_id_match_v(flashdata_v.size());
    std::vector<std::vector<int>>   track_pdg_match_v(flashdata_v.size());
    std::vector<std::vector<int>>   track_mid_match_v(flashdata_v.size());
    std::vector<std::vector<float>> track_E_match_v(flashdata_v.size());
    std::vector<std::vector<float>> track_dz_match_v(flashdata_v.size());
    
    int imctrack=-1;
    _mctrackid2index.clear();
    _nu_mctrackid.clear();
    
    for (auto& mct : *_mctrack_v ) {
      imctrack++;
      _mctrackid2index[mct.TrackID()] = imctrack;
      if ( mct.Origin()==1 )
	_nu_mctrackid.insert( mct.TrackID() );
      
      // get time
      //float track_tick = tsv->TPCG4Time2Tick( mct.Start().T() );
      float track_tick = mct.Start().T()*1.0e-3/0.5 + 3200;
      float flash_time = tsv->OpticalG4Time2TDC( mct.Start().T() );

      int nmatch = 0;
      int   best_flashidx =   -1;      
      float best_dtick    = 1e9;
      for ( size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
	float dtick = fabs(flashdata_v[iflash].tpc_tick - track_tick);
	//std::cout << "  iflash[" << iflash << "] dtick=" << dtick << std::endl;
	if ( dtick < 10 ) { // space charge?
	  track_id_match_v[iflash].push_back( mct.TrackID() );
	  track_mid_match_v[iflash].push_back( mct.MotherTrackID() );	  
	  track_pdg_match_v[iflash].push_back( mct.PdgCode() );
	  track_E_match_v[iflash].push_back( mct.Start().E() );
	  track_dz_match_v[iflash].push_back( fabs( mct.Start().Z()-flashdata_v[iflash].maxchposz) );
	  nmatch++;
	}
	if ( best_dtick > dtick ) {
	  best_dtick    = dtick;
	  best_flashidx = iflash;
	}
      }
      // FOR DEBUG
      // std::cout << "mctrack[" << mct.TrackID() << ",origin=" << mct.Origin() << "] pdg=" << mct.PdgCode()
      // 		<< " E=" << mct.Start().E() 
      // 		<< " T()=" << mct.Start().T() << " track_tick=" << track_tick << " optick=" << flash_time
      // 		<< " best_flashidx="  << best_flashidx << " "
      // 		<< " best_dtick=" << best_dtick
      // 		<< " nmatched=" << nmatch << std::endl;
    }

    // now loop over flashes
    for (size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
      std::vector<int>& id = track_id_match_v[iflash];
      std::vector<int>& pdg = track_pdg_match_v[iflash];
      std::vector<int>& mid = track_mid_match_v[iflash];
      std::vector<float>& dz = track_dz_match_v[iflash];      

      if ( id.size()==1 ) {
	// easy!!
	flashdata_v[iflash].mctrackid  = id.front();
	flashdata_v[iflash].mctrackpdg = pdg.front();
	if ( _nu_mctrackid.find( id[0] )!=_nu_mctrackid.end()  )
	  flashdata_v[iflash].isneutrino = true;
	  
      }
      else if (id.size()>1 ) {
	// to resolve multiple options.
	// (1) favor id==mid && pdg=|13| (muon) -- from these, pick best time
	int nmatches = 0;
	int idx = -1;
	int pdgx = -1;
	float closestz = 10000;
	bool isnu = false;	
	for (int i=0; i<(int)id.size(); i++) {
	  bool trackisnu = (_nu_mctrackid.find(id[i])!=_nu_mctrackid.end());
	  //std::cout << "  multiple-truthmatches[" << i << "] id=" << id[i] << " mid=" << mid[i] << " pdg=" << pdg[i] << " dz=" << dz[i] << " isnu=" << trackisnu << std::endl;
	  if ( (id[i]==mid[i] && dz[i]<closestz) || (trackisnu && flashdata_v[iflash].intime) ) {
	    idx = id[i];
	    pdgx = pdg[i];
	    closestz = dz[i];
	    //nmatches++;
	    if ( trackisnu )
	      isnu = true;
	  }
	}
	flashdata_v[iflash].mctrackid = idx;
	flashdata_v[iflash].mctrackpdg = pdgx;
	flashdata_v[iflash].isneutrino = isnu;
	
      }// if multipl matched ids
      int nmcpts = (*_mctrack_v)[ _mctrackid2index[flashdata_v[iflash].mctrackid] ].size();
      // std::cout << "FlashMCtrackMatch[" << iflash << "] "
      // 		<< "tick=" << flashdata_v[iflash].tpc_tick << " "
      // 		<< "nmatches=" << id.size() << " "
      // 		<< "trackid=" << flashdata_v[iflash].mctrackid << " "
      // 		<< "pdg=" << flashdata_v[iflash].mctrackpdg << " "
      // 		<< "isnu=" << flashdata_v[iflash].isneutrino << " "
      // 		<< "intime=" << flashdata_v[iflash].intime << " "
      // 		<< "isbeam=" << flashdata_v[iflash].isbeam << " "			
      // 		<< "nmcpts=" << nmcpts << std::endl;
    }
    _kFlashMatchedDone = true;
  }

  void LArFlowFlashMatch::doTruthCluster2FlashTruthMatching( std::vector<FlashData_t>& flashdata_v, std::vector<QCluster_t>& qcluster_v ) {
    for (int iflash=0; iflash<(int)flashdata_v.size(); iflash++) {
      FlashData_t& flash = flashdata_v[iflash];

      for (int iclust=0; iclust<(int)qcluster_v.size(); iclust++) {
	QCluster_t& cluster = qcluster_v[iclust];
	bool isnu = _nu_mctrackid.find( cluster.mctrackid )!=_nu_mctrackid.end();
	
	if ( flash.mctrackid!=-1 &&
	     ( flash.mctrackid==cluster.mctrackid || (flash.isneutrino && isnu) ) )
	  {
	    flash.truthmatched_clusteridx = iclust;
	    cluster.truthmatched_flashidx = iflash;
	    cluster.isneutrino = flash.isneutrino;
	    break;
	  }
      }
    }
  }

  // void LArFlowFlashMatch::buildClusterExtensionsWithMCTrack( bool appendtoclusters, std::vector<QCluster_t>& qcluster_v ) {
  //   // we create hits outside the tpc for each truth-matched cluster
  //   // outside of TPC, so no field, no space-charge correction
  //   // -----
  //   // 1) we find when we first cross the tpc and exit
  //   // 2) at these points we draw a line of charge using the direction at the crossing point

  //   const float maxstepsize=1.0; // (10 pixels)
  //   const float maxextension=30.0; // 1 meters (we stop when out of cryostat)
  //   const larutil::LArProperties* larp = larutil::LArProperties::GetME();
  //   const float  driftv = larp->DriftVelocity();    
    
  //   for ( int iclust=0; iclust<(int)qcluster_v.size(); iclust++) {
  //     QCluster_t& cluster = qcluster_v[iclust];
  //     if ( cluster.mctrackid<0 )
  // 	continue;
  //     const larlite::mctrack& mct = (*_mctrack_v).at( _mctrackid2index[cluster.mctrackid] );

  //     int nsteps = (int)mct.size();
  //     QCluster_t ext1;
  //     QCluster_t ext2;
  //     ext1.reserve(200);
  //     ext2.reserve(200);
  //     bool crossedtpc = false;
      
  //     for (int istep=0; istep<nsteps-1; istep++) {

  // 	const larlite::mcstep& thisstep = mct[istep];	
  // 	const larlite::mcstep& nextstep = mct[istep+1];

  // 	double thispos[4] = { thisstep.X(), thisstep.Y(), thisstep.Z(), thisstep.T() };
  // 	double nextpos[4] = { nextstep.X(), nextstep.Y(), nextstep.Z(), nextstep.T() };

  // 	  // are we in the tpc?
  // 	bool intpc = false;
  // 	if ( nextpos[0]>0 && nextpos[0]<256 && nextpos[1]>-116.5 && nextpos[1]<116.5 && nextpos[2]>0 && nextpos[2]<1036.0 )
  // 	  intpc = true;
	
  // 	if ( (intpc && !crossedtpc) || (!intpc && crossedtpc) ) {
  // 	  // entering point/make extension

  // 	  // make extension
  // 	  double dirstep[4];	
  // 	  double steplen = maxstepsize;
  // 	  int nsubsteps = maxextension/steplen;
	  
  // 	  for (int i=0; i<4; i++) {
  // 	    dirstep[i] = nextpos[i]-thispos[i];
  // 	    if ( i<3 )
  // 	      steplen += dirstep[i]*dirstep[i];
  // 	  }
  // 	  steplen = sqrt(steplen);
  // 	  for (int i=0; i<3; i++)
  // 	    dirstep[i] /= steplen;
  // 	  double stepdtick = dirstep[4]/nsubsteps;
	  
  // 	  double* extpos = nullptr;
  // 	  if ( !crossedtpc ) {
  // 	    // entering point, we go backwards
  // 	    for (int i=0; i<3; i++) dirstep[i] *= -1;
  // 	    extpos = nextpos;
  // 	  }
  // 	  else {
  // 	    extpos = thispos;
  // 	  }
	  
  // 	  for (int isub=0; isub<nsubsteps; isub++) {
  // 	    double pos[4];
  // 	    for (int i=0; i<3; i++) 
  // 	      pos[i] = extpos[i] + (double(isub)+0.5)*steplen*dirstep[i];
  // 	    pos[3] = extpos[3] + ((double)isub+0.5)*stepdtick;

  // 	    // once pos is outside cryostat, stop
  // 	    if ( pos[0]<-25 ) // past the plane of the pmts
  // 	      break;
  // 	    if ( pos[0]>260 ) // visibilty behind the cathode seems fucked up
  // 	      break;
  // 	    if ( pos[1]>150.0 )
  // 	      break;
  // 	    if ( pos[1]<-150.0 )
  // 	      break;
  // 	    if ( pos[2] < -60 )
  // 	      break;
  // 	    if ( pos[2] > 1100 )
  // 	      break;
	    
  // 	    // determine which ext
  // 	    QCluster_t& ext = (crossedtpc) ? ext2 : ext1;
  // 	    QPoint_t qpt;
  // 	    qpt.xyz.resize(3,0);
  // 	    qpt.xyz[0] = pos[0];
  // 	    qpt.xyz[1] = pos[1];
  // 	    qpt.xyz[2] = pos[2];
  // 	    qpt.tick   = (pos[3]*1.0e-3)/0.5 + 3200;
  // 	    // x-offset from tick relative to trigger, then actual x-depth
  // 	    // we do this, because reco will be relative to cluster which has this x-offset
  // 	    qpt.xyz[0] = (qpt.tick-3200)*0.5*driftv + pos[0];
  // 	    qpt.pixeladc = steplen; // here we leave distance. photon hypo must know what to do with this
  // 	    if ( pos[0]>250 ) {
  // 	      // this is a hack.
  // 	      // we want the extension points to the gap filler works.
  // 	      // but the visi behind cathode is messed up so we want no contribution
  // 	      qpt.pixeladc = 0.;
  // 	    }
  // 	    qpt.fromplaneid = -1;
  // 	    qpt.type = kExt;
  // 	    ext.emplace_back( std::move(qpt) );
		    
  // 	  }//end of extension
  // 	  crossedtpc = true; // mark that we have crossed	  
  // 	}
  //     }//end of step loop

  //     // what to do with these?
  //     if ( appendtoclusters ) {
  // 	for (auto& qpt : ext1 ) {
  // 	  cluster.emplace_back( std::move(qpt) );
  // 	}
  // 	for (auto& qpt : ext2 ) {
  // 	  cluster.emplace_back( std::move(qpt) );
  // 	}
  //     }
  //     else {
  // 	if ( ext1.size()>0 )
  // 	  qcluster_v.emplace_back( std::move(ext1) );
  // 	if ( ext2.size()>0 )
  // 	  qcluster_v.emplace_back( std::move(ext2) );
  //     }

  //     // for debug: cluster check =========================
  //     // std::cout << "extendcluster[" << iclust << "] --------------------------------------" << std::endl;
  //     // for (auto const& hit : cluster ) {
  //     // 	std::cout << "  (" << hit.xyz[0] << "," << hit.xyz[1] << "," << hit.xyz[2] << ") intpc=" << hit.intpc << std::endl;
  //     // }
  //     // std::cout << "[enter for next]" << std::endl;
  //     // std::cin.get();
  //     // ==================================================
      
  //   }// end of cluster loop
  // }

  void LArFlowFlashMatch::clearMCTruthInfo() {
    _mctrack_v = nullptr;
    _mctrackid2index.clear();
    _nu_mctrackid.clear();
    _flash_truthid.clear();
    _cluster_truthid.clear();
    _flash2truecluster.clear();
    _cluster2trueflash.clear();
    delete _psce;
    _psce = nullptr;
    _kDoTruthMatching  = false;
    _kFlashMatchedDone = false;
  }

  std::vector<larlite::larflowcluster> LArFlowFlashMatch::exportMatchedTracks() {
  //   // for each cluster, we use the best matching flash
  //   const larutil::LArProperties* larp = larutil::LArProperties::GetME();
  //   const float  usec_per_tick = 0.5; // usec per tick
  //   const float  tpc_trigger_tick = 3200;
  //   const float  driftv = larp->DriftVelocity();
    
    std::vector<larlite::larflowcluster> lfcluster_v(_qcluster_v.size());
    
  //   for (int iclust=0; iclust<_qcluster_v.size(); iclust++) {
  //     const QCluster_t& cluster = _qcluster_v[iclust];
  //     larlite::larflowcluster& lfcluster = lfcluster_v[iclust];
  //     lfcluster.resize(cluster.size());
      
  //     // is it flash matched?
  //     std::vector<float> matchscores = getMatchScoresForCluster(iclust);
  //     float maxfmatch = 0.;
  //     int   maxfmatch_idx = -1;      
  //     for (int iflash=0; iflash<_flashdata_v.size(); iflash++) {
  // 	if ( matchscores[iflash]>maxfmatch ) {
  // 	  maxfmatch = matchscores[iflash];
  // 	  maxfmatch_idx = iflash;
  // 	}
  //     }

  //     float matched_flash_tick = 0;
  //     float matched_flash_xoffset = 0;
  //     if ( maxfmatch_idx>=0 ) {
  // 	matched_flash_tick    = _flashdata_v[maxfmatch_idx].tpc_tick;
  // 	matched_flash_xoffset = (matched_flash_tick-tpc_trigger_tick)*usec_per_tick/driftv;
  //     }
      
  //     // transfer hit locations back to larlite
      
  //     int nhits=0;
  //     for (int ihit=0; ihit<(int)cluster.size(); ihit++) {
  // 	const QPoint_t& qpt = cluster.at(ihit);
  // 	larlite::larflow3dhit& lfhit = lfcluster[ihit];
  // 	lfhit.resize(3,0);
  // 	for (int i=0; i<3; i++)
  // 	  lfhit[i] = qpt.xyz[i];
  // 	lfhit.tick = qpt.tick;

  // 	if ( maxfmatch_idx>=0 ) {
  // 	  // match found, we shift the x positions
  // 	  lfhit[0]   -= matched_flash_xoffset;
  // 	  lfhit.tick -= matched_flash_tick; // now this is delta-t from t0
  // 	  // since we have this position, we can invert spacecharge effect
  // 	  // TODO
  // 	}
  // 	bool hitok = true;
  // 	for (int i=0; i<3; i++)
  // 	  if ( std::isnan(lfhit[i]) )
  // 	    hitok = false;

  // 	if (hitok)
  // 	  nhits++;
	
  // 	// here we should get larflow3dhit metadata and pass it on TODO
  //     }//end of hit loop

  //     // this is in case we learn how to filter out points
  //     // then we throw away excess capacity
  //     lfcluster.resize(nhits); 

  //     // metadata
  //     // --------

  //     // matched flash (reco)
  //     if ( maxfmatch_idx>=0 ) {
  // 	lfcluster.isflashmatched = 1;
  // 	lfcluster.flash_tick = matched_flash_tick;
  //     }
  //     else {
  // 	lfcluster.isflashmatched = -1;
  // 	lfcluster.flash_tick = -1;
  //     }

  //     // matched flash (truth)
  //     lfcluster.truthmatched_mctrackid = cluster.mctrackid;
  //     if ( cluster.truthmatched_flashidx>=0 )
  // 	lfcluster.truthmatched_flashtick = _flashdata_v[cluster.truthmatched_flashidx].tpc_tick;
  //     else
  // 	lfcluster.truthmatched_flashtick = -1;
      
  //   }//end of cluster loop

    return lfcluster_v;
  }


  void LArFlowFlashMatch::setFitParsWithTruthMatch() {
    // set the fitter parameters using truth-matching
    
    if ( !_kFlashMatchedDone ) {
      throw std::runtime_error("[larflow::LArFlowFlashMatch::setFitParsWithTruthMatch][ERROR] Truth-based flash-matching not yet done.");
    }

    std::vector<float> fmatch_truth( _matchidx2pair.size(), 0. );

    for ( auto& it_match : _matchidx2pair ) {
      
      int imatch = it_match.first;
      int iflash = it_match.second.flashidx;
      int iclust = it_match.second.clusteridx;

      const FlashData_t& flash = _flashdata_v[iflash];
      if ( flash.truthmatched_clusteridx==iclust )
	fmatch_truth[imatch] = 1;
      else
	fmatch_truth[imatch] = 0.;
    }

    _fitter.setFMatch( fmatch_truth );

    for ( auto& it_match : _matchidx2pair ) {
      
      int imatch = it_match.first;
      int iflash = it_match.second.flashidx;
      int iclust = it_match.second.clusteridx;

      CutVars_t& cutvars = getCutVars(iflash,iclust);
      
      cutvars.truthscore = _fitter.scoreMatch( imatch );

    }

  }
    
  // ---------------------------------------------------------------------
  // Ana Tree
  // ---------------------------------------------------------------------

  void LArFlowFlashMatch::saveAnaVariables( std::string anafilename ) {
    _ana_filename = anafilename;
    setupAnaTree();
  }
  
  void LArFlowFlashMatch::setupAnaTree() {
    _fanafile = new TFile( _ana_filename.c_str(), "recreate" );
    _anatree  = new TTree("flashmatchana", "LArFlow FlashMatch Ana Tree");
    _anatree->Branch("run",          &_run,          "run/I");
    _anatree->Branch("subrun",       &_subrun,       "subrun/I");
    _anatree->Branch("event",        &_event,        "event/I");
    _anatree->Branch("cutfailed",    &_cutfailed,    "cutfailed/I");
    _anatree->Branch("truthmatch",   &_truthmatch,   "truthmatch/I");
    _anatree->Branch("isneutrino",   &_isneutrino,   "isneutrino/I");        
    _anatree->Branch("intime",       &_intime,       "intime/I");        
    _anatree->Branch("isbeam",       &_isbeam,       "isbeam/I");
    _anatree->Branch("hypope",       &_hypope,       "hypope/F");
    _anatree->Branch("datape",       &_datape,       "datape/F");
    _anatree->Branch("dtickwin",     &_dtick_window, "dtickwin/F");
    _anatree->Branch("maxdist",      &_maxdist_best, "maxdist/F");
    _anatree->Branch("maxdist_wext", &_maxdist_wext, "maxdist_wext/F");
    _anatree->Branch("maxdist_noext",&_maxdist_noext,"maxdist_noext/F");
    _anatree->Branch("peratio",      &_peratio_best, "peratio/F");
    _anatree->Branch("peratio_wext", &_peratio_wext, "peratio_wext/F");
    _anatree->Branch("peratio_noext",&_peratio_noext,"peratio_noext/F");
    _anatree->Branch("enterlen",     &_enterlen,     "enterlen/F");
    _anatree->Branch("fmatch",       &_fmatch,       "fmatch/F");
    _anatree->Branch("fmatch_truth", &_fmatch_truth, "fmatch_truth/F");
    
    _save_ana_tree   = true;
    _anafile_written = false;    
  }

  void LArFlowFlashMatch::clearAnaVariables() {
    _cutfailed  = -1;
    _truthmatch = -1;
    _isneutrino = 0;
    _intime     = 0;
    _isbeam     = 0;
    _hypope     = 0;
    _datape     = 0;
    _dtick_window = 0;
    _maxdist_best = -1;
    _maxdist_wext = -1;
    _maxdist_noext = -1;
    _peratio_best = 0;
    _peratio_wext = 0;
    _peratio_noext = 0;
    _enterlen = -1;
    _fmatch = 0;
    _fmatch_truth=0;
  }

  void LArFlowFlashMatch::setRSE( int run, int subrun, int event ) {
    _run = run;
    _subrun = subrun;
    _event = event;
  }
  
  void LArFlowFlashMatch::saveAnaMatchData() {

    for (int iflash=0; iflash<_flashdata_v.size(); iflash++) {
      
      const FlashData_t&    flash = _flashdata_v[iflash];
      
      for (int iclust=0; iclust<_qcluster_v.size(); iclust++) {
	
	const QCluster_t& cluster = _qcluster_v[iclust];
	CutVars_t& cutvars = getCutVars(iflash,iclust);
    
	_cutfailed  = cutvars.cutfailed;
	if ( flash.truthmatched_clusteridx==iclust )
	  _truthmatch = 1;

	_isbeam     = (flash.isbeam) ? 1 : 0;
	_isneutrino = (cluster.isneutrino) ? 1 : 0;
	_intime     = (flash.intime) ? 1 : 0;
	
	_hypope     = 0;
	_datape     = 0;
	
	_dtick_window  = cutvars.dtick_window;
	_maxdist_wext  = cutvars.maxdist_wext;
	_maxdist_noext = cutvars.maxdist_noext;
	_maxdist_best  = (_maxdist_wext<_maxdist_noext) ? _maxdist_wext : _maxdist_noext;
    
	_peratio_wext  = cutvars.peratio_wext;
	_peratio_noext = cutvars.peratio_noext;
	_peratio_best  = (_peratio_wext<_peratio_noext) ? _peratio_wext : _peratio_noext;

	_enterlen      = cutvars.enterlen;
	_fmatch        = cutvars.fit1fmatch;

	// find truthmatch
	_fmatch_truth = -1.0;
	if ( flash.truthmatched_clusteridx>=0 ) {
	  MatchPair_t pair;
	  pair.flashidx   = iflash;
	  pair.clusteridx = flash.truthmatched_clusteridx;
	  auto it_pair = _pair2matchidx.find(pair);
	  if ( it_pair!=_pair2matchidx.end() ) {
	    _fmatch_truth = _fitter._fmatch_v.at(it_pair->second);
	  }
	}

	_anatree->Fill();
	
      }// end of cluster loop
    }//end of flash loop
  }
  
  void LArFlowFlashMatch::writeAnaFile() {
    if ( _save_ana_tree && !_anafile_written ) {
      _fanafile->cd();
      _anatree->Write();
      _anafile_written = true;
    }
  }

}
