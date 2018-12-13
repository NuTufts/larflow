#include "LArFlowFlashMatch.h"
#include <sstream>

// ROOT
#include "TCanvas.h"
#include "TH1D.h"
#include "TH1F.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2D.h"
#include "TEllipse.h"
#include "TBox.h"
#include "TGraph.h"
#include "TText.h"
#include "TStyle.h"
#include "TMarker.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"

// larcv
#include "larcv/core/DataFormat/ClusterMask.h"

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
      _input_lfcluster_v(nullptr),
      _nflashes(0),
      _nqclusters(0),
      _nelements(0),
      _compatibility_defined(false),
      _fMaxDistCut(0),
      _fPERatioCut(0),
      _fDumpPrefit(false),
      _fDumpPostfit(false),
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
      _flashtree(nullptr),      
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
    _fMaxDistCut  = 0.5;
    _fPERatioCut  = 3.0;
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
    clearFitter();
    clearAnaVariables();
    _evstatus = nullptr;
    _has_chstatus = false;
  }
  
  void LArFlowFlashMatch::match( const larlite::event_opflash& beam_flashes,
				 const larlite::event_opflash& cosmic_flashes,
				 const std::vector<larlite::larflowcluster>& clusters,
				 const std::vector<larcv::Image2D>& img_v,
				 const bool ignorelast ) {

    if ( clusters.size()==0 ) {
      std::cout << "[LArFlowFlashMatch::match] Nothing to do. No clusters!" << std::endl;
      return;
    }

    // save the input cluster pointer
    _input_lfcluster_v = &clusters;
    
    // first is to build the charge points for each cluster
    _qcluster_v.clear();
    // we ignore last cluster because sometimes we put store unclustered hits in that entry
    if ( ignorelast && clusters.size()>0 ) {
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

    // reduce matches by removing unique flash-cluster matches (no need to fit them)
    reduceRemoveUniqueMatches();
    printCompatSummary();
    
    std::cout << "[LArFlowFlashMatch::match][DEBUG] PREFIT COMPATIBILITY" << std::endl;
    printCompatInfo( _flashdata_v, _qcluster_v );

    _fitter.setUseBterms(true);
    prepareFitter();
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Fitter Loaded" << std::endl;
    if ( _fDumpPrefit )
      dumpQCompositeImages( "prefit" );
    if ( false )
      return; // for debug

    if ( _kDoTruthMatching && _kFlashMatchedDone ) {
      setFitParsWithTruthMatch();
    }
    
    _fitter.setLossFunction( LassoFlashMatch::kNLL );
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
    //std::cin.get();
    
    // setup learning schedule
    // two stages: first use SGD to minimize with noise
    LassoFlashMatch::LearningConfig_t epoch1;
    epoch1.iter_start =  0;
    epoch1.iter_end   =  30000;
    epoch1.lr = 1.0e-6;
    epoch1.use_sgd = true;
    epoch1.matchfrac = 0.5; // introduce noise
    LassoFlashMatch::LearningConfig_t epoch2;
    epoch2.iter_start =  30001;
    epoch2.iter_end   =  60000;
    epoch2.lr = 1.0e-5;
    epoch2.use_sgd = false;
    epoch2.matchfrac = 1.0; // introduce noise
    // last stage, setting to global min
    LassoFlashMatch::LearningConfig_t epoch3;
    epoch3.iter_start =  60001;
    epoch3.iter_end   = 120000;
    epoch3.lr = 1.0e-6;
    epoch3.use_sgd = false;
    epoch3.matchfrac = 1.0; // introduce noise
    _fitter.addLearningScheduleConfig( epoch1 );
    _fitter.addLearningScheduleConfig( epoch2 );
    _fitter.addLearningScheduleConfig( epoch3 );    

    LassoFlashMatch::LassoConfig_t lasso_cfg;
    lasso_cfg.minimizer = LassoFlashMatch::kCoordDescSubsample;
    //lasso_cfg.minimizer = LassoFlashMatch::kCoordDesc;
    lasso_cfg.match_l1        = 10.0;
    lasso_cfg.clustergroup_l2 = 1.0;    
    lasso_cfg.adjustpe_l2     = 1.0e-2;
    LassoFlashMatch::Result_t fitresult = _fitter.fitLASSO( lasso_cfg );
    saveFitterData( fitresult );

    // set compat from fit
    reduceUsingFitResults( 0.05 );
    std::cout << "----------------------------------------------------------" << std::endl;
    std::cout << "[LArFlowFlashMatch::match][DEBUG] POSTFIT COMPATIBILITY" << std::endl;
    printCompatInfo( _flashdata_v, _qcluster_v );
    _fitter.printState(false);      
    _fitter.printClusterGroups();
    _fitter.printFlashBundles( false );

    // build larflowclusters
    buildFinalClusters( fitresult, img_v );

    saveAnaMatchData();

    if ( _fDumpPostfit )
      dumpQCompositeImages( "postfit" );

    std::cout << "[ENTER] to continue" << std::endl;
    //std::cin.get();
    
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

    // build the qclusters -- ass<ociating 3D position and grabbing charge pixel value from 2D image
    const larcv::ImageMeta& src_meta = img_v[src_plane].meta();

    int nclusters = lfclusters.size();
    if ( ignorelast ) nclusters -= 1;
    for ( size_t icluster=0; icluster<nclusters; icluster++ ) {

      const larlite::larflowcluster& lfcluster = lfclusters[icluster];

      std::cout << "[LArFlowFlashMatch::buildInitialQClusters] import larflowcluster[" << icluster <<  "] "
		<< " numhits=" << lfcluster.size()
		<< std::endl;
      
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

	// debug
	//std::cout << "qhit: (" << qhit.tick << "," << qhit.xyz[0] << "," << qhit.xyz[1] << "," << qhit.xyz[2] << ")" << std::endl;
	
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
	qhit.pixeladc    = img_v[src_plane].pixel( row, col ); // hmm (non?)
	qhit.fromplaneid = src_plane;
	//std::cout << "debug: (r,c)=(" << row << "," << col << ") pixeladc=" << qhit.pixeladc << " fromplane=" << qhit.fromplaneid << std::endl;

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
      int maxid = -1;
      for (auto& it : mctrackid_counts ) {
	if ( maxcounts < it.second ) {
	  maxcounts = it.second;
	  maxid = it.first;
	}
      }
      qcluster.mctrackid = maxid;
      std::cout << "[LArFlowFlashMatch::buildInitialQClusters] qcluster[" << icluster <<  "] "
		<< " mctrackid=" << maxid << " (n pts with ID=" << maxcounts << ") "
		<< " numhits=" << qcluster.size()
		<< std::endl;
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

	if ( n==1 && flash.Time()>=0 && flash.Time()<=22.5 ) {
	  // cosmic disc within beam window, skip
	  continue;
	}
	
	FlashData_t newflash[4];

	for (int iwfm=0; iwfm<4; iwfm++) {
	  newflash[iwfm].resize( npmts, 0.0 );
	  newflash[iwfm].tot = 0.;
	  
	  float maxpmtpe = 0.;
	  int maxpmtch = 0;
	  
	  //int choffset = (n==1 && flash.nOpDets()>npmts) ? 200 : 0;
	  int choffset = iwfm*100;
	
	  for (size_t ich=0; ich<npmts; ich++) {
	    float pe = flash.PE( choffset + ich );
	    newflash[iwfm][ich] = pe;
	    newflash[iwfm].tot += pe;
	    if ( pe > maxpmtpe ) {
	      maxpmtpe = pe;
	      maxpmtch = ich;
	    }
	  }
	  newflash[iwfm].tpc_tick  = tpc_trigger_tick + flash.Time()/0.5;
	  newflash[iwfm].tpc_trigx = flash.Time()*driftv; // x-assuming x=0 occurs when t=trigger
	  newflash[iwfm].maxch     = maxpmtch;
	  newflash[iwfm].idx       = iflash;
	  newflash[iwfm].isbeam    = ( n==0 ) ? true : false;
	  if ( n==0 && flash.Time()>2.968 && flash.Time()<4.843 )
	    newflash[iwfm].intime = true;
	  else
	    newflash[iwfm].intime = false;
	
	  Double_t pmtpos[3];
	  geo->GetOpChannelPosition( maxpmtch, pmtpos );	
	  newflash[iwfm].maxchposz   = pmtpos[2];
	
	  // normalize
	  if (newflash[iwfm].tot>0) {
	    for (size_t ich=0; ich<npmts; ich++)
	      newflash[iwfm][ich] /= newflash[iwfm].tot;
	  }
	}


        int imaxwfm = 0;
	float maxwfm = 0;
	std::cout << "dataflash[" << iflash << "] "
		  << " intime=" << newflash[0].intime
		  << " tick=" << newflash[0].tpc_tick
		  << " totpe=" << newflash[0].tot
		  << " dT=" << flash.Time() << " us (intime=" << newflash[0].intime << ") ";
	std::cout << " PE[ ";
	for (int iwfm=0; iwfm<4; iwfm++) {
	  std::cout << newflash[iwfm].tot << " ";
	  if ( newflash[iwfm].tot>maxwfm ) {
	    maxwfm = newflash[iwfm].tot;
	    imaxwfm = iwfm;
	  }
	}
	std::cout << "]" << std::endl;
	
	flashdata.emplace_back( std::move(newflash[imaxwfm]) );	
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

	// calculate the x-bounds if the cluster is truly matched to this cluster
	float dtick_min = qcluster.min_tyz[0] - flash.tpc_tick;
	float dtick_max = qcluster.max_tyz[0] - flash.tpc_tick;
	float xpos_min = dtick_min*0.5*larp->DriftVelocity();
	float xpos_max = dtick_max*0.5*larp->DriftVelocity();
	
	CutVars_t& cutvar = getCutVars(iflash,iq);

	// must happen after (allow for some slop)
	if ( xpos_min < -10.0 ) {
	  cutvar.dtick_window = dtick_min;
	  cutvar.cutfailed = kWrongTime;
	  setCompat( iflash, iq, kWrongTime ); // too early
	}
	else if ( xpos_max > 256.0+10.0 ) {
	  cutvar.dtick_window = dtick_max;
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

	CutVars_t& cutvar = getCutVars( iflash, iq);

	// we only want to do this kind of analysis if the cut passes because of the
	// the use of extensions

	bool used_maxdist_wext = ( cutvar.maxdist_wext < cutvar.maxdist_noext );
	bool used_peratio_wext = ( cutvar.peratio_wext < cutvar.peratio_noext );

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
	float peratio  = ( fabs(cutvar.peratio_wext) < fabs(cutvar.peratio_noext) ) ? cutvar.peratio_wext : cutvar.peratio_noext;
	cutvar.pe_hypo = ( cutvar.maxdist_wext < cutvar.maxdist_noext ) ? hypo_wext.tot : hypo_noext.tot;
	cutvar.pe_data = flashdata.tot;

	if ( fabs(peratio) > _fPERatioCut ) {
	  cutvar.cutfailed = kFirstPERatio;
	  setCompat(iflash,iclust,kFirstPERatio);
	}
	
      }//end of cluster loop

    }//end of flash loop
    
  }

  void LArFlowFlashMatch::reduceRemoveUniqueMatches() {

    // std::
    // std::vector< std::vector<int> > flash_clusteridx;
    // std::vector< std::vector<int> > clust_flashidx;
    
    // for ( int imatch=0; imatch<_fitter._nmatches; imatch++ ) {
    //   int iflash = _matchidx2pair[imatch].flashidx;
    //   int iclust = _matchidx2pair[imatch].clusteridx;

    //   CutVars_t& cutvars = getCutVars( iflash, iclust );
    // }

    
  }
  
  void LArFlowFlashMatch::reduceUsingFitResults( const float score_threshold) {
    for ( int imatch=0; imatch<_fitter._nmatches; imatch++ ) {
      int iflash = _matchidx2pair[imatch].flashidx;
      int iclust = _matchidx2pair[imatch].clusteridx;

      CutVars_t& cutvars = getCutVars( iflash, iclust );
      cutvars.fit1fmatch = _fitter._fmatch_v[imatch];
      if ( cutvars.fit1fmatch<score_threshold ) {
	cutvars.cutfailed = kFirstFit;
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
      const FlashData_t& flash = flashdata_v[iflash];
      int ncompat = 0;
      std::vector<int> compatidx;
      for (int iclust=0; iclust<_nqclusters; iclust++) {
	if ( getCompat(iflash,iclust)==0 ) {
	  compatidx.push_back( iclust );
	  ncompat ++;
	}
      }
      std::cout << "flash[" << iflash << "] [Tot: " << ncompat << "] ";
      std::cout << "[mctrackid=" << flash.mctrackid << " ";
      if ( flash.tpc_visible )
	std::cout << " intpc ";
      if ( flash.img_visible==1 )
	std::cout << "truthclusteridx=" << flash.truthmatched_clusteridx << "] ";
      else
	std::cout << "not-img-vis] ";
      
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
      std::cout << "[LArFlowFlashMatch::prepareFitter()] after flash=" << iflash << std::endl;
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
    float imgedge_min = (2400-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
    float imgedge_max = (8448-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
    TBox box_imgedge_xy( imgedge_min, -116.5, imgedge_max, 116.5 );
    box_imgedge_xy.SetLineColor(kBlue);
    box_imgedge_xy.SetFillStyle(0);
    box_imgedge_xy.SetLineWidth(1);

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

    std::cout << "==========================================================" << std::endl;
    std::cout << "Dump QCompositeCluster DEBUG images" << std::endl;
    std::cout << "==========================================================" << std::endl;    
    std::cout << std::endl;
    
    // make charge graphs
    for (size_t iflash=0; iflash<_flashdata_v.size(); iflash++) {

      std::cout << "[FLASH " << iflash << "]" << std::endl;
      std::cout << "-----------------------------------------------" << std::endl;
      
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
      TH1F hflashdata(histname_data,"",32,0,32);
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
      std::cout << "  [data] total pe: " << hflashdata.Integral() << std::endl;

      // truth-matched track for flash
      std::vector<TGraph*> mctrack_data_zy;
      std::vector<TGraph*> mctrack_data_xy;
      std::vector<TMarker*> mcstart_zy;
      std::vector<TMarker*> mcstart_xy;
      TBox* mctrack_win_xy = nullptr;
      bool mctrack_match = false;
      if ( _flash_matched_mctracks_v[iflash].size()>0 || _flash_matched_mcshowers_v[iflash].size()>0 ) {
	if ( flash.mctrackid>=0 ) mctrack_match = true;      

	// we make a graph for each shower and track
	for ( auto const& pmct : _flash_matched_mctracks_v[iflash] ) {

	  TMarker* m_zy = new TMarker( pmct->Start().Z(), pmct->Start().Y(), 30 );
	  TMarker* m_xy = new TMarker( pmct->Start().X(), pmct->Start().Y(), 30 );
	  mcstart_zy.push_back( m_zy );
	  mcstart_xy.push_back( m_xy );
	  //std::cout << " [flash " << iflash << "] track steps = " << pmct->size() << std::endl;
	  if ( pmct->size()==0 ) continue;
	      
	  TGraph* gmc_zy = new TGraph( pmct->size() );
	  TGraph* gmc_xy = new TGraph( pmct->size() );
	  for (int istep=0; istep<(int)pmct->size(); istep++) {
	    gmc_zy->SetPoint(istep, (*pmct)[istep].Z(), (*pmct)[istep].Y() );
	    gmc_xy->SetPoint(istep, (*pmct)[istep].X(), (*pmct)[istep].Y() );
	  }
	  gmc_zy->SetLineColor(kBlue);
	  gmc_zy->SetLineWidth(1);
	  gmc_xy->SetLineColor(kBlue);
	  gmc_xy->SetLineWidth(1);
	  mctrack_data_zy.push_back( gmc_zy );
	  mctrack_data_xy.push_back( gmc_xy );
	}

	for ( auto const& pshr : _flash_matched_mcshowers_v[iflash] ) {
	  
	  TMarker* m_zy = new TMarker( pshr->Start().Z(), pshr->Start().Y(), 30 );
	  TMarker* m_xy = new TMarker( pshr->Start().X(), pshr->Start().Y(), 30 );
	  mcstart_zy.push_back( m_zy );
	  mcstart_xy.push_back( m_xy );
	  
	  TGraph* gmc_zy = new TGraph( 2 ); // we draw a triangle
	  TGraph* gmc_xy = new TGraph( 2 );
	  gmc_zy->SetPoint(0, pshr->Start().Z(), pshr->Start().Y() );
	  gmc_xy->SetPoint(0, pshr->Start().X(), pshr->Start().Y() );

	  
	  gmc_zy->SetPoint(1, pshr->Start().Z() + 14.0*pshr->StartDir().Z(), pshr->Start().Y() + 14.0*pshr->StartDir().Y() );
	  gmc_xy->SetPoint(1, pshr->Start().X() + 14.0*pshr->StartDir().X(), pshr->Start().Y() + 14.0*pshr->StartDir().Y() );
	  
	  gmc_zy->SetLineColor(kRed);
	  gmc_zy->SetLineWidth(1);
	  gmc_xy->SetLineColor(kRed);
	  gmc_xy->SetLineWidth(1);
	  mctrack_data_zy.push_back( gmc_zy );
	  mctrack_data_xy.push_back( gmc_xy );
	}
	
	
	mctrack_win_xy = new TBox( xoffset, -116.5, xoffset+256, 116.5 );
	mctrack_win_xy->SetLineColor(kRed);
	mctrack_win_xy->SetLineWidth(1);
	mctrack_win_xy->SetFillStyle(0);
      }

      // BEGIN CLUSTER LOOP
      std::vector< TGraph* > mctrack_clustertruth_xy_v;
      std::vector< TGraph* > mctrack_clustertruth_zy_v;
      std::vector< TH1F* > flash_hypotheses_v;
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
	std::vector< TH1F* > hypo_v(2, nullptr);
	char histname1[100];
	sprintf(histname1, "hhypo_flash%d_clust%d_wext",  (int)iflash, (int)iclust );
	hypo_v[0] = new TH1F( histname1, "", 32, 0, 32 );
	char histname2[100];
	sprintf(histname2, "hhypo_flash%d_clust%d_noext", (int)iflash, (int)iclust );
	hypo_v[1] = new TH1F( histname2, "", 32, 0, 32 );
	std::vector<TGraph> gtrack_v = qcomposite.getTGraphsAndHypotheses( flash, hypo_v );
	std::cout << "  [cluster " << iclust << "] hypo w/ext"  << histname1 << "=" << hypo_v[0]->Integral() << "  maxdist=" << cutvars.maxdist_wext << std::endl;
	std::cout << "  [cluster " << iclust << "] hypo no/ext" << histname2 << "=" << hypo_v[1]->Integral() << "  maxdist=" << cutvars.maxdist_noext << std::endl;


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

	TH1F* hhypo = nullptr;

	
	if ( cutvars.maxdist_wext < cutvars.maxdist_noext ) {
	  hhypo = hypo_v[0];
	  delete hypo_v[1];
	}
	else {
	  hhypo = hypo_v[1];
	  delete hypo_v[0];
	}
		  
	if ( cutvars.maxdist_wext > cutvars.maxdist_noext )
	  hhypo->SetLineStyle( 2 );
	if ( iclust==flash.truthmatched_clusteridx ) {
	  hhypo->SetLineColor(kGreen+2);
	  hhypo->SetLineWidth( 2 );
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
	if ( graphs_zy_v[ig].GetN()==0 ) continue;
	graphs_zy_v[ig].Draw("P");
      }	
      
      for (int ich=0; ich<32; ich++)
	chmarkers_v[ich]->Draw();

      for ( auto const& m : mcstart_zy )
	m->Draw();
      for ( auto const& mct : mctrack_data_zy )
	mct->Draw("L");
      
      // ---------------------------------------------
      // TOP XY-2D PLOT: FLASH-DATA/FLASH TRUTH TRACK
      // ---------------------------------------------
      dataxy.cd();
      bgxy.Draw();
      box_imgedge_xy.Draw();
      boxxy.Draw();
      
      for (size_t ig=0; ig<graphs_xy_v.size(); ig++ ) {
	if ( graphs_xy_v[ig].GetN()==0 ) continue;	
	graphs_xy_v[ig].Draw("P");
      }

      for ( auto const& m : mcstart_xy )
	m->Draw();      
      for ( auto const& mct : mctrack_data_xy )
	mct->Draw("L");
      if ( mctrack_win_xy ) {
	mctrack_win_xy->Draw();
      }
      
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

      //std::cout << "number of clusters draw: " << nclusters_drawn << std::endl;
      //std::cout << "graph_zy: " << graphs_zy_v.size() << std::endl;
      //std::cout << "graph_xy: " << graphs_xy_v.size() << std::endl;            
      std::cout << "[enter to continue]" << std::endl;
      //std::cin.get();

      for (int ich=0; ich<32; ich++) {
	delete datamarkers_v[ich];
      }

      for ( auto& pmct : mctrack_data_zy )
	delete pmct;
      for ( auto& pmct : mctrack_data_xy )
	delete pmct;
      for ( auto& pm : mcstart_zy )
	delete pm;
      for ( auto& pm : mcstart_xy )
	delete pm;

      for ( auto& ptrack : mctrack_clustertruth_zy_v )
	delete ptrack;
      for ( auto& ptrack : mctrack_clustertruth_xy_v )
	delete ptrack;
      for ( auto& phist : flash_hypotheses_v )
	delete phist;

      if ( mctrack_win_xy )
	delete mctrack_win_xy;
      
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
  
  // ==============================================================================================
  // MC TRUTH FUNCTIONS
  // ==============================================================================================
  
  void LArFlowFlashMatch::loadMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v,
					   const std::vector<larlite::mcshower>& mcshower_v,
					   bool do_truth_matching ) {
    _mctrack_v = &mctrack_v;
    _mcshower_v = &mcshower_v;    
    _kDoTruthMatching = do_truth_matching;
    _kFlashMatchedDone = false;
    std::cout << "[LArFlowFlashMatch::loadMCTrackInfo][INFO] Loaded MC tracks and showers." << std::endl;
  }

  void LArFlowFlashMatch::doFlash2MCTrackMatching( std::vector<FlashData_t>& flashdata_v ) {

    //space charge and time service; ideally initialized in algo constructor
    if ( _psce==nullptr ){
      _psce = new ::larutil::SpaceChargeMicroBooNE;
    }
    // note on TimeService: if not initialized from root file via GetME(true)
    // needs trig_offset hack in tufts_larflow branch head
    const ::larutil::TimeService* tsv = ::larutil::TimeService::GetME(false);
    const float cm_per_tick = larutil::LArProperties::GetME()->DriftVelocity()*0.5;

    _flash_matched_mctracks_v.clear();
    _flash_matched_mcshowers_v.clear();
    _flash_matched_mctracks_v.resize(  flashdata_v.size() );
    _flash_matched_mcshowers_v.resize( flashdata_v.size() );

    std::vector< std::vector<int> >   track_id_match_v(flashdata_v.size());
    std::vector< std::vector<int> >   track_pdg_match_v(flashdata_v.size());
    std::vector< std::vector<int> >   track_mid_match_v(flashdata_v.size());
    std::vector< std::vector<float> > track_E_match_v(flashdata_v.size());
    std::vector< std::vector<float> > track_dz_match_v(flashdata_v.size());
    
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
	  _flash_matched_mctracks_v[iflash].push_back( &mct );
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

    // Match showers
    int imcshower = 0;
    for (auto& mcs : *_mcshower_v ) {
      imcshower++;      

      if ( mcs.Origin()==1 )
	_nu_mctrackid.insert( mcs.TrackID() );
      
      // get time
      //float track_tick = tsv->TPCG4Time2Tick( mct.Start().T() );
      float track_tick = mcs.Start().T()*1.0e-3/0.5 + 3200;
      float flash_time = tsv->OpticalG4Time2TDC( mcs.Start().T() );
      
      int nmatch = 0;
      int   best_flashidx =   -1;      
      float best_dtick    = 1e9;
      for ( size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
	float dtick = fabs(flashdata_v[iflash].tpc_tick - track_tick);
	if ( dtick < 10 ) { 
	  _flash_matched_mcshowers_v[iflash].push_back( &mcs );
	}
      }
    }//end of shower loop
    std::cout << "[LArFlowFlashMatch::doFlash2MCTrackMatching] number of neutrin track IDs: " << _nu_mctrackid.size() << std::endl;

    std::cout << "====================================" << std::endl;
    std::cout << "MCTRACK/MCSHOWER MATCHES" << std::endl;
    std::cout << "------------------------" << std::endl;    
    for (size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
      
      std::cout << "[Flash " <<  iflash << " matches]  flash_tick=" << flashdata_v[iflash].tpc_tick << std::endl;

      FlashData_t& flash = flashdata_v[iflash];
      // set some truth-defaults
      flash.isneutrino = false;
      flash.tpc_visible = 1;
      flash.img_visible = 1;      
      flash.truthmatched_clusteridx = -1;
      flash.mctrackid = -1;

      int nvis_steps = 0;
      int ntpc_steps = 0;
      for ( auto const& ptrk : _flash_matched_mctracks_v[iflash] ) {
	float track_tick = ptrk->Start().T()*1.0e-3/0.5 + 3200;
	float dtick      = fabs(flashdata_v[iflash].tpc_tick - track_tick);
	if ( ptrk->Origin()==1 )
	  flash.isneutrino = true;
	for ( auto const& step : *ptrk ) {
	  float tick = step.T()*1.0e-3/0.5+3200 + step.X()/cm_per_tick;
	  if ( tick>=2400 && tick<8448 )
	    nvis_steps += 1;
	}
	ntpc_steps += ptrk->size();
	
	std::cout << "  mctrack[" << ptrk->TrackID() << ",origin=" << ptrk->Origin() << "] pdg=" << ptrk->PdgCode()
		  << " ancestor=" << ptrk->AncestorTrackID()
		  << " E=" << ptrk->Start().E() 
		  << " T()=" << ptrk->Start().T()
		  << " Start()=(" << ptrk->Start().X() << "," << ptrk->Start().Y() << "," << ptrk->Start().Z() << ")"
		  << " tracktick=" << track_tick
		  << " dtick=" << dtick
		  << std::endl;
	if ( flash.mctrackid<0 ) {
	  flash.mctrackid  = ptrk->AncestorTrackID();
	  flash.mctrackpdg = ptrk->PdgCode();
	}
      }
      
      for ( auto const& pshr : _flash_matched_mcshowers_v[iflash] ) {
	float track_tick = pshr->Start().T()*1.0e-3/0.5 + 3200;
	float dtick      = fabs(flashdata_v[iflash].tpc_tick - track_tick);
	if ( pshr->Origin()==1 )
	  flash.isneutrino = true;
	std::cout << "  mcshower[" << pshr->TrackID() << ",origin=" << pshr->Origin() << "] pdg=" << pshr->PdgCode()
		  << " ancestor=" << pshr->AncestorTrackID()	  
		  << " E=" << pshr->Start().E() 
		  << " T()=" << pshr->Start().T()
		  << " Start()=(" << pshr->Start().X() << "," << pshr->Start().Y() << "," << pshr->Start().Z() << ")"	  
		  << " tracktick=" << track_tick
		  << " dtick=" << dtick	  
		  << std::endl;
	float start_tick =  pshr->Start().T()*1.0e-3/0.5 + 3200 + pshr->Start().X()/cm_per_tick;
	if ( pshr->Start().E()>10.0 // arbitrary 10 MeV limit
	     && pshr->Start().X()>0 && pshr->Start().X()<256
	     && fabs(pshr->Start().Y())<116.0
	     && pshr->Start().Z()>0 && pshr->Start().Z()<1036.0 )  {
	  ntpc_steps += 1;
	  if ( start_tick>=2400 && start_tick<8448 )
	    nvis_steps += 1;
	}
	if ( flash.mctrackid<0 ) {
	  flash.mctrackid = pshr->AncestorTrackID();
	  flash.mctrackpdg = pshr->PdgCode();
	}
      }
      
      // if no visible steps
      if ( nvis_steps==0 ) {
	// not in the tpc, so nothing to match
	flash.img_visible = 0;
      }
      else {
	flash.img_visible = 1;
      }

      if ( ntpc_steps==0 )
	flash.tpc_visible = 0;
      else
	flash.tpc_visible = 1;

      // determinations
      std::cout << "  flash-is-img-visible:  " << flash.img_visible << std::endl;
      std::cout << "  flash-is-tpc-visible:  " << flash.tpc_visible << std::endl;      
      std::cout << "  flash-is-neutrino: " << flash.isneutrino << std::endl;
      std::cout << "  flash-mctrackid:   " << flash.mctrackid << std::endl;    

    }
    std::cout << "====================================" << std::endl;
    
    // now loop over flashes
    // [Ithink this is deprecated]
    // for (size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
    //   std::vector<int>& id = track_id_match_v[iflash];
    //   std::vector<int>& pdg = track_pdg_match_v[iflash];
    //   std::vector<int>& mid = track_mid_match_v[iflash];
    //   std::vector<float>& dz = track_dz_match_v[iflash];      

    //   if ( id.size()==1 ) {
    // 	// easy!!
    // 	flashdata_v[iflash].mctrackid  = id.front();
    // 	flashdata_v[iflash].mctrackpdg = pdg.front();
    // 	if ( _nu_mctrackid.find( id[0] )!=_nu_mctrackid.end()  )
    // 	  flashdata_v[iflash].isneutrino = true;
	  
    //   }
    //   else if (id.size()>1 ) {
    // 	// to resolve multiple options.
    // 	// (1) favor id==mid && pdg=|13| (muon) -- from these, pick best time
    // 	int nmatches = 0;
    // 	int idx = -1;
    // 	int pdgx = -1;
    // 	float closestz = 10000;
    // 	bool isnu = false;	
    // 	for (int i=0; i<(int)id.size(); i++) {
    // 	  bool trackisnu = (_nu_mctrackid.find(id[i])!=_nu_mctrackid.end());
    // 	  //std::cout << "  multiple-truthmatches[" << i << "] id=" << id[i] << " mid=" << mid[i] << " pdg=" << pdg[i] << " dz=" << dz[i] << " isnu=" << trackisnu << std::endl;
    // 	  if ( (id[i]==mid[i] && dz[i]<closestz) || (trackisnu && flashdata_v[iflash].intime) ) {
    // 	    idx = id[i];
    // 	    pdgx = pdg[i];
    // 	    closestz = dz[i];
    // 	    //nmatches++;
    // 	    if ( trackisnu )
    // 	      isnu = true;
    // 	  }
    // 	}
    // 	flashdata_v[iflash].mctrackid = idx;
    // 	flashdata_v[iflash].mctrackpdg = pdgx;
    // 	flashdata_v[iflash].isneutrino = isnu;
	
    //   }// if multipl matched ids
    //   int nmcpts = (*_mctrack_v)[ _mctrackid2index[flashdata_v[iflash].mctrackid] ].size();
    //   std::cout << "FlashMCtrackMatch[" << iflash << "] "
    //   		<< "tick=" << flashdata_v[iflash].tpc_tick << " "
    //   		<< "nmatches=" << id.size() << " "
    //   		<< "trackid=" << flashdata_v[iflash].mctrackid << " "
    //   		<< "pdg=" << flashdata_v[iflash].mctrackpdg << " "
    //   		<< "isnu=" << flashdata_v[iflash].isneutrino << " "
    //   		<< "intime=" << flashdata_v[iflash].intime << " "
    //   		<< "isbeam=" << flashdata_v[iflash].isbeam << " "			
    //   		<< "nmcpts=" << nmcpts << std::endl;
    // }//end of flash loop
    _kFlashMatchedDone = true;
  }

  void LArFlowFlashMatch::doTruthCluster2FlashTruthMatching( std::vector<FlashData_t>& flashdata_v, std::vector<QCluster_t>& qcluster_v ) {
    for (int iflash=0; iflash<(int)flashdata_v.size(); iflash++) {
      FlashData_t& flash = flashdata_v[iflash];
      
      if ( flash.tpc_visible==0 ) continue;
      
      for (int iclust=0; iclust<(int)qcluster_v.size(); iclust++) {
	QCluster_t& cluster = qcluster_v[iclust];
	bool isnu = _nu_mctrackid.find( cluster.mctrackid )!=_nu_mctrackid.end();

	if ( !flash.isneutrino ) {
	  if ( flash.mctrackid!=-1 && flash.mctrackid==cluster.mctrackid ) {
	    flash.truthmatched_clusteridx = iclust;
	    cluster.truthmatched_flashidx = iflash;
	    cluster.isneutrino = flash.isneutrino;
	  }
	}
	else {
	  // flash is neutrino
	  if ( isnu ) {
	    // so is cluster
	    flash.mctrackid = cluster.mctrackid;
	    flash.truthmatched_clusteridx = iclust;
	    cluster.truthmatched_flashidx = iflash;
	    cluster.isneutrino = flash.isneutrino;
	  }
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
    _mctrack_v  = nullptr;
    _mcshower_v = nullptr;
    _mctrackid2index.clear();
    _nu_mctrackid.clear();
    _flash_truthid.clear();
    _cluster_truthid.clear();
    _flash2truecluster.clear();
    _cluster2trueflash.clear();
    _flash_matched_mctracks_v.clear();
    _flash_matched_mcshowers_v.clear();
    delete _psce;
    _psce = nullptr;
    _kDoTruthMatching  = false;
    _kFlashMatchedDone = false;
  }

  std::vector<larlite::larflowcluster> LArFlowFlashMatch::exportMatchedClusters() {
    // for each cluster, we use the best matching flash
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float  usec_per_tick = 0.5; // usec per tick
    const float  tpc_trigger_tick = 3200;
    const float  driftv = larp->DriftVelocity();
    
    std::vector<larlite::larflowcluster> lfcluster_v(_qcluster_v.size());
    
    // for (int iclust=0; iclust<_qcluster_v.size(); iclust++) {

    //   // did we fit this?
      
    //   const QCluster_t& orig_cluster = _qcluster_v[iclust];
    //   const QCluster_t& comp_cluster = _qcomposite_v[iclust];
      

      

    //   float matched_flash_tick = 0;
    //   float matched_flash_xoffset = 0;
    //   if ( maxfmatch_idx>=0 ) {
    // 	matched_flash_tick    = _flashdata_v[maxfmatch_idx].tpc_tick;
    // 	matched_flash_xoffset = (matched_flash_tick-tpc_trigger_tick)*usec_per_tick/driftv;
    //   }
      
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
      CutVars_t& cutvars = getCutVars(iflash,iclust);
      
      const FlashData_t& flash = _flashdata_v[iflash];
      if ( flash.img_visible==1 && flash.truthmatched_clusteridx==iclust ) {
	fmatch_truth[imatch] = 1;
	cutvars.truthfmatch = 1;	
      }
      else {
	fmatch_truth[imatch] = 0.;
	cutvars.truthfmatch = 0;
      }
      
    }

    _fitter.setFMatch( fmatch_truth );

    // what am i trying to measure here?
    // for ( auto& it_match : _matchidx2pair ) {
      
    //   int imatch = it_match.first;
    //   int iflash = it_match.second.flashidx;
    //   int iclust = it_match.second.clusteridx;

    //   CutVars_t& cutvars = getCutVars(iflash,iclust);
      
    //   cutvars.truthfmatch = _fitter.scoreMatch( imatch );

    // }

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
    _anatree  = new TTree("matchana", "Variables for each flash-cluster candidate match");
    _anatree->Branch("run",              &_run,              "run/I");
    _anatree->Branch("subrun",           &_subrun,           "subrun/I");
    _anatree->Branch("event",            &_event,            "event/I");
    _anatree->Branch("flash_isneutrino", &_flash_isneutrino, "flash_isneutrino/I");        
    _anatree->Branch("flash_intime",     &_flash_intime,      "flash_intime/I");        
    _anatree->Branch("flash_isbeam",     &_flash_isbeam,      "flash_isbeam/I");
    _anatree->Branch("flash_isvisible",  &_flash_isvisible,   "flash_isvisible/I");
    _anatree->Branch("flash_crosstpc",   &_flash_crosstpc,    "flash_crosstpc/I");    
    _anatree->Branch("flash_mcid",       &_flash_mcid,        "flash_mcid/I");
    _anatree->Branch("flash_tick",       &_flash_tick,        "flash_tick/F");    
    
    _anatree->Branch("truthmatched",     &_truthmatched,      "truthmatched/I");
    _anatree->Branch("cutfailed",        &_cutfailed,         "cutfailed/I");    
    _anatree->Branch("hypope",           &_hypope,            "hypope/F");
    _anatree->Branch("datape",           &_datape,            "datape/F");
    _anatree->Branch("dtickwin",         &_dtick_window,      "dtickwin/F");
    _anatree->Branch("maxdist",          &_maxdist_best,      "maxdist/F");
    _anatree->Branch("maxdist_wext",     &_maxdist_wext,      "maxdist_wext/F");
    _anatree->Branch("maxdist_noext",    &_maxdist_noext,     "maxdist_noext/F");
    _anatree->Branch("peratio",          &_peratio_best,      "peratio/F");
    _anatree->Branch("peratio_wext",     &_peratio_wext,      "peratio_wext/F");
    _anatree->Branch("peratio_noext",    &_peratio_noext,     "peratio_noext/F");
    _anatree->Branch("enterlen",         &_enterlen,          "enterlen/F");
    _anatree->Branch("fmatch",           &_fmatch_singlefit,  "fmatch1/F");
    _anatree->Branch("fmatch1_truth",    &_fmatch_truth,      "fmatch1_truth/F");
    _anatree->Branch("fmatchm_frac1",    &_fmatch_multifit_fracabove1, "fmatchm_frac1/F");
    _anatree->Branch("fmatchm_frac2",    &_fmatch_multifit_fracabove2, "fmatchm_frac2/F");
    _anatree->Branch("fmatchm_mean",     &_fmatch_multifit_mean,       "fmatchm_mean/F");            
    _anatree->Branch("fmatchm_nsamples", &_fmatch_multifit_nsamples,   "fmatchm_nsamples/I");


    _flashtree = new TTree("anaflashtree", "Variables for each flash");
    _flashtree->Branch("run",    &_run,    "run/I");
    _flashtree->Branch("subrun", &_subrun, "subrun/I");
    _flashtree->Branch("event",  &_event,  "event/I");
    _flashtree->Branch("flash_isneutrino",  &_flash_isneutrino,  "flash_isneutrino/I");        
    _flashtree->Branch("flash_intime",      &_flash_intime,      "flash_intime/I");        
    _flashtree->Branch("flash_isbeam",      &_flash_isbeam,      "flash_isbeam/I");
    _flashtree->Branch("flash_isvisible",   &_flash_isvisible,   "flash_isvisible/I");
    _flashtree->Branch("flash_crosstpc",    &_flash_crosstpc,    "flash_crosstpc/I");    
    _flashtree->Branch("flash_mcid",        &_flash_mcid,        "flash_mcid/I");
    _flashtree->Branch("flash_bestclustidx",&_flash_bestclustidx,"flash_bestclustidx/I");
    _flashtree->Branch("flash_bestfmatch",  &_flash_bestfmatch,  "flash_bestfmatch/F");
    _flashtree->Branch("flash_truthfmatch", &_flash_truthfmatch, "flash_truthfmatch/F");
    _flashtree->Branch("flash_tick",        &_flash_tick,        "flash_tick/F");

    
    _save_ana_tree   = true;
    _anafile_written = false;
  }

  void LArFlowFlashMatch::clearAnaVariables() {

    _flash_mcid       = -1;
    _flash_isvisible  = -1;
    _flash_isneutrino = 0;
    _flash_intime     = 0;
    _flash_isbeam     = 0;

    _cutfailed    = -1;
    _truthmatched = -1;    
    _hypope       = 0;
    _datape       = 0;
    _dtick_window = 0;
    _maxdist_best = -1;
    _maxdist_wext = -1;
    _maxdist_noext = -1;
    _peratio_best = 0;
    _peratio_wext = 0;
    _peratio_noext = 0;
    _enterlen = -1;
    _fmatch_singlefit = 0;
    _fmatch_multifit_fracabove1 = 0;
    _fmatch_multifit_fracabove2 = 0;
    _fmatch_multifit_nsamples = 0;
    _fmatch_multifit_mean = 0;    
    _fmatch_truth=0;
  }

  void LArFlowFlashMatch::setRSE( int run, int subrun, int event ) {
    _run = run;
    _subrun = subrun;
    _event = event;
  }

  void LArFlowFlashMatch::saveFitterData( const LassoFlashMatch::Result_t& fitdata ) {

    for ( size_t imatch=0; imatch<fitdata.nmatches; imatch++ ) {

      int flashidx   = _fitter.userFlashIndexFromMatchIndex( imatch );
      int clusteridx = _fitter.userClusterIndexFromMatchIndex( imatch );

      if ( getCompat( flashidx, clusteridx )!=kUncut )
	continue;
      
      const FlashData_t& flashdata        = _flashdata_v[flashidx];
      const QClusterComposite& qcomposite = _qcomposite_v[clusteridx];
      
      //float xoffset = (flashdata.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
      CutVars_t& cutvar = getCutVars( flashidx, clusteridx );

      // store simple fit score
      cutvar.fit1fmatch = fitdata.beta[imatch];

      // store value of x with CDF>0.9 CDF>0.95
      // get the total
      int nsamples_tot = 0.;
      for ( int i=0; i<100; i++ ) {
	nsamples_tot += fitdata.subsamplebeta[imatch][i];
      }

      if ( nsamples_tot>0 ) {
      
	float cdf90 = 0; // > 0.90
	float cdf95 = 0; // > 0.95
	float cdf   = 0;
	for ( int i=0; i<100; i++ ) {
	  cdf += fitdata.subsamplebeta[imatch][i];
	  if ( cdf90==0 && (cdf/float(nsamples_tot))>0.90 ) cdf90 = i*0.01;
	  if ( cdf95==0 && (cdf/float(nsamples_tot))>0.95 ) cdf95 = i*0.01;
	  if ( cdf95>0 && cdf90>0 )
	    break;
	}
	cutvar.subsamplefit_fracabove1 = cdf90;
	cutvar.subsamplefit_fracabove2 = cdf95;
      }
      else {
	cutvar.subsamplefit_fracabove1 = 1.;
	cutvar.subsamplefit_fracabove2 = 1.;
      }
      cutvar.nsubsample_converged = nsamples_tot;
      cutvar.subsamplefit_mean = fitdata.subsamplebeta_mean[imatch];
      
    }//end of match loop
  }
  
  void LArFlowFlashMatch::saveAnaMatchData( ) {

    for (int iflash=0; iflash<_flashdata_v.size(); iflash++) {
      
      const FlashData_t&    flash = _flashdata_v[iflash];

      _flash_isbeam     = (flash.isbeam)     ? 1 : 0;
      _flash_isneutrino = (flash.isneutrino) ? 1 : 0;
      _flash_intime     = (flash.intime) ? 1 : 0;
      _flash_isvisible  = flash.img_visible;
      _flash_crosstpc   = flash.tpc_visible;
      _flash_mcid       = flash.mctrackid;
      _flash_tick       = flash.tpc_tick;

      _flash_bestclustidx = -1;
      _flash_bestfmatch = -1;      
      _flash_truthfmatch   = -1; 
	
      for (int iclust=0; iclust<_qcluster_v.size(); iclust++) {
	
	const QCluster_t& cluster = _qcluster_v[iclust];
	CutVars_t& cutvars = getCutVars(iflash,iclust);
    
	_cutfailed  = cutvars.cutfailed;
	if ( flash.truthmatched_clusteridx>=0 && flash.truthmatched_clusteridx==iclust )
	  _truthmatched = 1;
	else
	  _truthmatched = 0;
	
	_hypope     = cutvars.pe_hypo;
	_datape     = cutvars.pe_data;
	
	_dtick_window  = cutvars.dtick_window;
	_maxdist_wext  = cutvars.maxdist_wext;
	_maxdist_noext = cutvars.maxdist_noext;
	_maxdist_best  = (_maxdist_wext<_maxdist_noext) ? _maxdist_wext : _maxdist_noext;
    
	_peratio_wext  = cutvars.peratio_wext;
	_peratio_noext = cutvars.peratio_noext;
	_peratio_best  = (fabs(_peratio_wext)<fabs(_peratio_noext)) ? _peratio_wext : _peratio_noext;

	_enterlen      = cutvars.enterlen;

	// fit results
	_fmatch_singlefit           = cutvars.fit1fmatch;
	_fmatch_multifit_fracabove1 = cutvars.subsamplefit_fracabove1;
	_fmatch_multifit_fracabove2 = cutvars.subsamplefit_fracabove2;
	_fmatch_multifit_mean       = cutvars.subsamplefit_mean;
	_fmatch_multifit_nsamples   = cutvars.nsubsample_converged;

	if ( _truthmatched==1 )
	  _flash_truthfmatch = _fmatch_singlefit;

	if ( _fmatch_singlefit>=0 && _flash_bestfmatch<_fmatch_singlefit ) {
	  _flash_bestfmatch = _fmatch_singlefit;
	  _flash_bestclustidx = iclust;
	}
	
	// find actual truthmatch (to compare)
	_fmatch_truth = -1.0;
	if ( flash.truthmatched_clusteridx>=0 ) {
	  int flashidx   = iflash;
	  int clusteridx = flash.truthmatched_clusteridx;
	  _fmatch_truth = getCutVars( flashidx, clusteridx ).fit1fmatch;
	}
	
	_anatree->Fill();
	
      }// end of cluster loop

      _flashtree->Fill();      
    }//end of flash loop
  }
  
  void LArFlowFlashMatch::writeAnaFile() {
    if ( _save_ana_tree && !_anafile_written ) {
      _fanafile->cd();
      _anatree->Write();
      _flashtree->Write();            
      _anafile_written = true;
    }
  }

  void LArFlowFlashMatch::clearFinalClusters() {
    _final_lfcluster_v.clear();
    _final_clustermask_v.clear();
    _intime_lfcluster_v.clear();
    _intime_clustermask_v.clear();
  }

  void LArFlowFlashMatch::buildFinalClusters( LassoFlashMatch::Result_t& fitresult,
					      const std::vector<larcv::Image2D>& img_v ) {

    std::cout << "[LArFlowFlashMatch::buildFinalClusters] start." << std::endl;
    
    // we take the solutions stored in the fitter+cutvars
    // we assign a single flash to each cluster, using the best fit
    // we then combine clusters matched into the same flash
    //
    // we treat the intime-flashmatched clusters different, but not merging them
    // and accepting all that pass some threshold

    clearFinalClusters();
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float  usec_per_tick = 0.5; // usec per tick
    const float  tpc_trigger_tick = 3200;
    const float  driftv = larp->DriftVelocity();
    
    std::vector<int> flashidx_matched_to_cluster( _fitter.numClusterGroups() );
    for ( int icluster=0; icluster<_fitter.numClusterGroups(); icluster++ ) {
      const std::vector<int>& fitmatchindices = _fitter.matchIndicesFromInternalClusterIndex( icluster );
      int bestmatchidx = -1;
      int bestflashidx = -1;      
      float maxmatchscore = 0;
      
      int thresholdlevel = 0;
      // 0 = use subsamplefit_fracabove2 (>0.1 score)
      // 1 = use subsamplefit_fracabove2 (>0.05 score)
      // 2 = anything above zero

      while ( bestmatchidx<0 && thresholdlevel<3 ) {

	for ( size_t m : fitmatchindices ) {
	  
	  int flashidx   = _fitter.userFlashIndexFromMatchIndex( m );
	  int clusteridx = _fitter.userClusterIndexFromMatchIndex( m );
	  
	  // retrieve fraction
	  CutVars_t& cutvar = getCutVars( flashidx, clusteridx );
	  
	  // we ignore clusters with poor matching, judged by low ensemble scores
	  float var = 0;
	  switch ( thresholdlevel ) {
	  case 0:
	    var = cutvar.subsamplefit_fracabove2; // fmatch where cdf>0.9
	    break;
	  case 1:
	    var = cutvar.subsamplefit_fracabove1;
	    break;
	  case 2:
	    var = cutvar.fit1fmatch;
	    break;
	  }
	  
	  if ( var > 0.1 ) {
	    if ( var>maxmatchscore ) {
	      bestmatchidx  = m;
	      maxmatchscore = var;
	      bestflashidx  = flashidx;
	    }
	  }
	}//end of match loop
	thresholdlevel++;
      }//end of while loop
      
      flashidx_matched_to_cluster[icluster] = bestflashidx;
      
    }//end of cluster group list

    // find the clusters
    std::vector< std::vector<int> > group_indices;
    std::vector< std::vector<int> > group_matchindex;
    std::vector< int > group_flashidx;
    std::map< int, int > matchedflash_index; // flash index to group entry in above vector
    for ( int icluster=0; icluster<_fitter.numClusterGroups(); icluster++ ) {
      int flashidx = flashidx_matched_to_cluster[icluster];
      if ( flashidx<0 ) {
	// no match, we simply pass the cluster through
	std::vector<int> group( 1, _fitter.userClusterIndexFromInternal(icluster) );
	group_indices.push_back( std::move(group) );
	group_flashidx.push_back( -1 );
	continue;
      }
      
      // if passes, we had a flash match, we check to see if
      // a group for that is already defined. else we create a new one
      auto it_group = matchedflash_index.find( flashidx );
      
      if ( it_group==matchedflash_index.end() ) {
	// not found, create a new group entry
	std::vector<int> group( 1, _fitter.userClusterIndexFromInternal(icluster) );
	group_indices.push_back( std::move(group) );
	group_flashidx.push_back( flashidx );	
	matchedflash_index[flashidx] = group_indices.size()-1;
      }
      else {
	// found, so append to group
	std::vector<int>& group = group_indices[ it_group->second ];
	group.push_back( _fitter.userClusterIndexFromInternal(icluster) );
      }
      
    }//end of cluster groups fit
    
    // merge clusters matched to same flash
    // to-do
    
    // make the output clusters
    // fill: _final_lfcluster_v, _final_clustermask_v
    for ( size_t igroup=0; igroup<group_indices.size(); igroup++ ) {
      int flashidx = group_flashidx[igroup];
      std::vector<int>& group_clusteridx = group_indices[igroup];
      std::cout << "[LArFlowFlashMatch::buildFinalClusters] final cluster group " << igroup
		<< " cluster indices[";
      for ( auto& cidx : group_clusteridx )
	std::cout << " " << cidx;
      std::cout << " ]";
      std::cout << " flashidx=" << flashidx << std::endl;
	

      // make the larflow cluster (almost there!)
      larlite::larflowcluster lfcluster;
      lfcluster._flash_hypo_v.resize( 32, 0.0 );
      lfcluster._flash_data_v.resize( 32, 0.0 );
      
      // make a simple copy
      float fitscore = 0.;
      for ( auto const& clustidx : group_clusteridx ) {
	const larlite::larflowcluster& input = _input_lfcluster_v->at(clustidx);
	fitscore += fitresult.beta[ _fitter.matchIndexFromUserClusterIndex( clustidx ) ];

	// copy the hits
	for ( auto const& hit : input ) {
	  if ( hit.srcwire>=0 && hit.targetwire.size()==2 && hit.targetwire[0]>=0 && hit.targetwire[1]>=0 ) 
	    lfcluster.push_back( hit );
	}
	
	auto const& hypo_v = _fitter.flashHypoFromMatchIndex( _fitter.matchIndexFromUserClusterIndex( clustidx ) );
	for ( size_t ipmt=0; ipmt<hypo_v.size(); ipmt++ )
	  lfcluster._flash_hypo_v[ipmt] += hypo_v[ipmt];
      }
      
      // now we fill the extras
      if ( flashidx>=0 ) {
	const FlashData_t& dataflash = _flashdata_v[flashidx];
	lfcluster.isflashmatched = 1;
	lfcluster.flash_tick    = dataflash.tpc_tick;
	lfcluster.flash_time_us = (dataflash.tpc_tick-2400)*usec_per_tick;
	lfcluster.flashmatch_score = fitscore;
	lfcluster.matchedflash_producer = ( dataflash.isbeam ) ? "simpleFlashBeam" : "simpleFlashCosmic"; // totes bad
	lfcluster.matchedflash_idx = flashidx;
	for ( size_t ipmt=0; ipmt<dataflash.size(); ipmt ++ )
	  lfcluster._flash_data_v[ipmt] = dataflash[ipmt];

	// because we flash-matched, change their x-position
	float offset_cm = (lfcluster.flash_tick-3200)*usec_per_tick*driftv;
	for ( auto& hit : lfcluster ) {
	  hit[0] += offset_cm;
	}
	
      }
      // for comparison, we add info on the truth-match cluster if it exists
      for ( auto const& dataflash : _flashdata_v ) {
	if ( dataflash.truthmatched_clusteridx<0 ) continue;
	bool truthmatches = false;
	for ( auto const& clustidx : group_clusteridx )
	  if ( dataflash.truthmatched_clusteridx==clustidx )
	    truthmatches = true;

	if (truthmatches) {
	  lfcluster.has_truthmatch = 1;
	  lfcluster.is_neutrino = ( dataflash.isneutrino ) ? 1 : 0;
	  lfcluster.truthmatched_mctrackid = dataflash.mctrackid;
	  lfcluster.truthmatched_flashtick = dataflash.tpc_tick;
	}
      }//end of dataflash loop to look for truth match


      if ( lfcluster.size()>0 ) {      
	_final_lfcluster_v.emplace_back( std::move(lfcluster) );
      }
      else {
	std::cout << "[LArFlowFlashMatch::buildFinalCluster] empty skip final cluster group, " << igroup << "." << std::endl;
      }
      
    }//end of loop over (combined) cluster groups
    

    // make the intime cluster
    // a little different, and simpler. If flashes score well enough to fit in-time, keep it
    for ( int flashidx=0; flashidx<(int)_flashdata_v.size(); flashidx++ ) {
      auto const& dataflash = _flashdata_v[flashidx];
      if ( !dataflash.intime ) continue;

      std::cout << "[LArFlowFlashMatch::buildFinalClusters] intime flash at flashidx=" << flashidx << std::endl;
      
      // collect matches
      const std::vector<int>& match_indices =  _fitter.matchIndicesFromUserFlashIndex( flashidx );
      for ( auto const& imatch : match_indices ) {
	int clustidx = _fitter.userClusterIndexFromMatchIndex( imatch );
	CutVars_t& cutvar = getCutVars( flashidx, clustidx );
	std::cout << "[LArFlowFlashMatch::buildFinalClusters] intime flash-cluster (" << flashidx << "," << clustidx << ") "
		  << " match candidate. "
		  << " cutflash=" << cutvar.cutfailed
		  << " fit1fmatch=" << cutvar.fit1fmatch
		  << " fracabove1=" << cutvar.subsamplefit_fracabove1
		  << std::endl;
										     
	if ( cutvar.fit1fmatch>0.15 ) {
	  
	  larlite::larflowcluster lfcluster;
	  for ( auto const& hit : _input_lfcluster_v->at( clustidx ) ) {
	    if ( hit.srcwire>=0 && hit.targetwire.size()==2
		 && hit.targetwire[0]>=0 && hit.targetwire[1]>=0 ) 
	      lfcluster.push_back( hit );
	  }
	  
	  lfcluster.isflashmatched = 1;
	  lfcluster.flash_tick    = dataflash.tpc_tick;
	  lfcluster.flash_time_us = (dataflash.tpc_tick-2400)*usec_per_tick;
	  lfcluster.flashmatch_score = fitresult.beta[ _fitter.matchIndexFromUserClusterIndex( clustidx ) ];
	  lfcluster.matchedflash_producer = ( dataflash.isbeam ) ? "simpleFlashBeam" : "simpleFlashCosmic"; // totes bad
	  lfcluster.matchedflash_idx = flashidx;
	  lfcluster._flash_data_v.resize( dataflash.size(), 0.0 );	  
	  for ( size_t ipmt=0; ipmt<dataflash.size(); ipmt++ )
	    lfcluster._flash_data_v[ipmt] = dataflash[ipmt];
	  
	  
	  auto const& hypo_v = _fitter.flashHypoFromMatchIndex( _fitter.matchIndexFromUserClusterIndex( clustidx ) );
	  lfcluster._flash_hypo_v.resize( hypo_v.size(), 0 );
	  for ( size_t ipmt=0; ipmt<hypo_v.size(); ipmt++ )
	    lfcluster._flash_hypo_v[ipmt] += hypo_v[ipmt];
	  
	  // for comparison, we add info on the truth-match cluster if it exists
	  for ( auto const& dataflash : _flashdata_v ) {
	    if ( dataflash.truthmatched_clusteridx==clustidx ) { 
	      lfcluster.has_truthmatch = 1;
	      lfcluster.is_neutrino = ( dataflash.isneutrino ) ? 1 : 0;
	      lfcluster.truthmatched_mctrackid = dataflash.mctrackid;
	      lfcluster.truthmatched_flashtick = dataflash.tpc_tick;
	    }
	  }//end of dataflash loop to look for truth match
	  
	  if ( lfcluster.size()>0 ) {
	    _intime_lfcluster_v.emplace_back( std::move(lfcluster) );
	  }
	  else {
	    std::cout << "[LArFlowFlashMatch::buildFinalCluster] empty skip cluster for in-time fash" << std::endl;
	  }
	}// if subsample fit > 0.05
      }//match loop
    }// loop over flashes (only running on in-time flashes)
    
    
    // last -- make corresponding cluster mask objects!
    std::vector< larlite::larflowcluster>* pfinalclusters[2]         = { &_final_lfcluster_v,   &_intime_lfcluster_v };
    std::vector< std::vector<larcv::ClusterMask> >* pfinalmasks[2] = { &_final_clustermask_v, &_intime_clustermask_v };
    for ( size_t i=0; i<2; i++ ) {
      for ( auto const& lfcluster : *pfinalclusters[i] ) {

	if ( lfcluster.size()==0 ) {
	  std::cout << "empty cluster" << std::endl;
	  continue;
	}
	
	std::vector<larcv::Point2D> ptU_v;
	std::vector<larcv::Point2D> ptY_v;	
	std::vector<larcv::Point2D> ptV_v;    
	ptU_v.reserve( lfcluster.size() );
	ptV_v.reserve( lfcluster.size() );
	ptY_v.reserve( lfcluster.size() );
	
	float bb_wiremin[3] = {5000,5000,5000};
	float bb_wiremax[3] = {0,0,0};
	float bb_tickmin[3] = {10000,10000,10000};
	float bb_tickmax[3] = {0,0,0};

	for ( auto const& hit : lfcluster ) {

	  if ( hit.srcwire>=0 && hit.targetwire.size()==2 ) {
	    larcv::Point2D ptY( hit.srcwire, img_v[2].meta().row(hit.tick) );
	    ptY_v.emplace_back( std::move(ptY) );
	    bb_tickmin[2] = ( bb_tickmin[2]>hit.tick ) ? hit.tick : bb_tickmin[2];
	    bb_tickmax[2] = ( bb_tickmax[2]<hit.tick ) ? hit.tick : bb_tickmax[2];	
	    bb_wiremin[2] = ( bb_wiremin[2]>hit.srcwire ) ? hit.srcwire : bb_wiremin[2];
	    bb_wiremax[2] = ( bb_wiremax[2]<hit.srcwire ) ? hit.srcwire : bb_wiremax[2];	

	    larcv::Point2D ptU( hit.targetwire[0], img_v[0].meta().row(hit.tick) );
	    ptU_v.emplace_back( std::move(ptU) );
	    bb_tickmin[0] = ( bb_tickmin[0]>hit.tick ) ? hit.tick : bb_tickmin[0];
	    bb_tickmax[0] = ( bb_tickmax[0]<hit.tick ) ? hit.tick : bb_tickmax[0];	
	    bb_wiremin[0] = ( bb_wiremin[0]>hit.targetwire[0] ) ? hit.targetwire[0] : bb_wiremin[0];
	    bb_wiremax[0] = ( bb_wiremax[0]<hit.targetwire[0] ) ? hit.targetwire[0] : bb_wiremax[0];		  


	    larcv::Point2D ptV( hit.targetwire[1], img_v[1].meta().row(hit.tick) );
	    ptV_v.emplace_back( std::move(ptV) );
	    bb_tickmin[1] = ( bb_tickmin[1]>hit.tick ) ? hit.tick : bb_tickmin[1];
	    bb_tickmax[1] = ( bb_tickmax[1]<hit.tick ) ? hit.tick : bb_tickmax[1];	
	    bb_wiremin[1] = ( bb_wiremin[1]>hit.targetwire[1] ) ? hit.targetwire[1] : bb_wiremin[1];
	    bb_wiremax[1] = ( bb_wiremax[1]<hit.targetwire[1] ) ? hit.targetwire[1] : bb_wiremax[1];		  	  
	    
	  }
	}

	for ( size_t p=0; p<3; p++ ) {
	  std::cout << "[LArFlowFlashMatch::buildFinalClusters] bbox plane " << p << " "
		    << "(" << bb_wiremin[p] << "," << bb_tickmin[p] << "," << bb_wiremax[p] << "," << bb_tickmax[p] << ")" << std::endl;
	}
	
	std::vector< larcv::ClusterMask > planemasks_v;
	std::vector<larcv::Point2D>* pPts_v[3] = { &ptU_v, &ptV_v, &ptY_v };
	for ( size_t p=0; p<3; p++ ) {
	  //larcv::ImageMeta meta( bb_wiremin[p], bb_tickmin[p], bb_wiremax[p], bb_tickmax[p],
	  //abs( (bb_tickmax[p]-bb_tickmin[p])/6.0 ), abs( bb_wiremax[p]-bb_wiremin[p] ),
	  //(larcv::ProjectionID_t)p );
	  larcv::BBox2D  bbox( bb_wiremin[p], bb_tickmin[p], bb_wiremax[p], bb_tickmax[p], (larcv::ProjectionID_t)p );
	  larcv::ClusterMask mask( bbox, img_v[p].meta(), *pPts_v[p], 0 );
	  planemasks_v.emplace_back( std::move(mask) );
	}
	pfinalmasks[i]->emplace_back( std::move(planemasks_v) );
      }//end of loop over final/intime cluster container
      
    }// end of cluster loop

    std::cout << "[LArFlowFlashMatch::buildFinalClusters] "
	      << "Built " << _final_lfcluster_v.size() << " flash-matched clusters "
	      << "from original " << _input_lfcluster_v->size() << " larflow clusters" << std::endl;
    std::cout << "[LArFlowFlashMatch::buildFinalClusters] "    
	      << "Built " << _intime_lfcluster_v.size() << " intime clusters" << std::endl;
    
  }//end of buildFlash
  
}
