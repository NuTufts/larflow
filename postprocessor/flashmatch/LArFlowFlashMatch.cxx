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
    : m_compatibility(nullptr),
      _nclusters_compat_wflash(nullptr),
      _compatibility_defined(false),
      _reindexed(false),     
      m_flash_hypo(nullptr),
      m_flash_data(nullptr),
      m_flashhypo_norm(nullptr),
      m_flashdata_norm(nullptr),
      m_iscosmic(nullptr),
      _pair2index(nullptr),
      flightyield(nullptr),
      fmatch(nullptr),
      fmatch_nll(nullptr),
      fmatch_maxdist(nullptr),
      fpmtweight(nullptr),
      _parsdefined(false),
      _psce(nullptr),
      kFlashMatchedDone(false),
      _evstatus(nullptr),
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
    m_flash_hypo_v.clear();

    // more default parameters
    _fMaxDistCut = 0.5;
    _fCosmicDiscThreshold = 10.0;
    _fclustsum_weight = 1e2;
    _fflashsum_weight = 0.5e2;
    _fl1norm_weight = 0.1;
    _flightyield_weight = 1.0;

    // random generators
    _rand = new TRandom3( 4357 );
  }

  LArFlowFlashMatch::~LArFlowFlashMatch() {
    delete _rand;
    resetCompatibilityMatrix();
    clearFittingData();
    clearFitParameters();
    clearMCTruthInfo();
    if ( _fanafile && _anafile_written )
      writeAnaFile();
    if ( _anatree )
      delete _anatree;
    if ( _fanafile )
      _fanafile->Close();
  }

  void LArFlowFlashMatch::clearEvent() {
    _flashdata_v.clear();
    _qcluster_v.clear();
    _qcomposite_v.clear();
    resetCompatibilityMatrix();
    clearMatchHypotheses();
    clearFirstRefinementVariables();
    clearFittingData();
    clearFitParameters();
    clearMCTruthInfo();
    _evstatus = nullptr;
  }
  
  LArFlowFlashMatch::Results_t LArFlowFlashMatch::match( const larlite::event_opflash& beam_flashes,
							 const larlite::event_opflash& cosmic_flashes,
							 const std::vector<larlite::larflowcluster>& clusters,
							 const std::vector<larcv::Image2D>& img_v,
							 const bool ignorelast ) {

    Results_t result_output;
    
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
    if ( kDoTruthMatching && _mctrack_v!=nullptr ) {
      std::cout << "[LArFlowFlashMatch::match][INFO] Doing MCTrack truth-reco matching" << std::endl;
      doFlash2MCTrackMatching( _flashdata_v );
      doTruthCluster2FlashTruthMatching( _flashdata_v, _qcluster_v );
      bool appendtoclusters = true;
    }
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Flash-Cluster-MCTrack Truth-Matching Performed" << std::endl;    
    //dumpQClusterImages();
    //assert(false);
    
    // modifications to fill gaps
    //applyGapFill( _qcluster_v );

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
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Initial Compatible matches formed" << std::endl;

    // refined compabtibility: incompatible-z
    bool adjust_pe_for_cosmic_disc = true;
    reduceMatchesWithShapeAnalysis( _flashdata_v, _qcluster_v, adjust_pe_for_cosmic_disc );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] reduce matches using shape analysis" << std::endl;
    printCompatInfo( _flashdata_v, _qcluster_v );

    dumpQCompositeImages();
    assert(false);
    

    // now build hypotheses: we only do so for compatible pairs
    buildFlashHypotheses( _flashdata_v, _qcluster_v );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Compatible Flash-Cluster match hypotheses formed" << std::endl;
    //std::cin.get();

    if ( kDoTruthMatching && _mctrack_v!=nullptr ) {
      // we provide truth info to the candidates
      for ( auto& candidate : m_matchcandidate_hypo_v ) {
	candidate.addMCTrackInfo( *_mctrack_v );
      }      
    }

    for ( auto& candidate : m_matchcandidate_hypo_v ) {
      if ( candidate.isTruthMatch() )
    	candidate.dumpMatchImage();
    }

    // second refinement
    secondMatchRefinement();
    printCompatInfo( _flashdata_v, _qcluster_v );
    std::cout << "[LArFlowFlashMatch::match][DEBUG] Finished Second Reduction" << std::endl;
    //std::cin.get();

    if ( true ) {
      std::cout << "EARLY EXIT" << std::endl;
      return result_output;
    }
    
    // define the fitting data members
    buildFittingData( _flashdata_v, _qcluster_v );
    
    // define fit parameter variables
    defineFitParameters();


    if ( kDoTruthMatching ) {
      std::cout << "[larflow:::LArFlowFlashMatch::match][INFO] Setting Fit Parameters using truth flashmatch" << std::endl;
      setFitParsWithTruthMatch();
    }
    else {
      // find initial best fit to start
      std::cout << "[larflow:::LArFlowFlashMatch::match][INFO] Setting Fit Parameters using reco flashmatch" << std::endl;      
      setInitialFitPoint( _flashdata_v, _qcluster_v );
    }

    // fit: gradient descent

    // fit: MCMC
    //runMCMC;

    // build flash-posteriors
    //calcFlashPosteriors;
    *flightyield = 1.0;
    _fweighted_scalefactor_sig  = 0.5;
    _fweighted_scalefactor_mean = 1.0;    

    printCompatInfo( _flashdata_v, _qcluster_v );
    dumpMatchImages( _flashdata_v, false, false );
    assert(false);

    int nsamples = 500000;
    int naccept = 0;
    
    // save the current state
    int step = 0;
    int step_accepted = 0;
    float state_ly = *flightyield;
    std::vector<float> state_v(_nmatches,0.0);
    memcpy( state_v.data(), fmatch, _nmatches*sizeof(float) );
    float lastnll = calcNLL(true);

    std::cout << "[enter to start]" << std::endl;
    std::cin.get();

    // proposal vars
    float proposal_ly = 0.;
    std::vector<float> proposal_v(_nmatches,0.0);
    float proposal_nll;
    int dstate;
    
    // accumulation
    TFile fout("out_mcmc_flash.root", "recreate");
    TTree tmcmc("mcmc","MCMC");
    char brname_state[100];
    sprintf( brname_state, "state_v[%d]/F", _nmatches );
    
    tmcmc.Branch("step", &step, "step/I" );
    tmcmc.Branch("accepted", &step_accepted, "accepted/I" );    
    tmcmc.Branch("naccepted", &naccept, "naccepted/I" );    
    tmcmc.Branch("state_v",state_v.data(), brname_state );
    tmcmc.Branch("dstate",&dstate,"dstate/I");
    tmcmc.Branch("ly", &state_ly, "ly/F" );
    tmcmc.Branch("nll", &lastnll, "nll/F" );
    tmcmc.Fill();
    
    int n_ave_updates = 0;
    std::vector<float> ave_v(_nmatches,0.0);
    float best_nll = 1.0e9;
    float best_ly  = 1.0;
    float ave_ly   = 0.0;
    
    auto start = std::clock();    
    
    for (int i=1; i<nsamples; i++) {
      step = i;
      if ( step%10000==0 )
	std::cout << "step[" << step << "]" << std::endl;
      
      // set initial proposal value to current state
      memcpy( proposal_v.data(), state_v.data(), sizeof(float)*_nmatches );
      proposal_ly = state_ly;

      // generate proposal. by effect.
      dstate = (int)generateProposal( 0.5, 0.0, 0.05, proposal_v, proposal_ly );
      
      // set the fit variables to calc nll
      memcpy( fmatch, proposal_v.data(), sizeof(float)*_nmatches );
      *flightyield = proposal_ly;
      
      // accept or reject
      float proposal_nll = calcNLL(step%10000==0);
      bool accept = true;
      //std::cout << "step[" << i << "] (proposalLL<lastnll)=" << (proposal_nll<lastnll) << " ";
      if ( proposal_nll>lastnll )  {
	// worse step, so we have to decide to reject or not
	float logratio = -0.5*(proposal_nll-lastnll);
	//std::cout << " logratio=" << logratio;
	if ( logratio>-1e4 ) {
	  // give it a chance to jump. ay other prob, forget about it
	  float mc = _rand->Uniform();
	  if ( mc > exp(logratio) ) {
	    //std::cout << "  drew step change (" << mc << ">" << exp(logratio) << ") ";
	    accept = false; // we reject
	  }
	}
	else
	  accept = false;
      }
      
      if ( accept ) {
	// update state
	//std::cout << "  accept" << std::endl;

	// copy proposal to state_v, state_ly
	memcpy( state_v.data(), proposal_v.data(), sizeof(float)*_nmatches );
	state_ly = proposal_ly;
	lastnll = proposal_nll;
	step_accepted = 1;
	naccept++;
      }
      else {
	//std::cout << "  reject" << std::endl;
	step_accepted = 0;	
      }

      if ( step>nsamples/2 ) {
	// start to update after burn in
	n_ave_updates += 1;
	for (int i=0; i<_nmatches; i++) {
	  ave_v[i] += state_v[i];
	}
	ave_ly += state_ly;
      }
      
      tmcmc.Fill();
    }//end of mcmc loop
    
    for (int i=0; i<_nmatches; i++) {
      ave_v[i] /= float(n_ave_updates);
    }
    ave_ly  /= float(n_ave_updates);    
    
    double tnll_s = (std::clock() - start) / (double)(CLOCKS_PER_SEC);
    std::cout << "Time: " << tnll_s << " s (per sample: " << tnll_s/float(1000)/float(nsamples) << " ms)" << std::endl;
    std::cout << "num accepted proposals: " << naccept << " out of " << nsamples << std::endl;    
    // set with ending state
    std::cout << "Set average state. ave_ly=" << ave_ly << std::endl;
    memcpy( fmatch, ave_v.data(), sizeof(float)*_nmatches );
    *flightyield = ave_ly;
    calcNLL(true);
    tmcmc.Write();
    fout.Close();

    std::cout << "ave match vector" << std::endl;
    for (int imatch=0; imatch<_nmatches; imatch++) {
      std::cout << "imatch[" << imatch << "] "
		<< "flash=" << _match_flashidx_orig[imatch] << " "
		<< "clust=" << _match_clustidx_orig[imatch] << " "
		<< "ave=" << ave_v[imatch] << std::endl;
    }

    std::cout << std::endl;

    std::cout << "distribution over clusters" << std::endl;
    for (int iclust=0; iclust<_nqclusters; iclust++) {
      auto it_clust = _clust_reindex.find( iclust );
      if ( it_clust==_clust_reindex.end() )
	continue;

      int reclust = it_clust->second;
      std::cout << "cluster[" << iclust << ": truthflash=" << _qcluster_v[iclust].truthmatched_flashidx << "] ";
      
      // find non-zero flash
      for (int iflash=0; iflash<_nflashes; iflash++) {
	auto it_flash = _flash_reindex.find( iflash );
	if ( it_flash==_flash_reindex.end() )
	  continue;

	int reflash = it_flash->second;
	int imatch = getMatchIndex( reflash, reclust );
	if ( ave_v[imatch]>0.05 ) {
	  std::cout << "fl[" << iflash << "]=" << ave_v[imatch] <<  " ";
	}
      }
      std::cout << std::endl;
    }

    std::cout << "distribution over flashes" << std::endl;
    for (int iflash=0; iflash<_nflashes; iflash++) {
      auto it_flash = _flash_reindex.find( iflash );
      if ( it_flash==_flash_reindex.end() )
	continue;

      int reflash = it_flash->second;
      std::cout << "flash[" << iflash << ": truthcluster=" << _flashdata_v[iflash].truthmatched_clusteridx << "] ";
      
      // find non-zero flash
      for (int iclust=0; iclust<_nqclusters; iclust++) {
	auto it_clust = _clust_reindex.find( iclust );
	if ( it_clust==_clust_reindex.end() )
	  continue;

	int reclust = it_clust->second;
	int imatch = getMatchIndex( reflash, reclust );
	if ( ave_v[imatch]>0.05 ) {
	  std::cout << "cl[" << iclust << "]=" << ave_v[imatch] <<  " ";
	}
      }
      std::cout << std::endl;
    }
    std::cout << "FIN" << std::endl;
    dumpMatchImages( _flashdata_v, false, true );

    return result_output;
  }

  // ==============================================================================
  // CHARGE CLUSTER TOOLS
  // -------------------------------------------------------------------------------

  void LArFlowFlashMatch::buildInitialQClusters( const std::vector<larlite::larflowcluster>& lfclusters, std::vector<QCluster_t>& qclusters,
						 const std::vector<larcv::Image2D>& img_v, const int src_plane, bool ignorelast ) {

    // we take the larflow 3dhit clusters and convert them into QCluster_t objects
    // these are then used to build QClusterComposite objects, which
    //  are able to generate flash-hypotheses in a number of configurations
    
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
    
  }
  std::vector<FlashData_t> LArFlowFlashMatch::collectFlashInfo( const larlite::event_opflash& beam_flashes,
								const larlite::event_opflash& cosmic_flashes ) {

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
    
    return flashdata;
  }

  void LArFlowFlashMatch::buildFullCompatibilityMatrix( const std::vector< FlashData_t >& flash_v,
							const std::vector< QCluster_t>& qcluster_v ) {
    if ( m_compatibility )
      delete [] m_compatibility;

    _nflashes   = flash_v.size();
    _nqclusters = qcluster_v.size();
    _nelements  = _nflashes*_nqclusters;
    m_compatibility = new int[ _nelements ];
    _nclusters_compat_wflash = new int[_nflashes];
    memset( _nclusters_compat_wflash, 0, sizeof(int)*_nflashes );

    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float max_drifttime_ticks = (256.0+20.0)/larp->DriftVelocity()/0.5; // allow some slop

    //std::cout << "max drifttime in ticks: " << max_drifttime_ticks << std::endl;
    
    // we mark the compatibility of clusters
    int ncompatmatches = 0;
    for (size_t iflash=0; iflash<flash_v.size(); iflash++) {
      const FlashData_t& flash = flash_v[iflash];
      
      for ( size_t iq=0; iq<qcluster_v.size(); iq++) {
	const QCluster_t& qcluster = qcluster_v[iq];

	float dtick_min = qcluster.min_tyz[0] - flash.tpc_tick;
	float dtick_max = qcluster.max_tyz[0] - flash.tpc_tick;

	// must happen after (allow for some slop)
	if ( dtick_min < -10 || dtick_max < -10 ) {
	  setCompat( iflash, iq, 1 ); // too early
	}
	else if ( dtick_min > max_drifttime_ticks ) {
	  setCompat( iflash, iq, 2 ); // too late
	}
	else {
	  setCompat( iflash, iq, 0 ); // ok
	  ncompatmatches++;
	  _nclusters_compat_wflash[iflash] += 1;
	}
      }
    }
    _compatibility_defined = true;
    std::cout << "number of compat flash-cluster matches: " << ncompatmatches << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "NUM COMPAT CLUSTERS PER FLASH" << std::endl;
    for (size_t iflash=0; iflash<flash_v.size(); iflash++)
      std::cout << "[dataflash " << iflash << "] " << _nclusters_compat_wflash[iflash] << std::endl;
    std::cout << "------------------------------------------" << std::endl;      
  }

  void LArFlowFlashMatch::resetCompatibilityMatrix() {
    _nflashes = 0;
    _nqclusters = 0;
    _nelements = 0;
    _compatibility_defined = false;
    delete [] _nclusters_compat_wflash;
    delete [] m_compatibility;    
    _nclusters_compat_wflash = nullptr;
    m_compatibility = nullptr;
  }

  void LArFlowFlashMatch::buildFlashHypotheses( const std::vector<FlashData_t>& flashdata_v,
						const std::vector<QCluster_t>&  qcluster_v ) {
    
    // each (flash,cluster) pair builds a hypothesis
    m_flash_hypo_map.clear();
    m_flash_hypo_v.clear(); // deprecated
    m_matchcandidate_hypo_v.reserve( flashdata_v.size()*qcluster_v.size() );
    
    for (int iflash=0; iflash<flashdata_v.size(); iflash++) {

      const FlashData_t& flash = flashdata_v[iflash]; // original flash

      for ( int iq=0; iq<qcluster_v.size(); iq++) {
	int compat = getCompat( iflash, iq );
	if ( compat!=0 && flash.truthmatched_clusteridx!=iq )
	  continue;
	
	const QCluster_t& qcluster = qcluster_v[iq];
	FlashMatchCandidate match_candidate( flash, qcluster );
	match_candidate.setChStatusData( _evstatus );

	// store
	int idx = m_flash_hypo_v.size();
	m_flash_hypo_map[ flashclusterpair_t((int)iflash,iq) ] = idx;
	m_matchcandidate_hypo_v.emplace_back( std::move(match_candidate) );
	
	// DEBUG HACK
	//break;
      }//end of loop over clusters
    }//end of loop over flashes
    
  }

  void LArFlowFlashMatch::clearMatchHypotheses() {
    m_flash_hypo_map.clear();
    m_flash_hypo_v.clear();
    m_matchcandidate_hypo_v.clear();
  }
  
  bool LArFlowFlashMatch::hasHypothesis( int flashidx, int clustidx ) {
    flashclusterpair_t fcpair( flashidx, clustidx );
    auto it = m_flash_hypo_map.find( fcpair );
    if ( it==m_flash_hypo_map.end() ) return false;
    return true;
  }
  
  FlashHypo_t& LArFlowFlashMatch::getHypothesisWithOrigIndex( int flashidx, int clustidx ) {
    
    flashclusterpair_t fcpair( flashidx, clustidx );
    auto it = m_flash_hypo_map.find( fcpair );
    if ( it==m_flash_hypo_map.end() ) {
      std::stringstream ss;
      ss << "[LArFlowFlashMatch::getHypothesisWithOrigIndex][ERROR] could not find hypothesis with given (" << flashidx << "," << clustidx << ") original index" << std::endl;
      throw std::runtime_error(ss.str());
    }
    int index = it->second;
    return m_flash_hypo_v[index];
  }
  
  int LArFlowFlashMatch::getMatchIndexFromOrigIndices( int flashidx, int clustidx ) {
    flashclusterpair_t fcpair( flashidx, clustidx );
    auto it = m_flash_hypo_map.find( fcpair );
    if ( it==m_flash_hypo_map.end() )
      return -1;
    return it->second;
  }

  void LArFlowFlashMatch::getFlashClusterIndexFromMatchIndex( int matchidx, int& flashidx, int& clustidx ) {
    if ( matchidx<0 || matchidx>m_matchcandidate_hypo_v.size() ) {
      flashidx = -1;
      clustidx = -1;
      return;
    }
    m_matchcandidate_hypo_v.at(matchidx).getFlashClusterIndex( flashidx, clustidx );
  }

  void LArFlowFlashMatch::buildFittingData( const std::vector<FlashData_t>& flashdata_v,
					    const std::vector<QCluster_t>&  qcluster_v ) {

    if ( !_compatibility_defined )
      throw std::runtime_error("[LArFlowFlashMatch::buildFittingData][ERROR] compatbility matrix must be defined before calling this function.");
    
    clearFittingData();

    // we save only those flashes and clusters with more than one possible match
    const int npmts = 32;
    
    for (size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
      int ncluster_matches = 0;
      std::vector<int> clust_idx;
      clust_idx.reserve( qcluster_v.size() );
      for ( int iq=0; iq<qcluster_v.size(); iq++) {
	int compat = getCompat( iflash, iq );
	if ( compat==0 ) {
	  _nmatches++;
	  clust_idx.push_back( iq );
	  ncluster_matches++;
	}
      }
      if ( ncluster_matches>1 ) {
	// register flash index to be used
	_flash_reindex[iflash] = _flash_reindex.size(); 

	// register clusters to be used, record flash/cluster pair
	for (auto& qidx : clust_idx ) {
	  if ( _clust_reindex.find(qidx)==_clust_reindex.end() )
	    _clust_reindex[qidx] = _clust_reindex.size();
	  
	  _match_flashidx.push_back( _flash_reindex[iflash] ); // store reindex
	  _match_flashidx_orig.push_back( iflash ); // store original index
	  _match_clustidx.push_back( _clust_reindex[qidx] );   // store reindex
	  _match_clustidx_orig.push_back( qidx ); // store original index
	}
      }
    }
    
    // prep variables
    _nmatches        = _match_flashidx.size();
    _nflashes_red    = _flash_reindex.size();
    _nclusters_red   = _clust_reindex.size();
    m_flash_hypo     = new float[ _nmatches*npmts ];
    m_flash_data     = new float[ _nmatches*npmts ];
    m_flashdata_norm = new float[ _nmatches ];
    m_flashhypo_norm = new float[ _nmatches ];
    m_iscosmic       = new int[ _nmatches ];

    // create a map from (reflash,recluster) pair to match index
    _pair2index = new int[ _nflashes_red*_nclusters_red ];
    for (int i=0; i<_nflashes_red*_nclusters_red; i++)
      _pair2index[i] = -1;
    for (int imatch=0; imatch<_nmatches; imatch++){
      int reflash  = _match_flashidx[imatch];
      int reclust  = _match_clustidx[imatch];
      *(_pair2index + _nclusters_red*reflash + reclust ) = imatch;
    }

    // copy data flash and hypo flash to m_flash_data, m_flash_hypo, respectively
    for (size_t imatch=0; imatch<_nmatches; imatch++) {
      size_t reflash = _match_flashidx[imatch];
      int reclust    = _match_clustidx[imatch];
      size_t origflashidx = _match_flashidx_orig[imatch];
      size_t origclustidx = _match_clustidx_orig[imatch];

      // copy data
      memcpy( m_flash_data+imatch*npmts, flashdata_v[origflashidx].data(), sizeof(float)*npmts );
      *( m_flashdata_norm+imatch ) = flashdata_v[origflashidx].tot;

      FlashHypo_t& hypo = getHypothesisWithOrigIndex( origflashidx, origclustidx );
      memcpy( m_flash_hypo+imatch*npmts, hypo.data(), sizeof(float)*npmts );
      *( m_flashhypo_norm+imatch ) = hypo.tot;

      if ( flashdata_v[origflashidx].isbeam )
	*(m_iscosmic ) = 0;
      else
	*(m_iscosmic ) = 1;
    }


    
    std::cout << "prepared fit data. " << std::endl;
    std::cout << "  reindexed flashes: "  << _nflashes_red  << " (of " << flashdata_v.size() << " original)" << std::endl;
    std::cout << "  reindexed clusters: " << _nclusters_red << " (of " << qcluster_v.size()  << " original)" << std::endl;
    _reindexed = true;
  }

  bool LArFlowFlashMatch::doOrigIndexPairHaveMatch( int flashidx_orig, int clustidx_orig ) {
    auto it_flash = _flash_reindex.find( flashidx_orig );
    if ( it_flash==_flash_reindex.end() ) return false;
    auto it_clust = _clust_reindex.find( clustidx_orig );
    if ( it_clust==_clust_reindex.end() ) return false;
    return true;
  }

  std::vector<float> LArFlowFlashMatch::getMatchScoresForCluster( int icluster ) {

    if (!_parsdefined) {
      throw std::runtime_error( "[LArFlowFlashMatch::getMatchScoresForCluster][ERROR] defineFitParameters needed to have been called first." );
    }
    
    // icluster is for original index
    std::vector<float> matchscores_v( _nflashes, 0.0 );
    
    auto it_clust = _clust_reindex.find( icluster );
    if ( it_clust==_clust_reindex.end() ) {
      // wasnt in fit. return the zero vector
      return matchscores_v;
    }

    int reclust = it_clust->second;

    // find non-zero flash
    for (int iflash=0; iflash<_nflashes; iflash++) {
      auto it_flash = _flash_reindex.find( iflash );
      if ( it_flash==_flash_reindex.end() )
	continue;

      int reflash = it_flash->second;
      int imatch = getMatchIndex( reflash, reclust );
      matchscores_v[iflash] = *(fmatch+imatch);
    }
    return matchscores_v;
  }

  void LArFlowFlashMatch::clearFittingData() {
    _nmatches      = 0;
    _nflashes_red  = 0;
    _nclusters_red = 0;
    _clust_reindex.clear();
    _flash_reindex.clear();
    _match_flashidx.clear();
    _match_flashidx_orig.clear();    
    _match_clustidx.clear();
    _match_clustidx_orig.clear();        

    delete [] _pair2index;    
    delete [] m_flash_hypo;
    delete [] m_flash_data;
    delete [] m_flashhypo_norm;
    delete [] m_flashdata_norm;
    
    m_flash_hypo = nullptr;
    m_flash_data = nullptr;
    m_flashhypo_norm = nullptr;
    m_flashdata_norm = nullptr;
    _pair2index = nullptr;
    _reindexed = false;
  }

  void LArFlowFlashMatch::zeroMatchVector() {
    memset( fmatch, 0, sizeof(float)*_nmatches );
  }

  void LArFlowFlashMatch::defineFitParameters() {
    if ( !_reindexed )
      throw std::runtime_error("[LArFlowFlashMatch::defineFitParametere][ERROR] must call buildFittingData first");
    clearFitParameters();
    flightyield = new float;
    fmatch = new float[_nmatches];
    fmatch_nll = new float[_nmatches];
    fmatch_maxdist = new float[_nmatches];
    fpmtweight = new float[_nflashes_red*32];
    
    memset( fpmtweight, 0, sizeof(_nflashes_red )*32 );
    memset( fmatch, 0, sizeof(float)*_nmatches );
    memset( fmatch_nll, 0, sizeof(float)*_nmatches );
    memset( fmatch_maxdist, 0, sizeof(float)*_nmatches );    
    _parsdefined = true;
  }

  void LArFlowFlashMatch::clearFitParameters() {
    delete flightyield;
    delete [] fmatch;
    delete [] fmatch_nll;
    delete [] fmatch_maxdist;
    delete [] fpmtweight;
    flightyield = nullptr;
    fmatch = nullptr;
    fmatch_nll = nullptr;
    fmatch_maxdist = nullptr;
    fpmtweight = nullptr;
    _parsdefined = false;
  }
  
  void LArFlowFlashMatch::setInitialFitPoint(const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>&  qcluster_v ) {

    if ( !_parsdefined )
      throw std::runtime_error("[LArFlowFlashMatch::setInitialFitPoint][ERROR] must call setInitialFitPoint first");
    
    (*flightyield) = _fweighted_scalefactor_mean;
    memset( fmatch, 0, sizeof(float)*_nmatches );
    for (int iclust=0; iclust<_nqclusters; iclust++) {
      auto it = _clust_reindex.find( iclust );
      if ( it==_clust_reindex.end() )
	continue; // not paired
      int bestflashidx = _clustdata_best_hypo_chi2_idx[iclust];
      int reflashidx   = _flash_reindex[bestflashidx];
      int reclustidx   = (*it).second;
      int imatch = getMatchIndex( reflashidx, reclustidx );
      *(fmatch+imatch) = 1.0;
    }
    std::cout << "initialstateset: ";
    for (int i=0; i<_nmatches; i++)
      std::cout << fmatch[i];
    std::cout << std::endl;
    
  }

  // float LArFlowFlashMatch::shapeComparison( const FlashHypo_t& hypo, const FlashData_t& data, float data_norm, float hypo_norm ) {
  //   // we sum over z-bins and measure the max-distance of the CDF
  //   const int nbins_cdf = _zbinned_pmtchs.size();
  //   float hypo_cdf[nbins_cdf] = {0};
  //   float data_cdf[nbins_cdf] = {0};
  //   float maxdist = 0;

  //   float norm_hypo = 0.;
  //   float norm_data = 0.;
    
  //   // fill cdf
  //   for (int ibin=0; ibin<nbins_cdf; ibin++) {
  //     float binsum_hypo = 0.;
  //     float binsum_data = 0.;                
  //     for (auto& ich : _zbinned_pmtchs[ibin] ) {
  // 	binsum_hypo += hypo[ich]*hypo_norm;
  // 	binsum_data += data[ich]*data_norm;
  //     }
  //     norm_hypo += binsum_hypo;
  //     norm_data += binsum_data;
  //     hypo_cdf[ibin] = norm_hypo;
  //     data_cdf[ibin] = norm_data;
  //   }

  //   // norm cdf and find maxdist
  //   for (int ibin=0; ibin<nbins_cdf; ibin++) {
  //     if ( norm_hypo>0 )
  // 	hypo_cdf[ibin] /= norm_hypo;
  //     if ( norm_data>0 )
  // 	data_cdf[ibin] /= norm_data;
    
  //     float dist = fabs( hypo_cdf[ibin]-data_cdf[ibin]);
  //     if ( dist>maxdist )
  // 	maxdist = dist;
  //   }
  //   //std::cout << "tot_hypo=" << norm_hypo << " tot_data=" << norm_data << " maxdist=" << maxdist << std::endl;
  //   return maxdist;
  // }

  float LArFlowFlashMatch::chi2Comparison( const FlashHypo_t& hypo, const FlashData_t& data, float data_norm, float hypo_norm ) {
    // chi2
    float chi2 = 0;
    for (size_t ich=0; ich<data.size(); ich++) {
      float pred = hypo[ich]*hypo_norm;
      float obs  = data[ich];
      float err = sqrt( pred + obs );
      if (pred+obs<=0) {
	err = 1.0e-3;
      }
      chi2 += (pred-obs)*(pred-obs)/(err*err)/((float)data.size());
    }
    return chi2;
  }

  void LArFlowFlashMatch::reduceMatchesWithShapeAnalysis( const std::vector<FlashData_t>& flashdata_v,
							  const std::vector<QCluster_t>&  qcluster_v,
							  bool adjust_pe_for_cosmic_disc ) {
    // FIRST STAGE REDUCTION
    // from this function we reduced number of possible flash-cluster matches
    // this is done by
    // (1) comparing shape (CDF maxdist) and chi2
    // (2) for chi2, if adjust_pe_for_cosmic_disc, we try to account for pe lost due to cosmic disc threshold
    
    _flashdata_bestmatch.clear();
    _flashdata_bestmatch.resize( flashdata_v.size() );
    
    _clustdata_best_hypo_maxdist_idx.resize(qcluster_v.size(),-1);
    _clustdata_best_hypo_chi2_idx.resize(qcluster_v.size(),-1);    

    std::vector<float> hyposcale_v;
    std::vector<float> scaleweight_v;    
    hyposcale_v.reserve( flashdata_v.size()*qcluster_v.size() );
    
    for (int iflash=0; iflash<flashdata_v.size(); iflash++) {
      
      const FlashData_t& flashdata = flashdata_v[iflash];
      float xoffset = (flashdata.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
      
      std::vector< int > clustmatches;
      std::vector< float > maxdist;
      float bestdist = 2.0;
      int bestidx = -1;
      int bestchi2_idx = -1;
      float bestchi2 = -1;
      
      for (int iclust=0; iclust<qcluster_v.size(); iclust++) {

	if ( _save_ana_tree ) {
	  // clear ana tree variables
	  clearAnaVariables();
	}

	// set ana tree variables
	if ( flashdata.mctrackid>=0 ) {
	  _truthmatch = (flashdata.truthmatched_clusteridx==iclust) ? 1 : 0;
	}
	else
	  _truthmatch = -1;
	_intime     = (flashdata.intime) ? 1 : 0;
	_isneutrino = (flashdata.isneutrino) ? 1 : 0;
	_isbeam     = (flashdata.isbeam) ? 1 : 0;
	_datape     = flashdata.tot;
	
	if ( getCompat( iflash, iclust )!=0 ) {
	  // fails time-overlap compatibility
	  if (_save_ana_tree ) {
	    _redstep = 0;
	    _anatree->Fill();
	  }
	  continue;
	}

	const QClusterComposite& qcomposite = _qcomposite_v[iclust];
	
	// build flash hypothesis for qcluster-iflash pair
	FlashCompositeHypo_t comphypo_wext  = qcomposite.generateFlashCompositeHypo( flashdata, true );
	FlashCompositeHypo_t comphypo_noext = qcomposite.generateFlashCompositeHypo( flashdata, false );

	FlashHypo_t hypo_wext = comphypo_wext.makeHypo();
	FlashHypo_t hypo_noext = comphypo_noext.makeHypo();	

	float maxdist_wext  = FlashMatchCandidate::getMaxDist( flashdata, hypo_wext, false );
	float maxdist_noext = FlashMatchCandidate::getMaxDist( flashdata, hypo_noext, false );	
	
	// remove clearly bad matches
	float maxdist = ( maxdist_wext<maxdist_noext ) ? maxdist_wext : maxdist_noext;
	if ( maxdist > 0.5 ) {
	  setCompat(iflash,iclust,4); // fails shape match
	}
	
	
	// float hypo_renorm = 0.;
	// if ( hypo_renorm == 0.0 ) {
	//   // no overlap between data and hypo -- good, can reject
	//   if ( _save_ana_tree ) {
	//     _redstep = 1;
	//     _maxdist_orig = 1.0; // this is basically the test we did
	//     _anatree->Fill();
	//   }
	//   setCompat(iflash,iclust,3); // no overlap
	  
	// }
	// else {
	//   //FlashHypo_t& copy = hypo;
	//   // give ourselves a new working copy
	//   FlashHypo_t copy(hypo);
	  
	//   //float hypo_scale = flashdata.tot/(hypo_renorm/hypo.tot); // we want
	//   float hypo_scale = flashdata.tot;
	//   //std::cout << "data.tot=" << flashdata.tot << " hypo_scale=" << hypo_scale << " copy.tot=" << copy.tot << " copy.size=" << copy.size() << std::endl;
	  
	//   // we enforce cosmic dic. threshold by scaling hypo to data and zero-ing below threshold
	//   copy.tot = 0.0; // copy norm
	//   for (size_t ich=0; ich<hypo.size(); ich++) {
	//     float copychpred = hypo[ich]*hypo_scale;
	//     if ( adjust_pe_for_cosmic_disc && copychpred<_fCosmicDiscThreshold )
	//       copy[ich] = 0.;
	//     else
	//       copy[ich] = copychpred;
	//     //std::cout << "copy.chpred=" << copy[ich] << " vs. chpred=" << copychpred << std::endl;	  
	//     copy.tot += copy[ich];
	//   }
	//   //std::cout << "copy.tot=" << copy.tot << std::endl;
	//   if ( copy.tot==0 ) {
	//     setCompat(iflash,iclust,3);
	//     if ( _save_ana_tree ) { 	    
	//       _redstep = 1;
	//       _maxdist_orig = 1.0; // this is basically the test we did
	//       _anatree->Fill();
	//     }
	//     continue;
	//   }

	//   // normalize
	//   for (size_t ich=0; ich<flashdata.size(); ich++)
	//     copy[ich] /= copy.tot;
	  
	//   float maxdist = FlashMatchCandidate::getMaxDist( flashdata, copy );
	//   float chi2    = chi2Comparison( copy, flashdata, flashdata.tot, copy.tot );
	  
	//   hyposcale_v.push_back( copy.tot/flashdata.tot  ); // save data/mc ratio
	//   scaleweight_v.push_back( exp(-0.5*chi2 ) );

	//   //std::cout << "hyposcale=" << hypo_scale << "  chi2=" << chi2 << std::endl;

	//   if ( maxdist>_fMaxDistCut ) {
	//     setCompat(iflash,iclust,4); // fails shape match
	//     if ( _save_ana_tree ) {
	//       _redstep = 1;
	//       _maxdist_orig = maxdist; // this is basically the test we did
	//       _peratio_orig = FlashMatchCandidate::getPERatio( flashdata, copy );
	//       _anatree->Fill();
	//     }
	//     continue;
	//   }

	//   // update bests for flash
	//   if ( maxdist < bestdist ) {
	//     bestdist = maxdist;
	//     bestidx = iclust;
	//   }
	//   if ( chi2 < bestchi2 || bestchi2<0 ) {
	//     bestchi2 = chi2;
	//     bestchi2_idx = iclust;
	//   }

	//   // update bests for clust
	//   // if ( maxdist < _clustdata_best_hypo_maxdist[iclust] ) {
	//   //   _clustdata_best_hypo_maxdist_idx[iclust] = iflash;
	//   //   std::cout << "update best cluster flash: maxdist=" << maxdist << " idx=" << iflash << std::endl;
	//   // }
	//   // if ( chi2 < _clustdata_best_hypo_chi2[iclust] ) {
	//   //   _clustdata_best_hypo_chi2_idx[iclust] = iflash;
	//   // }
	  
	// }//end of if valid renorm


	
      }//end of cluster loop

      // store bests
      // _flashdata_best_hypo_maxdist_idx[iflash] = bestidx;
      // _flashdata_best_hypo_maxdist[iflash]     = bestdist;
      // _flashdata_best_hypo_chi2_idx[iflash]    = bestchi2_idx;
      // _flashdata_best_hypo_chi2[iflash]        = bestchi2;

    }//end of flash loop

    if ( true ) {
      TCanvas c("c","c",800,600);
      TH1D hscale("hscale", "",50,0,1e3);
      for ( size_t i=0; i<hyposcale_v.size(); i++) {
	hscale.Fill( hyposcale_v[i], scaleweight_v[i] );
      }
      hscale.Draw("hist");
      c.Update();
      c.Draw();
      c.SaveAs("hyposcale.png");
    }

    // calculate weight-mean light-yield value;
    _fweighted_scalefactor_mean  = 0.;
    _fweighted_scalefactor_var = 0.;
    _fweighted_scalefactor_sig = 0.;    
    float totweight = 0;
    float xxsum = 0;
    float xsum  = 0;
    float nweights  = 0;
    for ( size_t i=0; i<hyposcale_v.size(); i++) {
      float scale = hyposcale_v[i];
      float weight = scaleweight_v[i];

      if ( weight>1.0e-6 ) {
	xsum  += scale*weight;
	xxsum += scale*scale*weight;
	totweight += weight;
	nweights += 1.0;
      }
    }
    _fweighted_scalefactor_mean  = xsum/totweight;
    _fweighted_scalefactor_var = (xxsum - 2.0*xsum*_fweighted_scalefactor_mean + totweight*_fweighted_scalefactor_mean*_fweighted_scalefactor_mean)/(totweight*(nweights-1.0)/nweights);
    _fweighted_scalefactor_sig = sqrt( _fweighted_scalefactor_var );
    //float x_zero = _fweighted_scalefactor_mean/_fweighted_scalefactor_sig;
    //_ly_neg_prob = 0.5*TMath::Erf( x_zero );

    std::cout << "total weight: " << totweight << std::endl;
    std::cout << "total number of weights: " << int(nweights) << std::endl;
    std::cout << "Weighted scale-factor mean: "   << _fweighted_scalefactor_mean  << std::endl;
    std::cout << "Weighted scale-factor variance (stdev): " << _fweighted_scalefactor_var << " ("  << _fweighted_scalefactor_sig << ")" << std::endl;
  }

  void LArFlowFlashMatch::clearFirstRefinementVariables() {
    // _flashdata_best_hypo_chi2_idx.clear();
    // _flashdata_best_hypo_chi2.clear();
    // _flashdata_best_hypo_maxdist_idx.clear();
    // _flashdata_best_hypo_maxdist.clear();
    _clustdata_best_hypo_chi2_idx.clear();
    _clustdata_best_hypo_maxdist_idx.clear();
  }

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
	// if ( _flashdata_best_hypo_maxdist_idx.size()==_nflashes && _flashdata_best_hypo_maxdist_idx[iflash]==idx )
	//   bestmaxdist = true;
	// if ( _flashdata_best_hypo_chi2_idx.size()==_nflashes && _flashdata_best_hypo_chi2_idx[iflash]==idx )
	//   bestchi2 = true;

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

  void LArFlowFlashMatch::dumpQClusterImages() {
    /*
    gStyle->SetOptStat(0);
    
    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();    
    const float  driftv = larp->DriftVelocity();    
    
    TCanvas c2d("c2d","pmt flash", 1200, 600);
    TPad datayz("pad1", "",0.0,0.5,1.0,1.0);
    TPad dataxy("pad2", "",0.0,0.0,1.0,0.5);
    datayz.SetRightMargin(0.05);
    datayz.SetLeftMargin(0.05);    
    dataxy.SetRightMargin(0.05);
    dataxy.SetLeftMargin(0.05);    

    // shapes/hists used for each plot
    TH2D bg("hyz","",105,-20,1050, 120, -130, 130);
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
    for (int iclust=0; iclust<(int)_qcluster_v.size(); iclust++) {

      std::cout << "[larflow::LArFlowFlashMatch::dumpQClusterImages][INFO] Cluster " << iclust << std::endl;
      
      const QCluster_t& qcluster = _qcluster_v[iclust];
      const QClusterComposite& qcore  = _qcomposite_v[iclust];
      const QCluster_t& qfill = qcore._gapfill_qcluster;

      // make graph of core and non-core
      TGraph* gcore_zy    = new TGraph( qcore._core.size() );
      TGraph* gcore_xy    = new TGraph( qcore._core.size() );     
      TGraph* gnoncore_zy = new TGraph( qcore._noncore_hits );
      TGraph* gnoncore_xy = new TGraph( qcore._noncore_hits );
      TGraph* gfill_zy = new TGraph( qfill.size() );
      TGraph* gfill_xy = new TGraph( qfill.size() );
      
      for ( int iq=0; iq<(int)qcore._core.size(); iq++ ) {
	gcore_zy->SetPoint( iq, qcore._core[iq].xyz[2], qcore._core[iq].xyz[1] );
	gcore_xy->SetPoint( iq, qcore._core[iq].xyz[0], qcore._core[iq].xyz[1] );	
      }
      int inoncore = 0;
      for (auto const& qnoncore : qcore._noncore ) {
	for ( int iq=0; iq<(int)qnoncore.size(); iq++ ) {
	  gnoncore_zy->SetPoint( inoncore, qnoncore[iq].xyz[2], qnoncore[iq].xyz[1] );
	  gnoncore_xy->SetPoint( inoncore, qnoncore[iq].xyz[0], qnoncore[iq].xyz[1] );
	  inoncore++;
	}
      }
      gnoncore_zy->Set(inoncore);
      gnoncore_xy->Set(inoncore);
      for ( int iq=0; iq<(int)qfill.size(); iq++ ) {
	gfill_zy->SetPoint( iq, qfill[iq].xyz[2], qfill[iq].xyz[1] );
	gfill_xy->SetPoint( iq, qfill[iq].xyz[0], qfill[iq].xyz[1] );	
      }      

      gcore_zy->SetMarkerColor(kRed);
      gcore_xy->SetMarkerColor(kRed);      
      gnoncore_zy->SetMarkerColor(kBlack);
      gnoncore_xy->SetMarkerColor(kBlack);
      gfill_zy->SetMarkerColor(kBlue);
      gfill_xy->SetMarkerColor(kBlue);
      TGraph* g_v[6] = { gcore_zy, gcore_xy, gnoncore_zy, gnoncore_xy, gfill_zy, gfill_xy };
      for (int i=0; i<6; i++) {
	g_v[i]->SetMarkerStyle(20);
	g_v[i]->SetMarkerSize(0.3);
      }

      // PCA lines
      TGraph* gcore_zy_pca = new TGraph(3);
      TGraph* gcore_xy_pca = new TGraph(3);      
      const larlite::pcaxis& pc = qcore._pca_core;
      gcore_zy_pca->SetPoint( 0,
			      pc.getAvePosition()[2] - 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[2][0],
			      pc.getAvePosition()[1] - 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[1][0]);
      gcore_zy_pca->SetPoint( 1, pc.getAvePosition()[2], pc.getAvePosition()[1] );
      gcore_zy_pca->SetPoint( 2,
			      pc.getAvePosition()[2] + 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[2][0],
			      pc.getAvePosition()[1] + 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[1][0]);
      gcore_xy_pca->SetPoint( 0,
			      pc.getAvePosition()[0] - 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[0][0],
			      pc.getAvePosition()[1] - 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[1][0]);
      gcore_xy_pca->SetPoint( 1, pc.getAvePosition()[0], pc.getAvePosition()[1] );
      gcore_xy_pca->SetPoint( 2,
			      pc.getAvePosition()[0] + 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[0][0],
			      pc.getAvePosition()[1] + 2.0*sqrt(pc.getEigenValues()[0])*pc.getEigenVectors()[1][0]);
      

      // Set the pads
      c2d.Clear();
      c2d.Draw();
      c2d.cd();
      
      // Draw the pads
      datayz.Draw();
      dataxy.Draw();

      datayz.cd();
      bg.Draw();
      boxzy.Draw();
      for ( auto& pbadchbox : zy_deadregions ) {
	pbadchbox->Draw();
      }
      
      for (int ich=0; ich<32; ich++) {
	pmtmarkers_v[ich]->Draw();
	chmarkers_v[ich]->Draw();
      }

      gnoncore_zy->Draw("P");
      gcore_zy->Draw("P");
      gfill_zy->Draw("P");
      gcore_zy_pca->Draw("L");      

      dataxy.cd();
      bgxy.Draw();
      boxxy.Draw();

      gnoncore_xy->Draw("P");
      gcore_xy->Draw("P");
      gfill_xy->Draw("P");
      gcore_xy_pca->Draw("L");


      char canvname[50];
      sprintf( canvname, "flashmatch_qclustimg_%02d.png", iclust );
      c2d.SaveAs( canvname );

      // clean up
      delete gnoncore_zy;
      delete gnoncore_xy;      
      delete gcore_zy;
      delete gcore_xy;
      delete gcore_zy_pca;
      delete gcore_xy_pca;      
    }

    // clean up vis items
    for (int ich=0; ich<32; ich++) {
      delete pmtmarkers_v[ich];
      delete chmarkers_v[ich];
    }
    for (int i=0; i<(int)zy_deadregions.size(); i++) {
      delete zy_deadregions[i];
    }
    */
  }


  void LArFlowFlashMatch::dumpQCompositeImages() {

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

      std::vector<TEllipse*> datamarkers_v(32,0);
      float norm = flash.tot;
      for (size_t ich=0; ich<32; ich++) {
	
	int opdet = geo->OpDetFromOpChannel(ich);
	double xyz[3];
	geo->GetOpChannelPosition( ich, xyz );
	
	float pe   = (flash[ich]*norm);
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
	mctrack_data_zy  = new TGraph( mct.size() );
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

      std::vector< TGraph* > mctrack_clustertruth_xy_v;
      std::vector< TGraph* > mctrack_clustertruth_zy_v;      
      for (size_t iclust=0; iclust<_qcluster_v.size(); iclust++) {

	int compat = getCompat( iflash, iclust );
	if ( compat!=0 )
	  continue;

	const QCluster_t& qclust = _qcluster_v[iclust];

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


      c2d.Update();
      c2d.Draw();

      char cname[100];
      sprintf(cname,"qcomposite_flash%02d.png",iflash);
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
  
  void LArFlowFlashMatch::dumpMatchImages( const std::vector<FlashData_t>& flashdata_v, bool shapeonly, bool usefmatch ) {
    // ===================================================
    // Dump images for debug
    // Each image is for a flash
    //  we plot compatible flashes (according to compat matrix,
    //    so it could be fit to the data)
    //  we also plot the best chi2 and best maxdist match
    //  finally, we plot the truth-matched hypothesis
    // ===================================================

    /*
    gStyle->SetOptStat(0);
    
    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();    
    const float  driftv = larp->DriftVelocity();    
    
    TCanvas c("c","",800,400);
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
    TH2D bg("hyz","",105,-20,1050, 120, -130, 130);
    TH2D bgxy("hxy","",25,-20,280, 120, -130, 130);
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
    }
    

    for (int iflash=0; iflash<_nflashes; iflash++) {
      c.Clear();
      c.cd();
      
      int ncompat = 0;

      // get data histogram
      // -------------------
      TH1D hdata("hdata","",32,0,32);
      float norm = 1.;
      if ( !shapeonly ) {
	norm = flashdata_v[iflash].tot;
      }
      for (int ipmt=0; ipmt<32; ipmt++) {
	hdata.SetBinContent( ipmt+1, flashdata_v[iflash][ipmt]*norm );
      }
      hdata.SetLineWidth(4);
      hdata.SetLineColor(kBlack);
      hdata.Draw("hist");
      std::vector< TH1D* > hclust_v;
      int tophistidx = 0;
      float max = hdata.GetMaximum();

      auto it_flash = _flash_reindex.find( iflash );

      bool mctrack_match = false;
      if ( flashdata_v[iflash].mctrackid>=0 )
	mctrack_match = true;


      // record some info for text for canvas
      // int bestchi2_idx = -1;
      // float bestchi2 = -1;
      // float bestchi2_peratio = -1.0;
      // int bestmaxdist_idx = -1;
      // float bestmaxdist = -1;
      // float bestmaxdist_peratio = -1.0;
      // int bestfmatch_idx = -1;
      // float matchscore_best = 0;
      // const FlashHypo_t* bestchi2_hypo = nullptr;
      // const FlashHypo_t* bestmaxdist_hypo = nullptr;
      // const FlashHypo_t* bestfmatch_hypo = nullptr;

      int truthmatch_idx = -1;
      float truthmatch_chi2 = -1;
      float truthmatch_maxdist = -1;
      float truthmatch_peratio = -1;
      const FlashHypo_t* truthmatch_hypo = nullptr;
      TH1D* truthmatch_hist = nullptr;

      for (int iclust=0; iclust<_nqclusters; iclust++) {
	
	bool truthmatched = (flashdata_v[iflash].truthmatched_clusteridx==iclust );
	
	if ( getCompat(iflash,iclust)!=0 && !truthmatched ) {
	  // we skip if deemed unmatchable unless it is the truth-match hypothesis
	  continue;
	}

	auto it_clust = _clust_reindex.find( iclust );
	int imatch = -1;
	if ( it_clust!=_clust_reindex.end() && it_flash!=_flash_reindex.end() )
	  imatch = getMatchIndex( it_flash->second, it_clust->second );
	
	if (usefmatch && !truthmatched) {
	  // we skip based on boor match vector value unless, again, it is the truth-match hypothesis
	  if ( imatch<0 || fmatch[ imatch ] < 0.001 )
	    continue;
	}

	// build hypothesis history
	char hname[20];
	sprintf( hname, "hhypo_%d", iclust);
	TH1D* hhypo = new TH1D(hname, "", 32, 0, 32 );
	hhypo->SetLineWidth(1.0);
	hhypo->SetLineColor(kRed);
	const FlashHypo_t& hypo = getHypothesisWithOrigIndex( iflash, iclust );
	float hypo_norm = 1.;
	if ( !shapeonly ) {
	  hypo_norm = (*flightyield)*hypo.tot;
	  // std::cout << "hypo[fl=" << iflash << ",cl=" << iclust << "]"
	  // 	    << " totalpe=" << hypo_norm
	  // 	    << " intpcpe=" << hypo.tot_intpc*(*flightyield)
	  // 	    << " outtpcpe=" << hypo.tot_outtpc*(*flightyield)
	  // 	    << std::endl;
	}
	for (int ipmt=0;ipmt<32;ipmt++) {
	  hhypo->SetBinContent(ipmt+1,hypo[ipmt]*hypo_norm);
	}
	// pe ratio of hypothesis
	float peratio = fabs(hypo_norm/norm-1.0);
	    
	float fmatchscore = 0.;
	if ( usefmatch && imatch>=0 ) {
	  // we mark the best score
	  fmatchscore = fmatch[imatch];
	  if ( fmatchscore >= matchscore_best || fmatchscore>0.98 ) {
	    // basically the same
	    //float peratio = fabs(hypo_norm/norm-1.0);
	    // if ( peratio < peratio_best ) {
	    //   bestfmatch_idx = iclust;
	    //   matchscore_best = fmatchscore;	      
	    //   bestfmatch_hypo = &hypo;
	    //   peratio_best = peratio;	      
	    //   tophistidx = hclust_v.size();
	    // }
	  }
	}
	
	// set colors/mark
	int ntop=0;
	if ( _flashdata_best_hypo_maxdist_idx.size()==_nflashes && _flashdata_best_hypo_maxdist_idx[iflash]==iclust ) {
	  // best maxdist hypo
	  hhypo->SetLineWidth(2);
	  hhypo->SetLineColor(kBlue);
	  bestmaxdist_idx = iclust;
	  bestmaxdist = _flashdata_best_hypo_maxdist[iflash];
	  bestmaxdist_hypo = &hypo;
	  bestmaxdist_peratio = peratio;
	  ntop++;
	}
	if ( _flashdata_best_hypo_chi2_idx.size()==_nflashes && _flashdata_best_hypo_chi2_idx[iflash]==iclust ) {
	  hhypo->SetLineWidth(2);
	  hhypo->SetLineColor(kCyan);
	  bestchi2_idx = iclust;
	  bestchi2 = _flashdata_best_hypo_chi2[iflash];
	  bestchi2_hypo = &hypo;
	  bestchi2_peratio = peratio;
	  ntop++;
	}

	// both best chi2 and maxdist
	if ( ntop==2 )  {
	  hhypo->SetLineColor(kMagenta);
	}

	if ( truthmatched ) {
	  truthmatch_idx = iclust;
	  truthmatch_hypo = &hypo;
	  truthmatch_peratio = fabs(hypo.tot/flashdata_v[iflash].tot-1.0);
	  hhypo->SetLineColor(kGreen+3);
	  hhypo->SetLineWidth(3);
	  truthmatch_hist = hhypo;
	  if ( getCompat( iflash, iclust)!=0 || ( usefmatch && (imatch<0 || fmatch[imatch]<0.001) ) )
	    hhypo->SetLineStyle(3);
	  if ( usefmatch ) {
	    truthmatch_chi2 = exp( -0.5* *(fmatch_nll+imatch) );
	    truthmatch_maxdist = *(fmatch_maxdist+imatch);
	  }
	  else {
	    // we have to calculate it
	    const FlashData_t& d = flashdata_v[iflash];
	    truthmatch_maxdist = shapeComparison( hypo, d, d.tot, hypo.tot );
	    truthmatch_chi2    = chi2Comparison(  hypo, d, d.tot, hypo.tot );
	  }
	}
	
	hhypo->Draw("hist same");
	// if ( max < hhypo->GetMaximum() )
	//   max = hhypo->GetMaximum();
	hclust_v.push_back( hhypo );
      }//end of cluster loop

      std::cout << "flash[" << iflash << "] bestchi2[" << _flashdata_best_hypo_chi2_idx[iflash] << "]  fromloop[" << bestchi2_idx << "]" << std::endl;

      if ( truthmatch_hypo ) {
	if ( truthmatch_hist->GetMaximum() > hdata.GetMaximum() ) {
	  hdata.SetMaximum( truthmatch_hist->GetMaximum()*1.1 );
	}
      }
      
      c.Update();
      c.Draw();

      // char cname[100];
      // sprintf( cname, "hflashdata_compat_pmtch%02d.png",iflash);
      // std::cout << "saving " << cname << std::endl;
      // c.SaveAs(cname);

      // ============================
      // DRAW 2D PLOT
      // ============================

      // -----------------------  ------
      // |   data flash yz     |  | xy |
      // | truth-matched track |  |    |
      // |                     |  |    |
      // -----------------------  ------
      // -----------------------  ------
      // | best-match flash yz |  | xy |
      // | best-match track    |  |    |
      // | truth-matched flash |  |    |
      // -----------------------  ------
      // -------------------------------
      // |                             |
      // |    PMT channel histogram    |
      // |                             |
      // -------------------------------
      
      // prepare drawing objects      
      //------------------------

      // hypothesis flash
      // ----------------
      const FlashHypo_t* hypo = nullptr;
      int bestmatch_iclust = -1;
      if ( bestmaxdist_idx>=0 ) {
	bestmatch_iclust = bestmaxdist_idx;
	hypo = bestmaxdist_hypo;
      }
      else if ( bestchi2_idx>=0 ) {
	bestmatch_iclust = bestchi2_idx;
	hypo  = bestchi2_hypo;
      }
      
      if ( usefmatch && bestfmatch_hypo ) {
	hypo   = bestfmatch_hypo;
	bestmatch_iclust = bestfmatch_idx;
      }
      
      float hypo_norm = 0.;
      if ( hypo )
	hypo_norm = (*flightyield)*hypo->tot;
      
      // draw pmt data markers
      std::vector<TEllipse*> pmtmarkers_v(32,0);
      std::vector<TText*>    chmarkers_v(32,0);
      std::vector<TEllipse*> datamarkers_v(32,0);
      std::vector<TEllipse*> hypomarkers_v(32,0);
      std::vector<TEllipse*> truthmarkers_v(32,0);
      
      for (int ich=0; ich<32; ich++) {
	int opdet = geo->OpDetFromOpChannel(ich);
	double xyz[3];
	geo->GetOpChannelPosition( ich, xyz );
	float pe = (flashdata_v[iflash][ich]*norm);
	if ( pe>10 )
	  pe = 10 + (pe-10)*0.10;
	float radius = ( pe>50 ) ? 50 : pe;
	datamarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
	datamarkers_v[ich]->SetFillColor(kRed);

	pmtmarkers_v[ich] = new TEllipse(xyz[2],xyz[1], 10.0, 10.0);
	pmtmarkers_v[ich]->SetLineColor(kBlack);

	char pmtname[10];
	sprintf(pmtname,"%02d",ich);
	chmarkers_v[ich] = new TText(xyz[2]-10.0,xyz[1]-5.0,pmtname);
	chmarkers_v[ich]->SetTextSize(0.04);

	// make a hypothesis: use 
	if ( hypo ) {
	  float hypope = (hypo->at(ich)*hypo_norm);
	  if ( hypope>10 )
	    hypope = 10 + (hypope-10)*0.10;
	  float radius = ( hypope>50 ) ? 50 : hypope;
	  hypomarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
	  hypomarkers_v[ich]->SetFillColor(kOrange);
	}//if hypo

	if ( truthmatch_hypo ) {
	  float truthpe = (truthmatch_hypo->at(ich)*truthmatch_hypo->tot);
	  if ( truthpe>10 )
	    truthpe = 10 + (truthpe-10)*0.10;
	  float radius = ( truthpe>50 ) ? 50 : truthpe;
	  truthmarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
	  truthmarkers_v[ich]->SetFillStyle(0);
	  truthmarkers_v[ich]->SetLineColor(kMagenta);
	  truthmarkers_v[ich]->SetLineWidth(3);
	}
	
      }// loop over channels

      // truth-matched charge cluster
      const QCluster_t* qtruth = nullptr;
      if ( flashdata_v[iflash].truthmatched_clusteridx>=0 )
	qtruth = &(_qcluster_v[ flashdata_v[iflash].truthmatched_clusteridx ]);

      // projections for truthmatch cluster
      TGraph* truthclust_zy[ kNumQTypes ]= {nullptr};
      TGraph* truthclust_xy[ kNumQTypes ]= {nullptr};
      int ntruthpts[ kNumQTypes ] = {0};
      if ( qtruth && qtruth->size()>0 ) {
	//std::cout << "qtruth[" << flashdata_v[iflash].truthmatched_clusteridx << "] npoints: " << qtruth->size() << std::endl;
	for (int iqt=0; iqt<kNumQTypes; iqt++) {
	  truthclust_zy[iqt] = new TGraph(qtruth->size());
	  truthclust_xy[iqt] = new TGraph(qtruth->size());
	}
	float xoffset = (flashdata_v[iflash].tpc_tick-3200)*0.5*driftv;
	for (int ipt=0; ipt<(int)qtruth->size(); ipt++) {
	  const QPoint_t& truthq = (*qtruth)[ipt];
	  truthclust_zy[ truthq.type ]->SetPoint(ntruthpts[truthq.type],truthq.xyz[2], truthq.xyz[1] );
	  truthclust_xy[ truthq.type ]->SetPoint(ntruthpts[truthq.type],truthq.xyz[0]-xoffset, truthq.xyz[1] );
	  ntruthpts[truthq.type]++;
	}
	//std::cout << "qtruth[0] = " << qtruth->at(ipt).xyz[0]-xoffset << " (w/ offset=" << xoffset << ")" << std::endl;
	for ( int iqt=0; iqt<kNumQTypes; iqt++ ) {
	  truthclust_zy[iqt]->Set( ntruthpts[iqt] );
	  truthclust_xy[iqt]->Set( ntruthpts[iqt] );

	  truthclust_zy[iqt]->Set( ntruthpts[iqt] );
	  truthclust_xy[iqt]->Set( ntruthpts[iqt] );

	  truthclust_zy[iqt]->SetMarkerSize(0.3);
	  truthclust_zy[iqt]->SetMarkerStyle( 20 );

	  truthclust_xy[iqt]->SetMarkerSize(0.3);
	  truthclust_xy[iqt]->SetMarkerStyle( 20 );
	}
	truthclust_zy[kGapFill]->SetMarkerColor(kRed);	
	truthclust_xy[kGapFill]->SetMarkerColor(kRed);
	truthclust_zy[kExt]->SetMarkerColor(kGreen+3);	
	truthclust_xy[kExt]->SetMarkerColor(kGreen+3);
	truthclust_zy[kNonCore]->SetMarkerColor(kYellow+2);
	truthclust_xy[kNonCore]->SetMarkerColor(kYellow+2);
      }

      // mc-track: of truth match if it exists
      TGraph* mctrack_data    = nullptr;
      TGraph* mctrack_data_xy = nullptr;      
      if ( mctrack_match ) {
	const larlite::mctrack& mct = (*_mctrack_v)[ _mctrackid2index[flashdata_v[iflash].mctrackid] ];
	std::cout << "matchedmctrack[" << flashdata_v[iflash].mctrackid << "] npoints: " << mct.size() << std::endl;	
	mctrack_data    = new TGraph( mct.size() );
	mctrack_data_xy = new TGraph( mct.size() );	
	for (int istep=0; istep<(int)mct.size(); istep++) {
	  mctrack_data->SetPoint(istep, mct[istep].Z(), mct[istep].Y() );
	  mctrack_data_xy->SetPoint(istep, mct[istep].X(), mct[istep].Y() );
	}
	mctrack_data->SetLineColor(kBlue);
	mctrack_data->SetLineWidth(1);
	mctrack_data_xy->SetLineColor(kBlue);
	mctrack_data_xy->SetLineWidth(1);
      }

      // ======================
      // BUILD PLOT
      // ======================
      c2d.Clear();
      c2d.Draw();
      c2d.cd();
      
      // Draw the pads
      datayz.Draw();
      hypoyz.Draw();
      dataxy.Draw();
      hypoxy.Draw();
      histpad.Draw();

      // YZ-2D PLOT: FLASH-DATA/TRUTH TRACK
      // -----------------------------------
      datayz.cd();
      bg.Draw();
      boxzy.Draw();
      for ( auto& pbadchbox : zy_deadregions ) {
	pbadchbox->Draw();
      }

      for (int ich=0; ich<32; ich++) {
	datamarkers_v[ich]->Draw();
	pmtmarkers_v[ich]->Draw();
      }
      
      if ( truthclust_zy[kCore] ) {
	for (int i=0; i<kNumQTypes; i++) 
	  truthclust_zy[i]->Draw("P");
      }
      if ( mctrack_data )
	mctrack_data->Draw("L");

      for (int ich=0; ich<32; ich++)
	chmarkers_v[ich]->Draw();
      
      // XY: DATA
      // --------
      dataxy.cd();
      bgxy.Draw();
      boxxy.Draw();
      if ( truthclust_xy[kCore] ) {
	for (int i=0; i<kNumQTypes; i++)
	  truthclust_xy[i]->Draw("P");
      }
      if ( mctrack_data_xy )
	mctrack_data_xy->Draw("L");
      
      // YZ-2D plots: hypothesis
      // -----------------------
      hypoyz.cd();
      bg.Draw();
      boxzy.Draw();
      for ( auto& pbadchbox : zy_deadregions ) {
	pbadchbox->Draw();
      }      

      for (int ich=0; ich<32; ich++) {
	if (hypo) 
	  hypomarkers_v[ich]->Draw();
	pmtmarkers_v[ich]->Draw();
	chmarkers_v[ich]->Draw();
	if ( truthmatch_hypo )
	  truthmarkers_v[ich]->Draw();
      }
      
      TGraph* bestmatchclust_zy[kNumQTypes] = {nullptr};
      TGraph* bestmatchclust_xy[kNumQTypes] = {nullptr};
      TGraph* mctrack_hypo_zy = nullptr;
      TGraph* mctrack_hypo_xy = nullptr;      
	
      if ( bestmatch_iclust>=0 ) {
	// if we have a best match

	// make graph for hypothesis cluster
	int nbestpts[kNumQTypes] = {0};
	QCluster_t* qc = &(_qcluster_v[bestmatch_iclust]);
	//std::cout << "qc[" << iclust << "] npoints: " << qc->size() << std::endl;
	for (int i=0; i<kNumQTypes; i++) {
	  bestmatchclust_zy[i] = new TGraph(qc->size());
	  bestmatchclust_xy[i] = new TGraph(qc->size());
	  
	  bestmatchclust_zy[i]->SetMarkerSize(0.3);
	  bestmatchclust_xy[i]->SetMarkerSize(0.3);

	  bestmatchclust_zy[i]->SetMarkerStyle(20);
	  bestmatchclust_xy[i]->SetMarkerStyle(20);
	  
	}
	float xoffset = (flashdata_v[iflash].tpc_tick-3200)*0.5*driftv;	
	for (int ipt=0; ipt<(int)qc->size(); ipt++) {
	  const QPoint_t& bestq = (*qc)[ipt];
	  bestmatchclust_zy[bestq.type]->SetPoint(nbestpts[bestq.type],bestq.xyz[2], bestq.xyz[1] );
	  bestmatchclust_xy[bestq.type]->SetPoint(nbestpts[bestq.type],bestq.xyz[0]-xoffset, bestq.xyz[1] );
	  nbestpts[bestq.type]++;
	}
	for (int i=0; i<kNumQTypes; i++) {
	  bestmatchclust_zy[i]->Set(nbestpts[i]);
	  bestmatchclust_xy[i]->Set(nbestpts[i]);
	}
	
	bestmatchclust_zy[kGapFill]->SetMarkerColor(kRed);	
	bestmatchclust_xy[kGapFill]->SetMarkerColor(kRed);

	bestmatchclust_zy[kExt]->SetMarkerColor(kGreen+3);	
	bestmatchclust_xy[kExt]->SetMarkerColor(kGreen+3);

	bestmatchclust_zy[kNonCore]->SetMarkerColor(kYellow+2);
	bestmatchclust_xy[kNonCore]->SetMarkerColor(kYellow+2);

	for (int i=0; i<kNumQTypes; i++) {
	  bestmatchclust_zy[i]->Draw("P");
	}

	// make graph for mctruth for hypothesis
	if ( bestmatch_iclust==truthmatch_idx ) {
	  mctrack_hypo_zy = mctrack_data;
	  mctrack_hypo_xy = mctrack_data_xy;
	}
	else {
	  const larlite::mctrack& mct = (*_mctrack_v)[ _mctrackid2index[qc->mctrackid] ];
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
	}
	
      }// if has best match

      
      if ( mctrack_hypo_zy )
	mctrack_hypo_zy->Draw("L");
      
      hypoxy.cd();
      bgxy.Draw();
      boxxy.Draw();
      if ( bestmatch_iclust>=0 ) {
      	for (int i=0; i<kNumQTypes; i++) {
	  bestmatchclust_xy[i]->Draw("P");
	}
      }
      if ( mctrack_hypo_xy )
	mctrack_hypo_xy->Draw("L");      

      // finally hist pad
      histpad.cd();
      hdata.Draw("hist");
      for (int ihist=0; ihist<(int)hclust_v.size(); ihist++) {
	hclust_v[ihist]->Draw("hist same");
      }
      if ( tophistidx<hclust_v.size() )
	hclust_v[tophistidx]->Draw("hist same");


      // text summary
      char ztruth[100];
      if ( truthmatch_idx<0 )
	sprintf( ztruth,"No truth-match");
      else
	sprintf( ztruth,"Truth-match idx (green): %d",truthmatch_idx );

      char ztruthscores[100];
      if ( truthmatch_idx<0 )
	sprintf( ztruthscores, "Truth-Chi2=NA  Truth-Maxdist=NA" );
      else
	sprintf( ztruthscores, "Truth-Chi2=%.2f  Truth-Maxdist=%.2f peratio=%.2f", truthmatch_chi2, truthmatch_maxdist, truthmatch_peratio );

      char zbestchi[100];
      if ( bestchi2_idx>=0 )
	sprintf( zbestchi, "Best Chi2 idx (cyan): %d  Chi2=%.1f peratio=%.2f", bestchi2_idx, bestchi2, bestchi2_peratio );
      else
	sprintf( zbestchi, "No chi2 match" );

      char zbestmaxdist[100];
      if ( bestmaxdist_idx>=0 )
	sprintf( zbestmaxdist, "Best maxdist idx (blue): %d  maxdist=%.2f peratio=%.2f", bestmaxdist_idx, bestmaxdist, bestmaxdist_peratio );
      else
	sprintf( zbestmaxdist, "No maxdist match" );
      
      char zbestfmatch[100];
      if ( usefmatch )
	sprintf( zbestfmatch, "Best fmatch index: %d fmatch=%.2f", bestfmatch_idx, matchscore_best );

      TText ttruth(0.6,0.85,ztruth);
      TText ttruthscore(0.6,0.80,ztruthscores);
      TText tbestchi(0.6,0.75,zbestchi);
      TText tbestmaxdist(0.6,0.70,zbestmaxdist);
      TText* tbestfmatch = nullptr;
      if ( usefmatch )
	tbestfmatch = new TText(0.6,0.65,zbestfmatch);

      ttruth.SetNDC(true);
      ttruth.Draw();
      ttruthscore.SetNDC(true);      
      ttruthscore.Draw();
      tbestchi.SetNDC(true);
      tbestchi.Draw();
      tbestmaxdist.SetNDC(true);
      tbestmaxdist.Draw();
      if ( usefmatch ) {
	tbestfmatch->SetNDC(true);
	tbestfmatch->Draw();
      }
      
      c2d.Update();
      char cname[100];
      sprintf(cname,"hflash2d_flashid%d.png",iflash);
      c2d.SaveAs(cname);

      std::cout << "clean up" << std::endl;
      
      for ( auto& pmarker : pmtmarkers_v ) {
	delete pmarker;
	pmarker = nullptr;
      }
      for ( auto& pmarker : datamarkers_v ) {
	delete pmarker;
	pmarker = nullptr;
      }
      for ( auto& pmarker : hypomarkers_v ) {
	delete pmarker;
	pmarker = nullptr;
      }
      for (int i=0; i<kNumQTypes; i++) {
	if ( truthclust_zy[i] )
	     delete truthclust_zy[i];
	if ( truthclust_xy[i] )
	  delete truthclust_xy[i];
	if ( bestmatchclust_zy[i] )
	     delete bestmatchclust_zy[i];
	if ( bestmatchclust_xy[i] )
	  delete bestmatchclust_xy[i];
      }
      
      if (mctrack_data)
	delete mctrack_data;
      if ( mctrack_data_xy )
	delete mctrack_data_xy;

      if ( mctrack_hypo_zy && bestmatch_iclust!=truthmatch_idx ) {
	delete mctrack_hypo_zy;
	delete mctrack_hypo_xy;
      }

      for (int ic=0; ic<(int)hclust_v.size(); ic++) {
	delete hclust_v[ic];
      }
      hclust_v.clear();      

      delete tbestfmatch;
      tbestfmatch = nullptr;
      
    } //end of flash loop
    */
  }

  float LArFlowFlashMatch::calcNLL( bool print ) {


    // calculate agreement
    float nll_data = 0.;

    // fit to data
    for (int imatch=0; imatch<_nmatches; imatch++) {
      float hyponorm = *(m_flashhypo_norm + imatch)*(*flightyield);
      float datanorm = *(m_flashdata_norm + imatch);

      float nll_match = 0.;
      float maxdist_match = 0;
      float cdf_data[32] = {0.};
      float cdf_pred[32] = {0.};
      for (int ich=0; ich<32; ich++) {
	float pred = *(m_flash_hypo + imatch*32 + ich );
	float obs  = *(m_flash_data + imatch*32 + ich );
	pred *= hyponorm;
	obs  *= datanorm;
	if ( m_iscosmic[imatch]==1 && pred<_fCosmicDiscThreshold )
	  pred = 1.0e-3;
	if ( pred<1.0e-3 )
	  pred = 1.0e-3;

	float nll_bin = (pred-obs);
	if ( obs>0 )
	  nll_bin += obs*(log(obs)-log(pred));
	//std::cout << "[" << imatch << "][" << ich << "] pred=" << pred << " obs=" << obs << " nllbin=" << nll_bin << std::endl;
	nll_match += 2.0*nll_bin;

	if ( ich==0 )
	  cdf_data[ich] = obs;
	else
	  cdf_data[ich] = cdf_data[ich-1] + obs;
	
	if ( ich==0 )
	  cdf_pred[ich] = pred;
	else
	  cdf_pred[ich] = cdf_pred[ich-1] + pred;

	float cdf_dist = 0.;
	if ( (datanorm==0 && hyponorm!=0 ) || (datanorm!=0 && hyponorm==0 ) )
	  cdf_dist = 1.0;
	else 
	  cdf_dist = fabs( cdf_data[ich]/datanorm - cdf_pred[ich]/hyponorm );
	if ( cdf_dist > maxdist_match )
	  maxdist_match = cdf_dist;
      }

      // store values
      *(fmatch_nll + imatch) = nll_match;
      *(fmatch_maxdist + imatch) = maxdist_match;
      
      // accumulate for system
      nll_data += nll_match*fmatch[imatch];
    }

    // constraints:
    float nll_clustsum = 0.;
    //for each cluster. sum of pairs to flashes should be 1
    for (int iclust=0; iclust<_nclusters_red; iclust++) {
      float clustsum = 0.;
      for (int iflash=0; iflash<_nflashes_red; iflash++) {
	int imatch = getMatchIndex( iflash, iclust );
	if ( imatch>=0 )
	  clustsum += *(fmatch+ imatch );
      }
      nll_clustsum += (clustsum-1.0)*(clustsum-1.0);
      //std::cout << "clustsum[" << iclust << "] " << clustsum << std::endl;
    }

    float nll_flashsum = 0.;
    //for each flash. sum of pairs to flashes should be 1 for the most part
    for (int iflash=0; iflash<_nflashes_red; iflash++) {
      float flashsum = 0.;
      for (int iclust=0; iclust<_nclusters_red; iclust++) {
	int imatch = getMatchIndex( iflash, iclust );
	if ( imatch>=0 )
	  flashsum += *(fmatch+ imatch );
      }
      nll_flashsum += (flashsum-1.0)*(flashsum-1.0);
      //std::cout << "clustsum[" << iclust << "] " << clustsum << std::endl;
    }
    
    // L1 norm: enforce sparsity
    float nll_l1norm = 0;
    for (int imatch=0; imatch<_nmatches; imatch++) {
      nll_l1norm += fabs(fmatch[imatch]);
    }

    // lightyield prior
    float nll_ly = (*flightyield - _fweighted_scalefactor_mean)*(*flightyield - _fweighted_scalefactor_mean)/_fweighted_scalefactor_var;

    float nll = nll_data + _fclustsum_weight*nll_clustsum + _fflashsum_weight*nll_flashsum + _fl1norm_weight*nll_l1norm + _flightyield_weight*nll_ly;
    if ( print )
      std::cout << "NLL(ly=" << *flightyield << "): nll_tot=" << nll
		<< ":  nll_data=" << nll_data
		<< " + nll_clustsum=" << nll_clustsum
		<< " + nll_flashsum=" << nll_flashsum	
		<< " + l1norm=" << nll_l1norm
		<< " nll_ly=" << nll_ly << "" << std::endl; 
    
    return nll;
  }

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

  // ==============================================================================================
  // MC TRUTH FUNCTIONS
  // ==============================================================================================
  
  void LArFlowFlashMatch::loadMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v, bool do_truth_matching ) {
    _mctrack_v = &mctrack_v;
    kDoTruthMatching = do_truth_matching;
    kFlashMatchedDone = false;
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
      std::cout << "mctrack[" << mct.TrackID() << ",origin=" << mct.Origin() << "] pdg=" << mct.PdgCode()
      		<< " E=" << mct.Start().E() 
      		<< " T()=" << mct.Start().T() << " track_tick=" << track_tick << " optick=" << flash_time
      		<< " best_flashidx="  << best_flashidx << " "
      		<< " best_dtick=" << best_dtick
      		<< " nmatched=" << nmatch << std::endl;
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
	  std::cout << "  multiple-truthmatches[" << i << "] id=" << id[i] << " mid=" << mid[i] << " pdg=" << pdg[i] << " dz=" << dz[i] << " isnu=" << trackisnu << std::endl;
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
      std::cout << "FlashMCtrackMatch[" << iflash << "] "
		<< "tick=" << flashdata_v[iflash].tpc_tick << " "
		<< "nmatches=" << id.size() << " "
		<< "trackid=" << flashdata_v[iflash].mctrackid << " "
		<< "pdg=" << flashdata_v[iflash].mctrackpdg << " "
		<< "isnu=" << flashdata_v[iflash].isneutrino << " "
		<< "intime=" << flashdata_v[iflash].intime << " "
		<< "isbeam=" << flashdata_v[iflash].isbeam << " "			
		<< "nmcpts=" << nmcpts << std::endl;
    }
    kFlashMatchedDone = true;
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

  void LArFlowFlashMatch::buildClusterExtensionsWithMCTrack( bool appendtoclusters, std::vector<QCluster_t>& qcluster_v ) {
    // we create hits outside the tpc for each truth-matched cluster
    // outside of TPC, so no field, no space-charge correction
    // -----
    // 1) we find when we first cross the tpc and exit
    // 2) at these points we draw a line of charge using the direction at the crossing point

    const float maxstepsize=1.0; // (10 pixels)
    const float maxextension=30.0; // 1 meters (we stop when out of cryostat)
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float  driftv = larp->DriftVelocity();    
    
    for ( int iclust=0; iclust<(int)qcluster_v.size(); iclust++) {
      QCluster_t& cluster = qcluster_v[iclust];
      if ( cluster.mctrackid<0 )
	continue;
      const larlite::mctrack& mct = (*_mctrack_v).at( _mctrackid2index[cluster.mctrackid] );

      int nsteps = (int)mct.size();
      QCluster_t ext1;
      QCluster_t ext2;
      ext1.reserve(200);
      ext2.reserve(200);
      bool crossedtpc = false;
      
      for (int istep=0; istep<nsteps-1; istep++) {

	const larlite::mcstep& thisstep = mct[istep];	
	const larlite::mcstep& nextstep = mct[istep+1];

	double thispos[4] = { thisstep.X(), thisstep.Y(), thisstep.Z(), thisstep.T() };
	double nextpos[4] = { nextstep.X(), nextstep.Y(), nextstep.Z(), nextstep.T() };

	  // are we in the tpc?
	bool intpc = false;
	if ( nextpos[0]>0 && nextpos[0]<256 && nextpos[1]>-116.5 && nextpos[1]<116.5 && nextpos[2]>0 && nextpos[2]<1036.0 )
	  intpc = true;
	
	if ( (intpc && !crossedtpc) || (!intpc && crossedtpc) ) {
	  // entering point/make extension

	  // make extension
	  double dirstep[4];	
	  double steplen = maxstepsize;
	  int nsubsteps = maxextension/steplen;
	  
	  for (int i=0; i<4; i++) {
	    dirstep[i] = nextpos[i]-thispos[i];
	    if ( i<3 )
	      steplen += dirstep[i]*dirstep[i];
	  }
	  steplen = sqrt(steplen);
	  for (int i=0; i<3; i++)
	    dirstep[i] /= steplen;
	  double stepdtick = dirstep[4]/nsubsteps;
	  
	  double* extpos = nullptr;
	  if ( !crossedtpc ) {
	    // entering point, we go backwards
	    for (int i=0; i<3; i++) dirstep[i] *= -1;
	    extpos = nextpos;
	  }
	  else {
	    extpos = thispos;
	  }
	  
	  for (int isub=0; isub<nsubsteps; isub++) {
	    double pos[4];
	    for (int i=0; i<3; i++) 
	      pos[i] = extpos[i] + (double(isub)+0.5)*steplen*dirstep[i];
	    pos[3] = extpos[3] + ((double)isub+0.5)*stepdtick;

	    // once pos is outside cryostat, stop
	    if ( pos[0]<-25 ) // past the plane of the pmts
	      break;
	    if ( pos[0]>260 ) // visibilty behind the cathode seems fucked up
	      break;
	    if ( pos[1]>150.0 )
	      break;
	    if ( pos[1]<-150.0 )
	      break;
	    if ( pos[2] < -60 )
	      break;
	    if ( pos[2] > 1100 )
	      break;
	    
	    // determine which ext
	    QCluster_t& ext = (crossedtpc) ? ext2 : ext1;
	    QPoint_t qpt;
	    qpt.xyz.resize(3,0);
	    qpt.xyz[0] = pos[0];
	    qpt.xyz[1] = pos[1];
	    qpt.xyz[2] = pos[2];
	    qpt.tick   = (pos[3]*1.0e-3)/0.5 + 3200;
	    // x-offset from tick relative to trigger, then actual x-depth
	    // we do this, because reco will be relative to cluster which has this x-offset
	    qpt.xyz[0] = (qpt.tick-3200)*0.5*driftv + pos[0];
	    qpt.pixeladc = steplen; // here we leave distance. photon hypo must know what to do with this
	    if ( pos[0]>250 ) {
	      // this is a hack.
	      // we want the extension points to the gap filler works.
	      // but the visi behind cathode is messed up so we want no contribution
	      qpt.pixeladc = 0.;
	    }
	    qpt.fromplaneid = -1;
	    qpt.type = kExt;
	    ext.emplace_back( std::move(qpt) );
		    
	  }//end of extension
	  crossedtpc = true; // mark that we have crossed	  
	}
      }//end of step loop

      // what to do with these?
      if ( appendtoclusters ) {
	for (auto& qpt : ext1 ) {
	  cluster.emplace_back( std::move(qpt) );
	}
	for (auto& qpt : ext2 ) {
	  cluster.emplace_back( std::move(qpt) );
	}
      }
      else {
	if ( ext1.size()>0 )
	  qcluster_v.emplace_back( std::move(ext1) );
	if ( ext2.size()>0 )
	  qcluster_v.emplace_back( std::move(ext2) );
      }

      // for debug: cluster check =========================
      // std::cout << "extendcluster[" << iclust << "] --------------------------------------" << std::endl;
      // for (auto const& hit : cluster ) {
      // 	std::cout << "  (" << hit.xyz[0] << "," << hit.xyz[1] << "," << hit.xyz[2] << ") intpc=" << hit.intpc << std::endl;
      // }
      // std::cout << "[enter for next]" << std::endl;
      // std::cin.get();
      // ==================================================
      
    }// end of cluster loop
  }

  void LArFlowFlashMatch::clearMCTruthInfo() {
    _flash_truthid.clear();
    _cluster_truthid.clear();
    _flash2truecluster.clear();
    _cluster2trueflash.clear();
    delete _psce;
    _psce = nullptr;
    kDoTruthMatching = false;
  }

  std::vector<larlite::larflowcluster> LArFlowFlashMatch::exportMatchedTracks() {
    // for each cluster, we use the best matching flash
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float  usec_per_tick = 0.5; // usec per tick
    const float  tpc_trigger_tick = 3200;
    const float  driftv = larp->DriftVelocity();
    
    std::vector<larlite::larflowcluster> lfcluster_v(_qcluster_v.size());
    
    for (int iclust=0; iclust<_qcluster_v.size(); iclust++) {
      const QCluster_t& cluster = _qcluster_v[iclust];
      larlite::larflowcluster& lfcluster = lfcluster_v[iclust];
      lfcluster.resize(cluster.size());
      
      // is it flash matched?
      std::vector<float> matchscores = getMatchScoresForCluster(iclust);
      float maxfmatch = 0.;
      int   maxfmatch_idx = -1;      
      for (int iflash=0; iflash<_flashdata_v.size(); iflash++) {
	if ( matchscores[iflash]>maxfmatch ) {
	  maxfmatch = matchscores[iflash];
	  maxfmatch_idx = iflash;
	}
      }

      float matched_flash_tick = 0;
      float matched_flash_xoffset = 0;
      if ( maxfmatch_idx>=0 ) {
	matched_flash_tick    = _flashdata_v[maxfmatch_idx].tpc_tick;
	matched_flash_xoffset = (matched_flash_tick-tpc_trigger_tick)*usec_per_tick/driftv;
      }
      
      // transfer hit locations back to larlite
      
      int nhits=0;
      for (int ihit=0; ihit<(int)cluster.size(); ihit++) {
	const QPoint_t& qpt = cluster.at(ihit);
	larlite::larflow3dhit& lfhit = lfcluster[ihit];
	lfhit.resize(3,0);
	for (int i=0; i<3; i++)
	  lfhit[i] = qpt.xyz[i];
	lfhit.tick = qpt.tick;

	if ( maxfmatch_idx>=0 ) {
	  // match found, we shift the x positions
	  lfhit[0]   -= matched_flash_xoffset;
	  lfhit.tick -= matched_flash_tick; // now this is delta-t from t0
	  // since we have this position, we can invert spacecharge effect
	  // TODO
	}
	bool hitok = true;
	for (int i=0; i<3; i++)
	  if ( std::isnan(lfhit[i]) )
	    hitok = false;

	if (hitok)
	  nhits++;
	
	// here we should get larflow3dhit metadata and pass it on TODO
      }//end of hit loop

      // this is in case we learn how to filter out points
      // then we throw away excess capacity
      lfcluster.resize(nhits); 

      // metadata
      // --------

      // matched flash (reco)
      if ( maxfmatch_idx>=0 ) {
	lfcluster.isflashmatched = 1;
	lfcluster.flash_tick = matched_flash_tick;
      }
      else {
	lfcluster.isflashmatched = -1;
	lfcluster.flash_tick = -1;
      }

      // matched flash (truth)
      lfcluster.truthmatched_mctrackid = cluster.mctrackid;
      if ( cluster.truthmatched_flashidx>=0 )
	lfcluster.truthmatched_flashtick = _flashdata_v[cluster.truthmatched_flashidx].tpc_tick;
      else
	lfcluster.truthmatched_flashtick = -1;
      
    }//end of cluster loop

    return lfcluster_v;
  }

  void LArFlowFlashMatch::setFitParsWithTruthMatch() {
    if ( !kFlashMatchedDone ) {
      throw std::runtime_error("[larflow::LArFlowFlashMatch::setFitParsWithTruthMatch][ERROR] Truth-based flash-matching not yet done.");
    }

    zeroMatchVector();
    
    for (int iflash=0; iflash<(int)_flashdata_v.size(); iflash++) {
      // does it have a true match?
      if ( _flashdata_v[iflash].truthmatched_clusteridx<0 )
	continue;

      FlashData_t& flashdata = _flashdata_v[iflash];      
      int iclust   = flashdata.truthmatched_clusteridx;
      
      // is there a match?
      if ( !doOrigIndexPairHaveMatch( iflash, iclust ) )
	continue;

      int reflash_idx = _flash_reindex[iflash];
      int reclust_idx = _clust_reindex[iclust];
      int imatch = getMatchIndex( reflash_idx, reclust_idx );
      
      *(fmatch + imatch) = 1.0;
    }
  }

  // ---------------------------------------------------------------------
  // second match refinement
  // ---------------------------------------------------------------------
  void LArFlowFlashMatch::secondMatchRefinement() {
    // we take the flashmatchcandidate objects in m_matchcandidate_hypo_v and
    // reject matches based on
    // 1) pe ratio
    // 2) maxdist
    
    int matchidx = 0;
    int flashidx = 0;
    int clustidx = 0;
    for ( auto& matchcandidate : m_matchcandidate_hypo_v ) {
      
      matchcandidate.getFlashClusterIndex( flashidx, clustidx );

      if ( _flashdata_v[flashidx].mctrackid>=0 ) {
	if ( matchcandidate.isTruthMatch() )
	_truthmatch = 1;
	else
	  _truthmatch = 0;
      }
      else
	_truthmatch = -1;

      FlashHypo_t match_wext = matchcandidate.getHypothesis( true,  true, 10.0 );
      FlashHypo_t match_orig = matchcandidate.getHypothesis( false, true, 10.0 );
      
      _maxdist_wext = FlashMatchCandidate::getMaxDist( *(matchcandidate._flashdata),  match_wext );
      _peratio_wext = FlashMatchCandidate::getPERatio( *(matchcandidate._flashdata),  match_wext );
      _maxdist_orig = FlashMatchCandidate::getMaxDist( *(matchcandidate._flashdata),  match_orig );
      _peratio_orig = FlashMatchCandidate::getPERatio( *(matchcandidate._flashdata),  match_orig );

      _maxdist_red2 = ( _maxdist_wext < _maxdist_orig ) ? _maxdist_wext : _maxdist_orig;
      _peratio_red2 = ( _peratio_wext < _peratio_orig ) ? _peratio_wext : _peratio_orig;

      _isbeam = (_flashdata_v[flashidx].isbeam) ? 1 : 0;
      _isneutrino = (_qcluster_v[clustidx].isneutrino) ? 1 : 0;
      _intime = (_flashdata_v[flashidx].intime) ? 1 : 0;
      

      if ( fabs(_peratio_red2)>0.5 ) {
	_redstep = 2;
	setCompat( flashidx, clustidx, 5 );
      }
      else if ( _maxdist_red2>0.25 ) {
	_redstep = 2;	
	setCompat( flashidx, clustidx, 6 );
      }
      else {
	_redstep = 3;
      }

      if (_save_ana_tree)
	_anatree->Fill();
      
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
    _anatree->Branch("redstep",      &_redstep,      "redstep/I");
    _anatree->Branch("truthmatch",   &_truthmatch,   "truthmatch/I");
    _anatree->Branch("isneutrino",   &_isneutrino,   "isneutrino/I");        
    _anatree->Branch("intime",       &_intime,       "intime/I");        
    _anatree->Branch("isbeam",       &_isbeam,       "isbeam/I");
    _anatree->Branch("hypope",       &_hypope,       "hypope/F");
    _anatree->Branch("datape",       &_datape,       "datape/F");    
    _anatree->Branch("maxdist_orig", &_maxdist_orig, "maxdist_orig/F");
    _anatree->Branch("peratio_orig", &_peratio_orig, "peratio_orig/F");
    _anatree->Branch("maxdist_wext", &_maxdist_wext, "maxdist_wext/F");
    _anatree->Branch("peratio_wext", &_peratio_wext, "peratio_wext/F");
    _anatree->Branch("maxdist",      &_maxdist_red2, "maxdist/F");
    _anatree->Branch("peratio",      &_peratio_red2, "peratio/F");
    _save_ana_tree   = true;
    _anafile_written = false;    
  }

  void LArFlowFlashMatch::clearAnaVariables() {
    _redstep    = -1;
    _truthmatch = -1;
    _isneutrino = 0;
    _intime     = 0;
    _isbeam     = 0;
    _hypope     = 0;
    _datape     = 0;
    _maxdist_orig = 0;
    _peratio_orig = 0;
    _maxdist_wext = 0;
    _peratio_wext = 0;
    _maxdist_red2 = 0;
    _peratio_red2 = 0;
  }
  
  void LArFlowFlashMatch::writeAnaFile() {
    if ( _save_ana_tree ) {
      _fanafile->cd();
      _anatree->Write();
      _anafile_written = true;
    }
  }
}
