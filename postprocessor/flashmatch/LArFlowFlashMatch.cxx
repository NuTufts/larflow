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
      kFlashMatchedDone(false)
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
  
  LArFlowFlashMatch::Results_t LArFlowFlashMatch::match( const std::vector<larlite::opflash>& beam_flashes,
							 const std::vector<larlite::opflash>& cosmic_flashes,
							 const std::vector<larlite::larflowcluster>& clusters,
							 const std::vector<larcv::Image2D>& img_v,
							 const bool ignorelast ) {
    
    // first is to build the charge points for each cluster
    _qcluster_v.clear();
    // we ignore last cluster because sometimes we put store unclustered hits in that entry
    if ( ignorelast )
      _qcluster_v.resize( clusters.size()-1 );
    else
      _qcluster_v.resize( clusters.size() );

    // we build up the charge clusters that are easy to grab
    buildInitialQClusters( clusters, _qcluster_v, img_v, 2, ignorelast );

    // collect the flashes
    _flashdata_v.clear();
    _flashdata_v = collectFlashInfo( beam_flashes, cosmic_flashes );
    
    // MC matching
    if ( kDoTruthMatching && _mctrack_v!=nullptr ) {
      std::cout << "[LArFlowFlashMatch::match][INFO] Doing MCTrack truth-reco matching" << std::endl;
      doFlash2MCTrackMatching( _flashdata_v );
      doTruthCluster2FlashTruthMatching( _flashdata_v, _qcluster_v );
      bool appendtoclusters = true;
      buildClusterExtensionsWithMCTrack(appendtoclusters, _qcluster_v );
    }

    // modifications to fill gaps
    applyGapFill( _qcluster_v );

    // we have to build up charge in dead regions in the Y-plane
    // [TODO]
    
    // also collect charge from pixels within the track
    //  that might have been flowed to the wrong cluster and not apart of the correct cluster.
    //  we do this using a neighbor fill (if # of pixels around pixel belong to given cluster
    //  make a qpoint
    // [TODO]


    std::cout << "Number of data flashes: " << _flashdata_v.size() << std::endl;
    std::cout << "Number of clusters: " << _qcluster_v.size() << std::endl;
    
    // build compbatility matrix
    buildFullCompatibilityMatrix( _flashdata_v, _qcluster_v );
    
    // now build hypotheses: we only do so for compatible pairs
    buildFlashHypotheses( _flashdata_v, _qcluster_v );
    
    // refined compabtibility: incompatible-z
    bool adjust_pe_for_cosmic_disc = true;
    reduceMatchesWithShapeAnalysis( _flashdata_v, _qcluster_v, adjust_pe_for_cosmic_disc );
    
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
  }

  // ==============================================================================
  // CHARGE CLUSTER TOOLS
  // -------------------------------------------------------------------------------

  void LArFlowFlashMatch::buildInitialQClusters( const std::vector<larlite::larflowcluster>& lfclusters, std::vector<QCluster_t>& qclusters,
						 const std::vector<larcv::Image2D>& img_v, const int src_plane, bool ignorelast ) {

    if ( ignorelast ) {
      if ( qclusters.size()!=lfclusters.size()-1 )
	qclusters.resize( lfclusters.size()-1 );
    }
    else {
      if ( qclusters.size()!=lfclusters.size() )
	qclusters.resize( lfclusters.size() );
    }

    const larcv::ImageMeta& src_meta = img_v[src_plane].meta();

    int nclusters = lfclusters.size();
    if ( ignorelast ) nclusters -= 1;
    for ( size_t icluster=0; icluster<nclusters; icluster++ ) {

      const larlite::larflowcluster& lfcluster = lfclusters[icluster];
      
      QCluster_t& qcluster = qclusters[icluster];
      qcluster.reserve( lfcluster.size() );
      for ( size_t i=0; i<3; i++) {
	qcluster.min_tyz[i] =  1.0e9;
	qcluster.max_tyz[i] = -1.0e9;
      }

      // store mctrackids
      std::map<int,int> mctrackid_counts;
      
      for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
	QPoint_t qhit;
	for (size_t i=0; i<3; i++)
	  qhit.xyz[i] = lfcluster[ihit][i];
	qhit.tick     = lfcluster[ihit].tick;
	qhit.intpc = 1; // true by definition for larflow-reco tracks
	qhit.gapfill = 0;

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
      
      //std::cout << "[qcluster " << icluster << "] tick range: " << qcluster.min_tyz[0] << "," << qcluster.max_tyz[0] << std::endl;
    }//end of cluster loop
    
  }

  void LArFlowFlashMatch::applyGapFill( std::vector<QCluster_t>& qcluster_v ) {
    for ( auto& cluster : qcluster_v )
      fillClusterGaps( cluster );
  }
  
  void LArFlowFlashMatch::fillClusterGaps( QCluster_t& cluster ) {
    // we sort the points in z, y, x
    // look for gaps. then look for points
    // to be less sensitive to noise,
    //   we require points to have N neighbors within L distance
    //   we use cilantro to help us build and query a fast kD tree

    const float kGapmin[3] = { 6.0, 6.0, 6.0 }; // (20 wires)
    const float kMaxStepSize = 2.0;
    
    struct pt_t {
      float x; // projected dim
      int idx; // index to cluster array
      bool operator<(const pt_t& rhs ) {
	if ( x < rhs.x ) return true;
	return false;
      }
    };

    std::vector< pt_t > dimx(cluster.size());
    std::vector< pt_t > dimy(cluster.size());
    std::vector< pt_t > dimz(cluster.size());
    for ( size_t ic=0; ic<cluster.size(); ic++) {
      dimx[ic].idx = ic;
      dimy[ic].idx = ic;
      dimz[ic].idx = ic;      
      dimx[ic].x = cluster[ic].xyz[0];
      dimy[ic].x = cluster[ic].xyz[1];
      dimz[ic].x = cluster[ic].xyz[2];      
    }
    std::sort( dimx.begin(), dimx.end() );
    std::sort( dimy.begin(), dimy.end() );
    std::sort( dimz.begin(), dimz.end() );

    // for debug
    // std::cout << "Sorted X ----------" << std::endl;
    // for (auto& pt : dimx )
    //   std::cout << " [" << pt.idx << "] " << pt.x << std::endl;
    // std::cout << "[enter to continue]" << std::endl;
    // std::cin.get();
    
    // we collect possible gaps
    struct gap_t {
      float len;      
      int dim;
      float start;
      float end;
      int start_idx;
      int end_idx;
      bool operator<( const gap_t& rhs ) {
	if ( len<rhs.len) return true;
	return false;
      }
    };
    
    std::vector<gap_t> gaps;
    std::vector<pt_t>* dim_v[3] = { &dimx, &dimy, &dimz };
    
    for (int idim=0; idim<3; idim++) {
      for (size_t i=1; i<dim_v[idim]->size(); i++) {
	float dx = (*dim_v[idim])[i].x - (*dim_v[idim])[i-1].x;
	if ( dx> kGapmin[idim] ) {
	  gap_t g;
	  g.len = dx;
	  g.dim = idim;
	  g.start = (*dim_v[idim])[i-1].x;
	  g.end   = (*dim_v[idim])[i].x;
	  g.start_idx = (*dim_v[idim])[i-1].idx;
	  g.end_idx   = (*dim_v[idim])[i].idx;
	  gaps.emplace_back(std::move(g));
	}
      }
    }
    std::cout << "number of gaps in cluster: " << gaps.size() << std::endl;
    std::sort( gaps.begin(), gaps.end() );

    // record where we fill gaps already -- forbid it
    std::vector<float> gapends[3];

    // fill the gap between the two points
    for ( auto& gap : gaps ) {
      int dim = gap.dim;
      QPoint_t& start = cluster[gap.start_idx];
      QPoint_t& end   = cluster[gap.end_idx];

      // we check if we've already filled in this neighborhood
      bool alreadyfilled = false;
      for (int idim=0; idim<3; idim++) {
	for (auto& endpos : gapends[idim] ) {
	  float dist = fabs(start.xyz[idim]-endpos);
	  if ( dist<kGapmin[idim] ) {
	    alreadyfilled = true;
	    break;
	  }
	}
	if ( alreadyfilled )
	  break;
      }

      if ( alreadyfilled )
	continue;

      for (int idim=0; idim<3; idim++) {
	gapends[idim].push_back( start.xyz[idim] );
	gapends[idim].push_back( end.xyz[idim] );
      }

      float gaplen=0;
      float dir[3];
      for (int i=0; i<3; i++) {
	dir[i] = end.xyz[i]-start.xyz[i];
	gaplen += dir[i]*dir[i];
      }
      gaplen = sqrt(gaplen);
      for (int i=0; i<3; i++) dir[i] /= gaplen;      
      float dtick = end.tick-start.tick;

      int nsteps = gaplen/kMaxStepSize + 1;
      float stepsize = gaplen/float(nsteps);
      float stepdtick = dtick/float(nsteps);

      std::cout << "  Fill gap in dim " << dim << " between: "
		<< " (" << start.xyz[0] << "," << start.xyz[1] << "," << start.xyz[2] << ") and "	
		<< " (" << end.xyz[0] << "," << end.xyz[1] << "," << end.xyz[2] << ") "
		<< " with " << nsteps << " steps of size " << stepsize << " cm"
		<< std::endl;
      
      // generate points
      for (int istep=0; istep<nsteps-1; istep++) {
	QPoint_t pt;
	for (int i=0; i<3; i++)
	  pt.xyz[i] = start.xyz[i] + (float(istep)+0.5)*stepsize*dir[i];
	pt.tick = start.tick + (float(istep)+0.5)*stepdtick;
	pt.pixeladc = stepsize; // we store step size insead of pixel value
	pt.intpc = 1;
	pt.gapfill = 1;
	cluster.emplace_back( std::move(pt) );
      }
    }
  }//end of fillClusterGaps

  void LArFlowFlashMatch::fillClusterGapsUsingCorePCA( QCluster_t& cluster ) {
    std::vector< std::vector<float> > clusterpts( cluster.size() ); // this copy is unfortunate
    for (size_t ihit=0; ihit<cluster.size(); ihit++ ) {
      clusterpts[ihit].resize(3);
      for (int i=0; i<3; i++ ) clusterpts[ihit][i] = cluster[ihit].xyz[i];
    }

    int minneighbors = 3;
    int minclusterpoints = 5;
    float maxdist = 10.0;
    CoreFilter corealgo( clusterpts, minneighbors, maxdist );
    std::vector< std::vector<float> > core = corealgo.getCore( minclusterpoints, clusterpts );

    // get pca of core points
    CilantroPCA pcalgo( core );
    larlite::pcaxis pca = pcalgo.getpcaxis();
    
  }

  std::vector<LArFlowFlashMatch::FlashData_t> LArFlowFlashMatch::collectFlashInfo( const std::vector<larlite::opflash>& beam_flashes,
										   const std::vector<larlite::opflash>& cosmic_flashes ) {

    const larutil::Geometry* geo       = larutil::Geometry::GetME();
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    //const size_t npmts = geo->NOpDets();
    const size_t npmts = 32;
    const float  usec_per_tick = 0.5; // usec per tick
    const float  tpc_trigger_tick = 3200;
    const float  driftv = larp->DriftVelocity();
    
    std::vector< FlashData_t > flashdata( beam_flashes.size()+cosmic_flashes.size() );
    int iflash = 0;
    const std::vector<larlite::opflash>* flashtypes[2] = { &beam_flashes, &cosmic_flashes };
    
    for (int n=0; n<2; n++) {
      for ( auto const& flash : *flashtypes[n] ) {
	flashdata[iflash].resize( npmts, 0.0 );
	flashdata[iflash].tot = 0.;
	float maxpmtpe = 0.;
	int maxpmtch = 0;
	for (size_t ich=0; ich<npmts; ich++) {
	  //float pe = flash.PE( geo->OpDetFromOpChannel( ich ) );
	  float pe = flash.PE( ich );
	  flashdata[iflash][ich] = pe;
	  flashdata[iflash].tot += pe;
	  if ( pe > maxpmtpe ) {
	    maxpmtpe = pe;
	    maxpmtch = ich;
	  }
	}
	flashdata[iflash].tpc_tick  = tpc_trigger_tick + flash.Time()/0.5;
	flashdata[iflash].tpc_trigx = flash.Time()*driftv; // x-assuming x=0 occurs when t=trigger
	flashdata[iflash].maxch     = maxpmtch;
	Double_t pmtpos[3];
	geo->GetOpChannelPosition( maxpmtch, pmtpos );	
	flashdata[iflash].maxchposz   = pmtpos[2];
	
	// normalize
	for (size_t ich=0; ich<npmts; ich++)
	  flashdata[iflash][ich] /= flashdata[iflash].tot;
	iflash++;

	std::cout << "dataflash[" << iflash-1 << "] tick=" << flashdata[iflash-1].tpc_tick << " totpe=" << flashdata[iflash-1].tot << std::endl;
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

  void LArFlowFlashMatch::resetCompatibiltyMatrix() {
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
    
    // each cluster builds a hypothesis for each compatible flash

    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float pixval2photons = (2.2/40)*0.3*40000*0.5*0.01; // [mip mev/cm]/(adc/MeV)*[pixwidth cm]*[phot/MeV]*[pe/phot] this is a WAG!!!
    const float gapfill_len2adc  = (60.0/0.3); // adc value per pixel for mip going 0.3 cm through pixel, factor of 2 for no field        
    const float outoftpc_len2adc = 2.0*gapfill_len2adc; // adc value per pixel for mip going 0.3 cm through pixel, factor of 2 for no field


    m_flash_hypo_map.clear();
    m_flash_hypo_v.clear();
    m_flash_hypo_v.reserve(flashdata_v.size()*qcluster_v.size());
    
    for (int iflash=0; iflash<flashdata_v.size(); iflash++) {

      const FlashData_t& flash = flashdata_v[iflash]; // original flash

      for ( int iq=0; iq<qcluster_v.size(); iq++) {
	int compat = getCompat( iflash, iq );
	if ( compat!=0 && flash.truthmatched_clusteridx!=iq )
	  continue;

	const QCluster_t& qcluster = qcluster_v[iq];
	
	FlashHypo_t hypo;
	hypo.resize(npmts,0.0);
	hypo.clusteridx = iq;     // use original
	hypo.flashidx   = iflash; // use original
	hypo.tot_intpc = 0.;
	hypo.tot_outtpc = 0.;
	float norm = 0.0;
	for ( size_t ihit=0; ihit<qcluster.size(); ihit++ ) {
	  double xyz[3];
	  xyz[1] = qcluster[ihit].xyz[1];
	  xyz[2] = qcluster[ihit].xyz[2];
	  xyz[0] = (qcluster[ihit].tick - flash.tpc_tick)*0.5*driftv;

	  if ( xyz[0]>256.0 )
	    continue; // i dont trust the hypotheses here
	  
	  const std::vector<float>* vis = photonlib.GetAllVisibilities( xyz );
	  int intpc = qcluster[ihit].intpc;
	  int gapfill = qcluster[ihit].gapfill;

	  if ( vis && vis->size()==npmts) {
	    for (int ich=0; ich<npmts; ich++) {
	      float q  = qcluster[ihit].pixeladc;
	      float pe = 0.;
	      if ( intpc==1 && gapfill==0 ) {
		// q is a pixel values
		pe = q*(*vis)[ geo->OpDetFromOpChannel( ich ) ];
		hypo.tot_intpc += pe;
	      }
	      else {
		// outside tpc, what is stored is the track length
		if ( intpc==0 )
		  pe = q*outoftpc_len2adc;
		else if ( gapfill==1 )
		  pe = q*gapfill_len2adc;
		
		pe *= (*vis)[ geo->OpDetFromOpChannel( ich ) ]; // naive: hardcoded factor of two for no-field effect
		hypo.tot_outtpc += pe;
	      }
	      hypo[ich] += pe;
	      norm += pe;
	    }
	  }
	  else if ( vis->size()>0 && vis->size()!=npmts ) {
	    throw std::runtime_error("[LArFlowFlashMatch::buildFlashHypotheses][ERROR] unexpected visibility size");
	  }
	}
	
	// normalize
	hypo.tot = norm;
	if ( norm>0 ) {
	  for (size_t ich=0; ich<hypo.size(); ich++)
	    hypo[ich] /= norm;
	}

	// store
	int idx = m_flash_hypo_v.size();
	m_flash_hypo_map[ flashclusterpair_t((int)iflash,iq) ] = idx;
	m_flash_hypo_v.emplace_back( std::move(hypo) );
      }//end of loop over clusters
    }//end of loop over flashes

  }

  bool LArFlowFlashMatch::hasHypothesis( int flashidx, int clustidx ) {
    flashclusterpair_t fcpair( flashidx, clustidx );
    auto it = m_flash_hypo_map.find( fcpair );
    if ( it==m_flash_hypo_map.end() ) return false;
    return true;
  }
  
  LArFlowFlashMatch::FlashHypo_t& LArFlowFlashMatch::getHypothesisWithOrigIndex( int flashidx, int clustidx ) {
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

  float LArFlowFlashMatch::shapeComparison( const FlashHypo_t& hypo, const FlashData_t& data, float data_norm, float hypo_norm ) {
    // we sum over z-bins and measure the max-distance of the CDF
    const int nbins_cdf = _zbinned_pmtchs.size();
    float hypo_cdf[nbins_cdf] = {0};
    float data_cdf[nbins_cdf] = {0};
    float maxdist = 0;

    float norm_hypo = 0.;
    float norm_data = 0.;
    
    // fill cdf
    for (int ibin=0; ibin<nbins_cdf; ibin++) {
      float binsum_hypo = 0.;
      float binsum_data = 0.;                
      for (auto& ich : _zbinned_pmtchs[ibin] ) {
	binsum_hypo += hypo[ich]*hypo_norm;
	binsum_data += data[ich]*data_norm;
      }
      norm_hypo += binsum_hypo;
      norm_data += binsum_data;
      hypo_cdf[ibin] = norm_hypo;
      data_cdf[ibin] = norm_data;
    }

    // norm cdf and find maxdist
    for (int ibin=0; ibin<nbins_cdf; ibin++) {
      if ( norm_hypo>0 )
	hypo_cdf[ibin] /= norm_hypo;
      if ( norm_data>0 )
	data_cdf[ibin] /= norm_data;
    
      float dist = fabs( hypo_cdf[ibin]-data_cdf[ibin]);
      if ( dist>maxdist )
	maxdist = dist;
    }
    std::cout << "tot_hypo=" << norm_hypo << " tot_data=" << norm_data << " maxdist=" << maxdist << std::endl;
    return maxdist;
  }

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
    // from this function we reduced number of possible flash-cluster matches
    // this is done by
    // (1) comparing shape (CDF maxdist) and chi2
    // (2) for chi2, if adjust_pe_for_cosmic_disc, we try to account for pe lost due to cosmic disc threshold
    
    _flashdata_best_hypo_maxdist_idx.resize(flashdata_v.size(),-1);
    _flashdata_best_hypo_maxdist.resize(flashdata_v.size(),-1);
    _flashdata_best_hypo_chi2_idx.resize(flashdata_v.size(),-1);
    _flashdata_best_hypo_chi2.resize(flashdata_v.size(),-1);
    
    _clustdata_best_hypo_maxdist_idx.resize(qcluster_v.size(),-1);
    _clustdata_best_hypo_chi2_idx.resize(qcluster_v.size(),-1);    

    std::vector<float> hyposcale_v;
    std::vector<float> scaleweight_v;    
    hyposcale_v.reserve( flashdata_v.size()*qcluster_v.size() );
    
    for (int iflash=0; iflash<flashdata_v.size(); iflash++) {
      const FlashData_t& flashdata = flashdata_v[iflash];
      std::vector< int > clustmatches;
      std::vector< float > maxdist;
      float bestdist = 2.0;
      int bestidx = -1;
      int bestchi2_idx = -1;
      float bestchi2 = -1;
      
      for (int iclust=0; iclust<qcluster_v.size(); iclust++) {
	if ( getCompat( iflash, iclust )!=0 )
	  continue;

	// get hypo
	FlashHypo_t& hypo = getHypothesisWithOrigIndex( iflash, iclust );

	// for most-generous comparison, we renorm hypo flash to data total pe
	// but we dont total predictions where flashdata is zero due to cosmic disc window

	// get the total pot, for overlapping pmts
	float hypo_renorm = 0.; // vis-only normalization, only including flashdata channel > 0
	float hypo_pe_belowcosmicdisc_thresh = 0.;
	for (size_t ich=0; ich<flashdata.size(); ich++) {
	  float chpe = hypo[ich]*hypo.tot;
	  if ( flashdata[ich]>0 || flashdata.isbeam ) {
	    hypo_renorm += chpe;
	  }
	  if ( chpe < _fCosmicDiscThreshold ) {
	    hypo_pe_belowcosmicdisc_thresh += chpe;
	  }
	}

	if ( hypo_renorm == 0.0 ) {
	  // no overlap between data and hypo -- good, can reject
	  setCompat(iflash,iclust,3); // no overlap
	}
	else {
	  //FlashHypo_t& copy = hypo;
	  // give ourselves a new working copy
	  FlashHypo_t copy(hypo);
	  
	  //float hypo_scale = flashdata.tot/(hypo_renorm/hypo.tot); // we want
	  float hypo_scale = flashdata.tot;
	  //std::cout << "data.tot=" << flashdata.tot << " hypo_scale=" << hypo_scale << " copy.tot=" << copy.tot << " copy.size=" << copy.size() << std::endl;
	  
	  // we enforce cosmic dic. threshold by scaling hypo to data and zero-ing below threshold
	  copy.tot = 0.0; // copy norm
	  for (size_t ich=0; ich<hypo.size(); ich++) {
	    float copychpred = hypo[ich]*hypo_scale;
	    if ( adjust_pe_for_cosmic_disc && copychpred<_fCosmicDiscThreshold )
	      copy[ich] = 0.;
	    else
	      copy[ich] = copychpred;
	    //std::cout << "copy.chpred=" << copy[ich] << " vs. chpred=" << copychpred << std::endl;	  
	    copy.tot += copy[ich];
	  }
	  //std::cout << "copy.tot=" << copy.tot << std::endl;
	  if ( copy.tot==0 ) {
	    setCompat(iflash,iclust,3);
	    continue;
	  }

	  // normalize
	  for (size_t ich=0; ich<flashdata.size(); ich++)
	    copy[ich] /= copy.tot;
	  
	  float maxdist = shapeComparison( copy, flashdata, flashdata.tot, copy.tot );
	  float chi2    = chi2Comparison( copy, flashdata, flashdata.tot, copy.tot );
	  
	  hyposcale_v.push_back( copy.tot/flashdata.tot  ); // save data/mc ratio
	  scaleweight_v.push_back( exp(-0.5*chi2 ) );

	  //std::cout << "hyposcale=" << hypo_scale << "  chi2=" << chi2 << std::endl;

	  if ( maxdist>_fMaxDistCut ) {
	    setCompat(iflash,iclust,4); // fails shape match
	    continue;
	  }

	  // update bests for flash
	  if ( maxdist < bestdist ) {
	    bestdist = maxdist;
	    bestidx = iclust;
	  }
	  if ( chi2 < bestchi2 || bestchi2<0 ) {
	    bestchi2 = chi2;
	    bestchi2_idx = iclust;
	  }

	  // update bests for clust
	  // if ( maxdist < _clustdata_best_hypo_maxdist[iclust] ) {
	  //   _clustdata_best_hypo_maxdist_idx[iclust] = iflash;
	  //   std::cout << "update best cluster flash: maxdist=" << maxdist << " idx=" << iflash << std::endl;
	  // }
	  // if ( chi2 < _clustdata_best_hypo_chi2[iclust] ) {
	  //   _clustdata_best_hypo_chi2_idx[iclust] = iflash;
	  // }
	  
	}//end of if valid renorm

      }//end of cluster loop

      // store bests
      _flashdata_best_hypo_maxdist_idx[iflash] = bestidx;
      _flashdata_best_hypo_maxdist[iflash]     = bestdist;
      _flashdata_best_hypo_chi2_idx[iflash]    = bestchi2_idx;
      _flashdata_best_hypo_chi2[iflash]        = bestchi2;

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
	if ( _flashdata_best_hypo_maxdist_idx.size()==_nflashes && _flashdata_best_hypo_maxdist_idx[iflash]==idx )
	  bestmaxdist = true;
	if ( _flashdata_best_hypo_chi2_idx.size()==_nflashes && _flashdata_best_hypo_chi2_idx[iflash]==idx )
	  bestchi2 = true;

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
    std::cout << "[Total " << totcompat << "]" << std::endl;
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
    

    for (int iflash=0; iflash<_nflashes; iflash++) {
      c.Clear();
      c.cd();
      
      int ncompat = 0;

      // get data histogram
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
      int bestchi2_idx = -1;
      float bestchi2 = -1;      
      int bestmaxdist_idx = -1;
      float bestmaxdist = -1;
      float peratio_best = 1.0e9;
      int bestfmatch_idx = -1;
      float matchscore_best = 0;
      const FlashHypo_t* bestchi2_hypo = nullptr;
      const FlashHypo_t* bestmaxdist_hypo = nullptr;
      const FlashHypo_t* bestfmatch_hypo = nullptr;

      int truthmatch_idx = -1;
      float truthmatch_chi2 = -1;
      float truthmatch_maxdist = -1;
      const FlashHypo_t* truthmatch_hypo = nullptr;

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

	float fmatchscore = 0.;
	if ( usefmatch && imatch>=0 ) {
	  // we mark the best score
	  fmatchscore = fmatch[imatch];
	  if ( fmatchscore >= matchscore_best || fmatchscore>0.98 ) {
	    // basically the same
	    float peratio = fabs(hypo_norm/norm-1.0);
	    if ( peratio < peratio_best ) {
	      bestfmatch_idx = iclust;
	      matchscore_best = fmatchscore;	      
	      bestfmatch_hypo = &hypo;
	      peratio_best = peratio;	      
	      tophistidx = hclust_v.size();
	    }
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
	  ntop++;
	}
	if ( _flashdata_best_hypo_chi2_idx.size()==_nflashes && _flashdata_best_hypo_chi2_idx[iflash]==iclust ) {
	  hhypo->SetLineWidth(2);
	  hhypo->SetLineColor(kCyan);
	  bestchi2_idx = iclust;
	  bestchi2 = _flashdata_best_hypo_chi2[iflash];
	  bestchi2_hypo = &hypo;
	  ntop++;
	}

	// both best chi2 and maxdist
	if ( ntop==2 )  {
	  hhypo->SetLineColor(kMagenta);
	}

	if ( truthmatched ) {
	  truthmatch_idx = iclust;
	  truthmatch_hypo = &hypo;
	  hhypo->SetLineColor(kGreen+3);
	  hhypo->SetLineWidth(3);
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
	if ( max < hhypo->GetMaximum() )
	  max = hhypo->GetMaximum();
	hclust_v.push_back( hhypo );
      }//end of cluster loop

      std::cout << "flash[" << iflash << "] bestchi2[" << _flashdata_best_hypo_chi2_idx[iflash] << "]  fromloop[" << bestchi2_idx << "]" << std::endl;

      hdata.SetMaximum( max*1.1 );
      
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
      TGraph* truthclust = nullptr;
      TGraph* truthclust_xy = nullptr;      
      if ( qtruth ) {
	std::cout << "qtruth[" << flashdata_v[iflash].truthmatched_clusteridx << "] npoints: " << qtruth->size() << std::endl;
	truthclust = new TGraph(qtruth->size());
	truthclust_xy = new TGraph(qtruth->size());
	float xoffset = (flashdata_v[iflash].tpc_tick-3200)*0.5*driftv;
	for (int ipt=0; ipt<(int)qtruth->size(); ipt++) {
	  truthclust->SetPoint(ipt,qtruth->at(ipt).xyz[2], qtruth->at(ipt).xyz[1] );
	  truthclust_xy->SetPoint(ipt,qtruth->at(ipt).xyz[0]-xoffset, qtruth->at(ipt).xyz[1] );
	  //std::cout << "qtruth[0] = " << qtruth->at(ipt).xyz[0]-xoffset << " (w/ offset=" << xoffset << ")" << std::endl;
	}
	truthclust->SetMarkerSize(0.5);
	truthclust->SetMarkerStyle(20);
	truthclust_xy->SetMarkerSize(0.5);
	truthclust_xy->SetMarkerStyle(20);	
      }

      // mc-track
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

      for (int ich=0; ich<32; ich++) {
	datamarkers_v[ich]->Draw();
	pmtmarkers_v[ich]->Draw();
	chmarkers_v[ich]->Draw();
      }
      
      if ( truthclust )
	truthclust->Draw("P");
      if ( mctrack_data )
	mctrack_data->Draw("L");

      // XY: DATA
      // --------
      dataxy.cd();
      bgxy.Draw();
      boxxy.Draw();
      if ( truthclust_xy )
	truthclust_xy->Draw("P");
      if ( mctrack_data_xy )
	mctrack_data_xy->Draw("L");
      
      // YZ-2D plots: hypothesis
      // -----------------------
      hypoyz.cd();
      bg.Draw();
      boxzy.Draw();

      for (int ich=0; ich<32; ich++) {
	if (hypo) 
	  hypomarkers_v[ich]->Draw();
	pmtmarkers_v[ich]->Draw();
	chmarkers_v[ich]->Draw();
	if ( truthmatch_hypo )
	  truthmarkers_v[ich]->Draw();
      }
      
      TGraph* truthclust2 = nullptr;
      TGraph* truthclust2_xy = nullptr;
	
      if ( bestmatch_iclust>=0 ) {
	
	QCluster_t* qc = &(_qcluster_v[bestmatch_iclust]);
	//std::cout << "qc[" << iclust << "] npoints: " << qc->size() << std::endl;
	truthclust2 = new TGraph(qc->size());
	truthclust2_xy = new TGraph(qc->size());
	float xoffset = (flashdata_v[iflash].tpc_tick-3200)*0.5*driftv;	
	for (int ipt=0; ipt<(int)qc->size(); ipt++) {
	  truthclust2->SetPoint(ipt,qc->at(ipt).xyz[2], qc->at(ipt).xyz[1] );
	  truthclust2_xy->SetPoint(ipt,qc->at(ipt).xyz[0]-xoffset, qc->at(ipt).xyz[1] );	  
	}
	truthclust2->SetMarkerSize(0.5);
	truthclust2->SetMarkerStyle(20);
	truthclust2_xy->SetMarkerSize(0.5);
	truthclust2_xy->SetMarkerStyle(20);
      }
            
      if ( truthclust2 )
	truthclust2->Draw("P");
      if ( mctrack_data )
	mctrack_data->Draw("L");
      
      hypoxy.cd();
      bgxy.Draw();
      boxxy.Draw();
      if ( truthclust2_xy )
	truthclust2_xy->Draw("P");
      if ( mctrack_data_xy )
	mctrack_data_xy->Draw("L");      

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
	sprintf( ztruthscores, "Truth-Chi2=%.2f  Truth-Maxdist=%.2f", truthmatch_chi2, truthmatch_maxdist );

      char zbestchi[100];
      if ( bestchi2_idx>=0 )
	sprintf( zbestchi, "Best Chi2 idx (cyan): %d  Chi2=%.1f", bestchi2_idx, bestchi2 );
      else
	sprintf( zbestchi, "No chi2 match" );

      char zbestmaxdist[100];
      if ( bestmaxdist_idx>=0 )
	sprintf( zbestmaxdist, "Best maxdist idx (blue): %d  maxdist=%.2f", bestmaxdist_idx, bestmaxdist );
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
      if ( truthclust )
	delete truthclust;
      truthclust = nullptr;
      if ( truthclust_xy )
	delete truthclust_xy;
      
      if ( !qtruth || usefmatch ) {
	delete truthclust2;
	delete truthclust2_xy;
      }
      if (mctrack_data)
	delete mctrack_data;
      if ( mctrack_data_xy )
	delete mctrack_data_xy;

      for (int ic=0; ic<(int)hclust_v.size(); ic++) {
	delete hclust_v[ic];
      }
      hclust_v.clear();      

      delete tbestfmatch;
      tbestfmatch = nullptr;
      
    } //end of flash loop
    
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
    for (auto& mct : *_mctrack_v ) {
      imctrack++;
      _mctrackid2index[mct.TrackID()] = imctrack;
      
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
	if ( dtick < 5 ) { // space charge?
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
      }
      else if (id.size()>1 ) {
	// to resolve multiple options.
	// (1) favor id==mid && pdg=|13| (muon) -- from these, pick best time
	int nmatches = 0;
	int idx = -1;
	int pdgx = -1;
	float closestz = 10000;
	for (int i=0; i<(int)id.size(); i++) {
	  std::cout << "multuple-truthmatches[" << i << "] id=" << id[i] << " mid=" << mid[i] << " pdg=" << pdg[i] << " dz=" << dz[i] << std::endl;
	  if ( id[i]==mid[i] && abs(pdg[i])==13 && dz[i]<closestz) {
	    idx = id[i];
	    pdgx = pdg[i];
	    closestz = dz[i];
	    //nmatches++;
	  }
	}
	flashdata_v[iflash].mctrackid = idx;
	flashdata_v[iflash].mctrackpdg = pdgx;	  
      }// if multipl matched ids
      int nmcpts = (*_mctrack_v)[ _mctrackid2index[flashdata_v[iflash].mctrackid] ].size();
      std::cout << "FlashMCtrackMatch[" << iflash << "] "
		<< "trackid=" << flashdata_v[iflash].mctrackid << " "
		<< "pdg=" << flashdata_v[iflash].mctrackpdg << " "
		<< "nmcpts=" << nmcpts << std::endl;
    }
    kFlashMatchedDone = true;
  }

  void LArFlowFlashMatch::doTruthCluster2FlashTruthMatching( std::vector<FlashData_t>& flashdata_v, std::vector<QCluster_t>& qcluster_v ) {
    for (int iflash=0; iflash<(int)flashdata_v.size(); iflash++) {
      FlashData_t& flash = flashdata_v[iflash];

      for (int iclust=0; iclust<(int)qcluster_v.size(); iclust++) {
	QCluster_t& cluster = qcluster_v[iclust];

	if ( flash.mctrackid!=-1 && flash.mctrackid==cluster.mctrackid ) {
	  flash.truthmatched_clusteridx = iclust;
	  cluster.truthmatched_flashidx = iflash;
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
	    qpt.intpc = 0;
	    qpt.gapfill = 0;
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


  
}
