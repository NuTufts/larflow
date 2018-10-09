#include "LArFlowFlashMatch.h"
#include <sstream>

// ROOT
#include "TCanvas.h"
#include "TH1D.h"
#include "TRandom3.h"
#include "TFile.h"
#include "TTree.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"

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
      fpmtweight(nullptr),
      _parsdefined(false)
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
    _fclustsum_weight = 1e4;
    _fflashsum_weight = 0.5e4;
    _fl1norm_weight = 0.1;
    _flightyield_weight = 1.0;

    // random generators
    _rand = new TRandom3( 4357 );
  }
  
  LArFlowFlashMatch::Results_t LArFlowFlashMatch::match( const std::vector<larlite::opflash>& beam_flashes,
							 const std::vector<larlite::opflash>& cosmic_flashes,
							 const std::vector<larlite::larflowcluster>& clusters,
							 const std::vector<larcv::Image2D>& img_v ) {
    
    // first is to build the charge points for each cluster
    std::vector< QCluster_t > qcluster_v( clusters.size() );

    // we build up the charge clusters that are easy to grab
    buildInitialQClusters( clusters, qcluster_v, img_v, 2 );

    // we have to build up charge in dead regions in the Y-plane
    // [TODO]
    
    // also collect charge from pixels within the track
    //  that might have been flowed to the wrong cluster and not apart of the correct cluster.
    //  we do this using a neighbor fill (if # of pixels around pixel belong to given cluster
    //  make a qpoint
    // [TODO]

    // collect the flashes
    std::vector< FlashData_t > flashdata_v = collectFlashInfo( beam_flashes, cosmic_flashes );

    std::cout << "Number of data flashes: " << flashdata_v.size() << std::endl;
    std::cout << "Number of clusters: " << qcluster_v.size() << std::endl;
    
    // build compbatility matrix
    buildFullCompatibilityMatrix( flashdata_v, qcluster_v );

    if ( kDoTruthMatching && _mctrack_v!=nullptr ) {
      std::cout << "[LArFlowFlashMatch::match][INFO] Doing MCTrack truth-reco matching" << std::endl;
    }
    
    // now build hypotheses: we only do so for compatible pairs
    buildFlashHypotheses( flashdata_v, qcluster_v );

    // refined compabtibility: incompatible-z
    reduceMatchesWithShapeAnalysis( flashdata_v, qcluster_v );
    
    // define the fitting data members
    buildFittingData( flashdata_v, qcluster_v );
    
    // define fit parameter variables
    defineFitParameters();

    // find initial best fit to start
    setInitialFitPoint( flashdata_v, qcluster_v );
    
    // fit: gradient descent

    // fit: MCMC
    //runMCMC;

    // build flash-posteriors
    //calcFlashPosteriors;
    *flightyield = 2.0;
    _fweighted_scalefactor_sig  = 0.5;
    _fweighted_scalefactor_mean = 2.0;    

    printCompatInfo();
    dumpMatchImages( flashdata_v, false, false );

    int nsamples = 1000000;
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
      dstate = (int)generateProposal( 1, 0.0, 0.1, proposal_v, proposal_ly );
      
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
      std::cout << "cluster[" << iclust << "] ";
      
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
      std::cout << "flash[" << iflash << "] ";
      
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
    dumpMatchImages( flashdata_v, false, true );    
  }


  void LArFlowFlashMatch::buildInitialQClusters( const std::vector<larlite::larflowcluster>& lfclusters, std::vector<QCluster_t>& qclusters,
						 const std::vector<larcv::Image2D>& img_v, const int src_plane ) {

    if ( qclusters.size()!=lfclusters.size() )
      qclusters.resize( lfclusters.size() );

    const larcv::ImageMeta& src_meta = img_v[src_plane].meta();
    
    for ( size_t icluster=0; icluster<lfclusters.size(); icluster++ ) {

      const larlite::larflowcluster& lfcluster = lfclusters[icluster];
      QCluster_t& qcluster = qclusters[icluster];

      qcluster.resize( lfcluster.size() );
      for ( size_t i=0; i<3; i++) {
	qcluster.min_tyz[i] =  1.0e9;
	qcluster.max_tyz[i] = -1.0e9;
      }
      for ( size_t ihit=0; ihit<lfcluster.size(); ihit++ ) {
	for (size_t i=0; i<3; i++)
	  qcluster[ihit].xyz[i] = lfcluster[ihit][i];
	qcluster[ihit].tick     = lfcluster[ihit].tick; 

	if ( qcluster[ihit].tick<src_meta.min_y() || qcluster[ihit].tick>src_meta.max_y() )
	  continue;

	if ( qcluster[ihit].tick > qcluster.max_tyz[0] )
	  qcluster.max_tyz[0] = qcluster[ihit].tick;
	if ( qcluster[ihit].tick < qcluster.min_tyz[0] )
	  qcluster.min_tyz[0] = qcluster[ihit].tick;
	
	for (size_t i=1; i<3; i++) {
	  if ( qcluster[ihit].xyz[i] > qcluster.max_tyz[i] )
	    qcluster.max_tyz[i] = qcluster[ihit].xyz[i];
	  if ( qcluster[ihit].xyz[i] < qcluster.min_tyz[i] )
	    qcluster.min_tyz[i] = qcluster[ihit].xyz[i];
	}

	int row = img_v[src_plane].meta().row( lfcluster[ihit].tick );
	int col = img_v[src_plane].meta().col( lfcluster[ihit].srcwire );
	qcluster[ihit].pixeladc    = img_v[src_plane].pixel( row, col );
	qcluster[ihit].fromplaneid = src_plane;
      }//end of hit loop

      //std::cout << "[qcluster " << icluster << "] tick range: " << qcluster.min_tyz[0] << "," << qcluster.max_tyz[0] << std::endl;
    }//end of cluster loop
    
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
	for (size_t ich=0; ich<npmts; ich++) {
	  float pe = flash.PE( geo->OpDetFromOpChannel( ich ) );
	  flashdata[iflash][ich] = pe;
	  flashdata[iflash].tot += pe;
	}
	flashdata[iflash].tpc_tick  = tpc_trigger_tick + flash.Time()/0.5;
	flashdata[iflash].tpc_trigx = flash.Time()*driftv; // x-assuming x=0 occurs when t=trigger
	
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

    m_flash_hypo_map.clear();
    m_flash_hypo_v.clear();
    m_flash_hypo_v.reserve(flashdata_v.size()*qcluster_v.size());
    
    for (int iflash=0; iflash<flashdata_v.size(); iflash++) {

      const FlashData_t& flash = flashdata_v[iflash]; // original flash

      for ( int iq=0; iq<qcluster_v.size(); iq++) {
	int compat = getCompat( iflash, iq );
	if ( compat!=0 )
	  continue;

	const QCluster_t& qcluster = qcluster_v[iq];
	
	FlashHypo_t hypo;
	hypo.resize(npmts,0.0);
	hypo.clusteridx = iq;     // use original
	hypo.flashidx   = iflash; // use original
	float norm = 0.0;
	for ( size_t ihit=0; ihit<qcluster.size(); ihit++ ) {
	  double xyz[3];
	  xyz[1] = qcluster[ihit].xyz[1];
	  xyz[2] = qcluster[ihit].xyz[2];
	  xyz[0] = (qcluster[ihit].tick - flash.tpc_tick)*0.5*driftv;
	  
	  const std::vector<float>* vis = photonlib.GetAllVisibilities( xyz );
	  if ( vis && vis->size()==npmts) {
	    for (int ich=0; ich<npmts; ich++) {
	      float pe = qcluster[ihit].pixeladc*(*vis)[ geo->OpDetFromOpChannel( ich ) ];
	      hypo[ich] += pe;
	      norm += pe;
	    }
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

  void LArFlowFlashMatch::defineFitParameters() {
    if ( !_reindexed )
      throw std::runtime_error("[LArFlowFlashMatch::defineFitParametere][ERROR] must call buildFittingData first");
    clearFitParameters();
    flightyield = new float;
    fmatch = new float[_nmatches];
    fpmtweight = new float[_nflashes_red*32];
    
    memset( fpmtweight, 0, sizeof(_nflashes_red )*32 );
    memset( fmatch, 0, sizeof(float)*_nmatches );
    _parsdefined = true;
  }

  void LArFlowFlashMatch::clearFitParameters() {
    delete flightyield;
    delete [] fmatch;
    delete [] fpmtweight;
    flightyield = nullptr;
    fmatch = nullptr;
    fpmtweight = nullptr;
    _parsdefined = false;
  }
  
  void LArFlowFlashMatch::setInitialFitPoint(const std::vector<FlashData_t>& flashdata_v, const std::vector<QCluster_t>&  qcluster_v ) {

    if ( !_parsdefined )
      throw std::runtime_error("[LArFlowFlashMatch::setInitialFitPoint][ERROR] must call setInitialFitPoint first");
    
    (*flightyield) = _fweighted_scalefactor_mean;
    // for ( int iflash=0; iflash<_nflashes; iflash++) {

    //   auto it = _flash_reindex.find( iflash );
    //   if ( it==_flash_reindex.end() )
    // 	continue; // not paired

    //   int bestclustidx = _flashdata_best_hypo_maxdist_idx[iflash]; // get cluster (original index) that best matches
      
    //   int reflashidx   = (*it).second;
    //   int reclustidx   = _clust_reindex[bestclustidx];
    //   int imatch = getMatchIndex( reflashidx, reclustidx );
    //   *(fmatch+imatch) = 1.0;
    // }
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

  void LArFlowFlashMatch::reduceMatchesWithShapeAnalysis( const std::vector<FlashData_t>& flashdata_v,
							  const std::vector<QCluster_t>&  qcluster_v ) {
    // from this function we get
    // (1) reduced number of possible flash-cluster matches
    // (2) an initial estimate of the ligh yield parameter and sigma bounds for each prior
    // (3) best fits using shape and chi2
    // (from truth-clusters, get about 180 matches left over. that's 2^180 vector we are trying to solve ... )
    
    _flashdata_best_hypo_maxdist_idx.resize(flashdata_v.size(),-1);
    _flashdata_best_hypo_chi2_idx.resize(flashdata_v.size(),-1);    
    _clustdata_best_hypo_maxdist_idx.resize(qcluster_v.size(),-1);
    _clustdata_best_hypo_chi2_idx.resize(qcluster_v.size(),-1);    

    std::vector<float> clustdata_best_hypo_maxdist(qcluster_v.size(),2.0);
    std::vector<float> clustdata_best_hypo_chi2(qcluster_v.size(),1e9);    

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
      float bestchi2 = 1e6;
      
      for (int iclust=0; iclust<qcluster_v.size(); iclust++) {
	if ( getCompat( iflash, iclust )!=0 )
	  continue;

	// get hypo
	FlashHypo_t& hypo = getHypothesisWithOrigIndex( iflash, iclust );

	// for most-generous comparison, we renorm hypo flash to data total pe
	// but we dont total predictions where flashdata is zeor due to cosmic disc window

	float hypo_renorm = 0.; // vis-only normalization, only including flashdata channel > 0
	for (size_t ich=0; ich<flashdata.size(); ich++) {
	  if ( flashdata[ich]>0 || flashdata.isbeam ) {
	    hypo_renorm += hypo[ich]*hypo.tot;
	  }
	}

	//std::cout << "hypo_renorm=" << hypo_renorm << " hypo.tot=" << hypo.tot << " ratio=" << hypo_renorm/hypo.tot << std::endl;

	if ( hypo_renorm == 0.0 ) {
	  // no overlap between data and hypo -- good, can reject
	  setCompat(iflash,iclust,3); // no overlap
	}
	else {
	  //FlashHypo_t& copy = hypo;
	  // give ourselves a new working copy
	  FlashHypo_t copy(hypo);
	  
	  float hypo_scale = flashdata.tot/(hypo_renorm/hypo.tot); // we want 

	  //std::cout << "data.tot=" << flashdata.tot << " hypo_scale=" << hypo_scale << " copy.tot=" << copy.tot << " copy.size=" << copy.size() << std::endl;
	  
	  // we enforce cosmic dic. threshold by scaling hypo to data and zero-ing below threshold
	  copy.tot = 0.0; // copy norm
	  for (size_t ich=0; ich<flashdata.size(); ich++) {
	    float copychpred = hypo[ich]*hypo_scale;
	    //if ( copychpred<_fCosmicDiscThreshold )
	    //copy[ich] = 0.;
	    //else
	    copy[ich] = copychpred;
	    //std::cout << "copy.chpred=" << copy[ich] << " vs. chpred=" << copychpred << std::endl;	  
	    copy.tot += copy[ich];
	  }
	  //std::cout << "copy.tot=" << copy.tot << std::endl;
	  if ( copy.tot==0 ) {
	    setCompat(iflash,iclust,3);
	    continue;
	  }
	  
	  for (size_t ich=0; ich<flashdata.size(); ich++)
	    copy[ich] /= copy.tot;
	  
	  float maxdist = shapeComparison( copy, flashdata, flashdata.tot, copy.tot );

	  // chi2
	  float chi2 = 0;
	  for (size_t ich=0; ich<flashdata.size(); ich++) {
	    float pred = copy[ich]*hypo_scale;
	    float obs  = flashdata[ich];
	    float err = sqrt( pred + obs );
	    if (pred+obs<=0) {
	      err = 1.0e-3;
	    }
	    chi2 += (pred-obs)*(pred-obs)/(err*err)/((float)flashdata.size());
	  }
	  hyposcale_v.push_back(hypo_scale); // save hyposcale
	  scaleweight_v.push_back( exp(-0.5*chi2 ) );

	  //std::cout << "hyposcale=" << hypo_scale << "  chi2=" << chi2 << std::endl;

	  if ( maxdist>_fMaxDistCut ) {
	    setCompat(iflash,iclust,4); // fails shape match
	  }

	  // update bests for flash
	  if ( maxdist < bestdist ) {
	    bestdist = maxdist;
	    bestidx = iclust;
	  }
	  if ( chi2 < bestchi2 ) {
	    bestchi2 = chi2;
	    bestchi2_idx = iclust;
	  }

	  // update bests for clust
	  if ( maxdist < clustdata_best_hypo_maxdist[iclust] ) {
	    clustdata_best_hypo_maxdist[iclust] = maxdist;
	    _clustdata_best_hypo_maxdist_idx[iclust] = iflash;
	    std::cout << "update best cluster flash: maxdist=" << maxdist << " idx=" << iflash << std::endl;
	  }
	  if ( chi2 < clustdata_best_hypo_chi2[iclust] ) {
	    clustdata_best_hypo_chi2[iclust] = chi2;
	    _clustdata_best_hypo_chi2_idx[iclust] = iflash;
	  }
	  
	}//end of if valid renorm

	// update cluser bests
      }//end of cluster loop

      _flashdata_best_hypo_maxdist_idx[iflash] = bestidx;
      _flashdata_best_hypo_chi2_idx[iflash]    = bestchi2_idx;      

    }

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
    float x_zero = _fweighted_scalefactor_mean/_fweighted_scalefactor_sig;
    _ly_neg_prob = 0.5*TMath::Erf( x_zero );

    std::cout << "total weight: " << totweight << std::endl;
    std::cout << "total number of weights: " << int(nweights) << std::endl;
    std::cout << "Weighted scale-factor mean: "   << _fweighted_scalefactor_mean  << std::endl;
    std::cout << "Weighted scale-factor variance (stdev): " << _fweighted_scalefactor_var << " ("  << _fweighted_scalefactor_sig << ")" << std::endl;
  }

  void LArFlowFlashMatch::printCompatInfo() {
    std::cout << "----------------------" << std::endl;
    std::cout << "COMPAT MATRIX" << std::endl;
    std::cout << "----------------------" << std::endl;
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
    TCanvas c("c","",500,400);

    for (int iflash=0; iflash<_nflashes; iflash++) {
      int ncompat = 0;
      TH1D hdata("hdata","",32,0,32);
      float norm = 1.;
      if ( !shapeonly ) {
	norm = flashdata_v[iflash].tot;
      }
      for (int ipmt=0; ipmt<32; ipmt++) {
	hdata.SetBinContent( ipmt+1, flashdata_v[iflash][ipmt]*norm );
      }
      //hdata.SetMaximum(1.);
      //hdata.SetMinimum(0.);
      hdata.SetLineWidth(4);
      hdata.SetLineColor(kBlack);
      hdata.Draw("hist");
      std::vector< TH1D* > hclust_v;

      auto it_flash = _flash_reindex.find( iflash );

      for (int iclust=0; iclust<_nqclusters; iclust++) {

	if ( getCompat(iflash,iclust)!=0 )
	  continue;
	
	if (usefmatch) {
	  auto it_clust = _clust_reindex.find( iclust );
	  if ( it_flash==_flash_reindex.end() || it_clust==_clust_reindex.end() )
	    continue;
	  int imatch = getMatchIndex( it_flash->second, it_clust->second );
	  if ( imatch<0 || fmatch[ imatch ] < 0.001 )
	    continue;
	}
	
	char hname[20];
	sprintf( hname, "hhypo_%d", iclust);
	TH1D* hhypo = new TH1D(hname, "", 32, 0, 32 );
	hhypo->SetLineWidth(1.0);
	hhypo->SetLineColor(kRed);
	bool both = true;
	if ( _flashdata_best_hypo_maxdist_idx.size()==_nflashes && _flashdata_best_hypo_maxdist_idx[iflash]==iclust ) {
	  hhypo->SetLineWidth(2);
	  hhypo->SetLineColor(kBlue);
	}
	else
	  both = false;
	if ( _flashdata_best_hypo_chi2_idx.size()==_nflashes && _flashdata_best_hypo_chi2_idx[iflash]==iclust ) {
	  hhypo->SetLineWidth(2);
	  hhypo->SetLineColor(kCyan);
	}
	else
	  both = false;
	
	if ( both )
	  hhypo->SetLineColor(kMagenta);
	   
	const FlashHypo_t& hypo = getHypothesisWithOrigIndex( iflash, iclust );
	float hypo_norm = 1.;
	if ( !shapeonly )
	  hypo_norm = (*flightyield)*hypo.tot;
	
	for (int ipmt=0;ipmt<32;ipmt++) {
	  hhypo->SetBinContent(ipmt+1,hypo[ipmt]*hypo_norm);
	}
	hhypo->Draw("hist same");
	hclust_v.push_back( hhypo );
      }

      c.Update();
      c.Draw();

      char cname[100];
      sprintf( cname, "hflashdata_compat_pmtch%02d.png",iflash);
      std::cout << "saving " << cname << std::endl;
      c.SaveAs(cname);

      for (int ic=0; ic<(int)hclust_v.size(); ic++) {
	delete hclust_v[ic];
      }
      hclust_v.clear();
    }//end of flash loop
  }

  float LArFlowFlashMatch::calcNLL( bool print ) {


    // calculate agreement
    float nll_data = 0.;

    // fit to data
    for (int imatch=0; imatch<_nmatches; imatch++) {
      float hyponorm = *(m_flashhypo_norm + imatch)*(*flightyield);
      float datanorm = *(m_flashdata_norm + imatch);

      float nll_match = 0.;
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
      }
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

  void LArFlowFlashMatch::loadMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v, bool do_truth_matching ) {
    _mctrack_v = &mctrack_v;
    kDoTruthMatching = do_truth_matching;
    std::cout << "[LArFlowFlashMatch::loadMCTrackInfo][INFO] Loaded MC tracks." << std::endl;
  }
}
