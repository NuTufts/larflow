#include "LArFlowFlashMatch.h"

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
      _pair2index(nullptr)
  {
    
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

    // build compbatility matrix
    buildFullCompatibilityMatrix( flashdata_v, qcluster_v );

    // define the fitting data members
    buildFittingData( flashdata_v, qcluster_v );

    // now build hypotheses: we only do so for compatible pairs
    buildFlashHypotheses( flashdata_v, qcluster_v );

    // refined compabtibility: total PE

    // refined compabtibility: incompatible-z

    // find initial best fit to start
    calcInitialFitPoint();
    
    // fit: gradient descent

    // fit: MCMC
    //runMCMC;

    // build flash-posteriors
    //calcFlashPosteriors;
    
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
	qcluster[ihit].tick     = lfcluster[ihit].tick; // convert this later ...

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

      std::cout << "qcluster. tick range: " << qcluster.min_tyz[0] << "," << qcluster.max_tyz[1] << std::endl;
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

	std::cout << "flash time: " << flashdata[iflash].tpc_tick << std::endl;
	
	// normalize
	for (size_t ich=0; ich<npmts; ich++)
	  flashdata[iflash][ich] /= flashdata[iflash].tot;
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
	float dtick_max = qcluster.min_tyz[1] - flash.tpc_tick;

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
	}
      }
    }
    _compatibility_defined = true;
    std::cout << "number of compat flash-cluster matches: " << ncompatmatches << std::endl;
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

  std::vector< std::vector<LArFlowFlashMatch::FlashHypo_t> > LArFlowFlashMatch::buildFlashHypotheses( const std::vector<FlashData_t>& flashdata_v,
												      const std::vector<QCluster_t>&  qcluster_v ) {
    
    // each cluster builds a hypothesis for each compatible flash
    // store the hypotheses into the fitting variables
    if ( !_reindexed )
      throw std::runtime_error("[LArFlowFlashMatch::buildFlashHypotheses][ERROR] Must reindex first");

    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float pixval2photons = (2.2/40)*0.3*40000*0.5*0.01; // [mip mev/cm]/(adc/MeV)*[pixwidth cm]*[phot/MeV]*[pe/phot] this is a WAG!!!

    std::vector< std::vector<FlashHypo_t> > hypo_vv(_flash_reindex.size()); // inner vector is for hypos from each compat cluster
    
    for (auto& it : _flash_reindex ) {
      size_t iflash  = it.first;  // flash index in original input data array
      size_t reflash = it.second; // reindexed flash
      const FlashData_t& flash = flashdata_v[iflash]; // original flash

      std::vector< FlashHypo_t >& hypo_v = hypo_vv.at(reflash); // reindexed flash
      for ( int iq=0; iq<qcluster_v.size(); iq++) {
	int compat = getCompat( iflash, iq );
	if ( compat!=0 )
	  continue;

	const QCluster_t& qcluster = qcluster_v[iq];
	size_t reclusteridx = _clust_reindex[iq];
	
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
	  for (int ich=0; ich<npmts; ich++) {
	    float pe = qcluster[ihit].pixeladc*(*vis)[ geo->OpDetFromOpChannel( ich ) ];
	    hypo[ich] += pe;
	    norm += pe;
	  }
	}

	// normalize
	hypo.tot = norm;
	for (size_t ich=0; ich<hypo.size(); ich++)
	  hypo[ich] /= norm;

	// copy hypothesis to fitting data member
	int imatch = *(_pair2index + _nclusters_red*iflash + reclusteridx); // get the match index
	memcpy( m_flash_hypo + imatch*npmts, hypo.data(), sizeof(float)*npmts );
	*(m_flashhypo_norm+imatch) = hypo.tot;
	
	hypo_v.emplace_back( std::move(hypo) );
      }//end of loop over clusters
    }//end of loop over flashes
    
    return hypo_vv;
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
	if ( compat==1 ) {
	  _nmatches++;
	  clust_idx.push_back( iq );
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
	  _match_flashidx_orig.push_back( iflash ); // store reindex	  
	  _match_clustidx.push_back( _clust_reindex[qidx] );   // store reindex
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

    // create a map from (reflash,recluster) pair to match index
    _pair2index = new int[ _nflashes_red*_nclusters_red ];
    for (int imatch=0; imatch<_nmatches; imatch++){
      int reflash = _flash_reindex[imatch];
      int reclust = _clust_reindex[imatch];
      *(_pair2index + _nclusters_red*reflash + reclust ) = imatch;
    }

    // copy data flash to m_flash_data
    for (size_t imatch=0; imatch<_nmatches; imatch++) {
      size_t reflash = _match_flashidx[imatch];
      size_t origidx = _match_flashidx_orig[imatch];
      memcpy( m_flash_data+imatch*npmts, flashdata_v[origidx].data(), sizeof(float)*npmts );
      *( m_flashdata_norm+imatch ) = flashdata_v[origidx].tot;
    }
    
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

  void LArFlowFlashMatch::calcInitialFitPoint() {
    // before we fit, we try to find a solution to the many-many match
    // and use it as a best fit
    //
    // we do a naive, iterative method
    //  0) if we have uniquely matched flash-cluster, we use it to set the light yield
    //  1) we loop through each cluster and set the best-flash using a shape comparison (CDF-maxdist for z-projection)
    //  2) adjust the best-fit lightyield
    //  3) repeat (1)+(2) until solved (or no flashes left to match)

    // need the z-order
    const larutil::Geometry* geo = larutil::Geometry::GetME();
    std::vector< std::vector<int> > zbinned_pmtchs(10);
    for (int ich=0; ich<32; ich++) {
      int opdet = geo->OpDetFromOpChannel(ich);
      double xyz[3];
      geo->GetOpChannelPosition( ich, xyz );
      int bin = xyz[2]/100.0;
      zbinned_pmtchs[bin].push_back( ich );
    }
    
  }
}
