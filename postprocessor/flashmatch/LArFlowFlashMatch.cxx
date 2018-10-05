#include "LArFlowFlashMatch.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"

namespace larflow {

  LArFlowFlashMatch::LArFlowFlashMatch()
    : m_compatibility(nullptr)
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
    
    // now build hypotheses: we only do so for compatible pairs
    buildFlashHypotheses( flashdata_v, qcluster_v );

    // refined compabtibility: total PE

    // refined compabtibility: incompatible-z

    // form reduced cluster list, reduced matching matrix
    //buildFitMatrices;
    
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

    std::cout << "max drifttime in ticks: " << max_drifttime_ticks << std::endl;
    
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
    std::cout << "number of compat flash-cluster matches: " << ncompatmatches << std::endl;
  }

  std::vector< std::vector<LArFlowFlashMatch::FlashHypo_t> >  LArFlowFlashMatch::buildFlashHypotheses( const std::vector<FlashData_t>& flashdata_v,
												       const std::vector<QCluster_t>&  qcluster_v ) {
    // each cluster builds a hypothesis for each compatible flash
    std::vector< std::vector<FlashHypo_t> > hypo_vv;

    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float pixval2photons = (2.2/40)*0.3*40000*0.5*0.01; // [mip mev/cm]/(adc/MeV)*[pixwidth cm]*[phot/MeV]*[pe/phot] this is a WAG!!!
    
    for (size_t iflash=0; iflash<flashdata_v.size(); iflash++) {
      const FlashData_t& flash = flashdata_v[iflash];

      int ncompat = 0;
      std::vector< FlashHypo_t > hypo_v;
      for ( int iq=0; iq<qcluster_v.size(); iq++) {
	const QCluster_t& qcluster = qcluster_v[iq];
	int compat = getCompat( iflash, iq );
	if ( compat!=0 )
	  continue;
	ncompat++;

	FlashHypo_t hypo;
	hypo.resize(npmts,0.0);
	hypo.clusteridx = iq;
	hypo.flashidx   = iflash;
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
	for (size_t ich=0; ich<hypo.size(); ich++)
	  hypo[ich] /= norm;

	hypo_v.emplace_back( std::move(hypo) );
      }//end of loop over clusters
      hypo_vv.emplace_back( std::move(hypo_v) );
    }//end of loop over flashes
    
    return hypo_vv;
  }

  
}
