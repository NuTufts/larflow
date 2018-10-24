#include "FlashMatchCandidate.h"

// ROOT
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraph.h"
#include "TPad.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TEllipse.h"
#include "TBox.h"
#include "TText.h"

// larcv
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/ChStatus.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"


namespace larflow {



  FlashMatchCandidate::FlashMatchCandidate( const FlashData_t& fdata, const QClusterCore& qcoredata ) :
    _flashdata(&fdata),
    _cluster(qcoredata._cluster),
    _core(&qcoredata),
    _evstatus(nullptr),
    _hasevstatus(false),
    _topend(0,0,0),
    _botend(0,0,0),
    _flash_mctrackid(-1),
    _flash_mctrackidx(-1),
    _cluster_mctrackid(-1),
    _cluster_mctrackidx(-1),
    _flash_mctrack(nullptr),
    _cluster_mctrack(nullptr)
  {
    // when dealing with information from the core, we will need this offset
    _xoffset = (_flashdata->tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
    // we create a qcluster where we subtract off the xoffset
    _offset_qcluster.clear();
    _offset_qcluster.reserve( _core->_core.size() );
    for (auto const& corehit : _core->_core ) {
      QPoint_t offsethit = corehit;
      offsethit.xyz[0] -= _xoffset;
      offsethit.type = kCore;
      _offset_qcluster.emplace_back( std::move(offsethit) );
    }
    _offset_qgap.clear();
    _offset_qgap.reserve( _core->_gapfill_qcluster.size() );
    for (auto const& gaphit : _core->_gapfill_qcluster ) {
      QPoint_t offsethit = gaphit;
      offsethit.xyz[0] -= _xoffset;
      offsethit.type = kGapFill;
      _offset_qgap.emplace_back( std::move(offsethit) );
    }
    
    _noncorehypo.resize(32,0.0);
    _entering_hypo.resize(32,0.0);
    _exiting_hypo.resize(32,0.0);
    
    // we build the core and gap hypos
    _corehypo = buildFlashHypothesis( *_flashdata, _offset_qcluster, 0.0  );
    _gapfill_hypo = buildFlashHypothesis( *_flashdata, _offset_qgap, 0.0 );

    // we identify the max pmt and its value -- we will use this to tune extensions
    _maxch = 0;
    _maxch_pe = 0;
    for (int ich=0; ich<(int)_flashdata->size(); ich++) {
      float datape = _flashdata->at(ich)*_flashdata->tot;
      if ( datape>_maxch_pe ) {
	_maxch    = ich;
	_maxch_pe = datape;
      }
    }

    ExtendEnteringEnd();
    ExtendExitingEnd();    

  }


  FlashHypo_t FlashMatchCandidate::buildFlashHypothesis( const FlashData_t& flashdata, const QCluster_t&  qcluster, float xoffset ) {
    
    // each cluster builds a hypothesis for each compatible flash
    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float pixval2photons = (2.2/40)*0.3*40000*0.5*0.01; // [mip mev/cm]/(adc/MeV)*[pixwidth cm]*[phot/MeV]*[pe/phot] this is a WAG!!!
    const float gapfill_len2adc  = (100.0/0.3); // adc value per pixel for mip going 0.3 cm through pixel
    const float outoftpc_len2adc = 2.0*gapfill_len2adc; // adc value per pixel for mip going 0.3 cm through pixel, factor of 2 for no field


    FlashHypo_t hypo;
    hypo.resize(npmts,0.0);
    hypo.clusteridx = qcluster.idx; 
    hypo.flashidx   = flashdata.idx;
    hypo.tot_intpc = 0.;
    hypo.tot_outtpc = 0.;
    float norm = 0.0;
    for ( size_t ihit=0; ihit<qcluster.size(); ihit++ ) {
      const QPoint_t& qhit = qcluster[ihit];
      double xyz[3];
      xyz[0] = qhit.xyz[0] - xoffset;
      xyz[1] = qhit.xyz[1];
      xyz[2] = qhit.xyz[2];
      //xyz[0] = (qhit.tick - flashdata.tpc_tick)*0.5*driftv; // careful I don't redo this every time
      
      if ( xyz[0]>250.0 )
	continue; // i dont trust the hypotheses here
	  
      const std::vector<float>* vis = photonlib.GetAllVisibilities( xyz );
      
      if ( vis && vis->size()==npmts) {
	for (int ich=0; ich<npmts; ich++) {
	  float q  = qhit.pixeladc;
	  float pe = 0.;
	  if ( qhit.type==kCore || qhit.type==kNonCore) {
	    // q is a pixel values
	    pe = q*(*vis)[ geo->OpDetFromOpChannel( ich ) ];
	    hypo.tot_intpc += pe;
	  }
	  else if ( qhit.type==kExt ) {
	    // outside tpc, what is stored is the track length
	    pe = q*outoftpc_len2adc;
	    pe *= (*vis)[ geo->OpDetFromOpChannel( ich ) ];
	    hypo.tot_outtpc += pe;
	  }
	  else if ( qhit.type==kGapFill ) {
	    pe = q*gapfill_len2adc;
	    pe *= (*vis)[ geo->OpDetFromOpChannel( ich ) ];
	    hypo.tot_intpc += pe;
	  }
	  else {
	    throw std::runtime_error("[LArFlowFlashMatch::buildFlashHypotheses][ERROR] unrecognized qpoint type");
	  }
	  
	  hypo[ich] += pe;
	  norm += pe;
	}
      }
      else if ( vis->size()>0 && vis->size()!=npmts ) {
	throw std::runtime_error("[LArFlowFlashMatch::buildFlashHypotheses][ERROR] unexpected visibility size");
      }
    }//end of hit loop
	
    // normalize
    hypo.tot = norm;
    if ( norm>0 ) {
      for (size_t ich=0; ich<hypo.size(); ich++)
	hypo[ich] /= norm;
    }

    return hypo;
  }

  float FlashMatchCandidate::getMaxDist( const FlashData_t& flashdata, const FlashHypo_t& flashhypo ) {

    float hypo_cdf = 0.;
    float data_cdf = 0.;
    float maxdist = 0.;
    std::vector<float> hypo(32,0.0);
    std::vector<float> data(32,0.0);
    for (int ich=0; ich<32; ich++) {
      float hypo_pe = flashhypo[ich]*flashhypo.tot;
      float data_pe = flashdata[ich]*flashdata.tot;
      hypo_cdf += hypo_pe;
      data_cdf += data_pe;
      hypo[ich] = hypo_cdf;
      data[ich] = data_cdf;
    }

    for (int ich=0; ich<32; ich++) {
      data[ich] /= data_cdf;
      hypo[ich] /= hypo_cdf;
      float dist = fabs(data[ich]-hypo[ich]);
      if ( maxdist<dist )
	maxdist = dist;
    }
    
    return maxdist;
  }

  float FlashMatchCandidate::getPERatio( const FlashData_t& flashdata, const FlashHypo_t& flashhypo ) {
    float peratio = fabs(flashdata.tot-flashhypo.tot)/flashdata.tot;
    return peratio;
  }
  
  void FlashMatchCandidate::ExtendEnteringEnd() {
    // we have to find which end we think is entering (cosmic assumptoin)
    // we pick the top-most end
    _entering_qcluster.clear();
    _entering_hypo.resize(32,0.0);
    float steplen = 1.0;    
	
    // we've sorted along pca lines, so grab the ends, average over n points
    const QCluster_t& core = _core->_core;
    size_t nptscore = core.size();
    int npoint_ave = 10;
    if ( nptscore<10 )
      npoint_ave = 1;
    else if ( nptscore<20 )
      npoint_ave = 3;
    else if ( nptscore<50 )
      npoint_ave = 5;


    float avepos[2][4] = {0};
    for (int ipt=0; ipt<npoint_ave; ipt++) {
      for (int i=0; i<3; i++) 
	avepos[0][i] += core[ipt].xyz[i]/float(npoint_ave);
      for (int i=0; i<3; i++) 
	avepos[1][i] += core[(int)nptscore-ipt-1].xyz[i]/float(npoint_ave);
      avepos[0][3] += core[ipt].tick/float(npoint_ave);
      avepos[1][3] += core[(int)nptscore-ipt-1].tick/float(npoint_ave);			   
    }
    avepos[0][0] -= _xoffset;
    avepos[1][0] -= _xoffset;    
    
    // choose the top-most end
    _usefront = 0;
    _topend = Eigen::Vector3f(0,0,0);
    _botend = Eigen::Vector3f(0,0,0);    
    _toptick = 0;
    _bottick  = 0;
    if ( avepos[0][1]>avepos[1][1] ) {
      for (int i=0; i<3; i++) _topend(i) = avepos[0][i];
      for (int i=0; i<3; i++) _botend(i) = avepos[1][i];      
      _usefront = 1;
      _toptick = avepos[0][3];
      _bottick = avepos[1][3];
    }
    else {
      for (int i=0; i<3; i++) _topend(i) = avepos[1][i];
      for (int i=0; i<3; i++) _botend(i) = avepos[0][i];      
      _usefront = 0;
      _toptick = avepos[1][3];
      _bottick = avepos[0][3];      
    }

    // extend the points, to the edge of the detector
    // we need a direction: pca. for now just use the current pca
    Eigen::Vector3f centerpos( _core->_pca_core.getAvePosition()[0]-_xoffset,
			       _core->_pca_core.getAvePosition()[1],
			       _core->_pca_core.getAvePosition()[2] );
    Eigen::Vector3f pcavec( _core->_pca_core.getEigenVectors()[0][0],
			    _core->_pca_core.getEigenVectors()[1][0],
			    _core->_pca_core.getEigenVectors()[2][0] );

    if ( pcavec.norm()<0.01 ) {
      return;
    }
    
    // determine sign of pca-axis
    float extsign = ( (_topend-centerpos).dot(pcavec) > 0 ) ? 1.0 : -1.0;

    bool indet = true;
    bool intpc = true;
    bool isanode = false;
    int istep = 0;
    while (indet) {
      Eigen::Vector3f currentpos = _topend + (extsign*steplen*float(istep))*pcavec;
      // check the position
      if ( currentpos(0)<0 ) {
	// went through the anode!
	isanode = true;
	intpc   = false;
      }
      else if ( currentpos(0)>256 || currentpos(1)>117.0 || currentpos(1)<-117 || currentpos(2)<0 || currentpos(2)>1036.0 ) {
	intpc = false;
	isanode = false;
      }

      if ( currentpos(0)<0 || currentpos(0)>256 || currentpos(1)<-137 || currentpos(1)>137 || currentpos(2)<-20 || currentpos(2)>1056.0 ) {
	indet = false;
	break;
      }
      
      if ( indet ) {
	// create the hit
	//std::cout << "create extended hit@ " << currentpos.transpose() << std::endl;
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = currentpos(i);
	qpt.tick   = (currentpos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5) + 3200.0;
	qpt.pixeladc = steplen; 
	qpt.fromplaneid = -1;
	if ( intpc )
	  qpt.type = kGapFill; // we need it to set the light yield according to in-tpc gap fill levels
	else
	  qpt.type = kExt;
	_entering_qcluster.emplace_back( std::move(qpt) );
      }
      istep++;
    }

    if ( isanode ) {
      // we pierced the anode.
      // we keep stepping until the maxpe is matched (stop if we alreay are over)
      // to do this, we need hypothesis to have been formed
      _entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster, 0.0 );
      float _maxdiff = _entering_hypo[_maxch] - _maxch_pe;
      //std::cout << "Is ANODE: extend to match maxpmt" << std::endl;
      istep = 0;
      while ( _maxdiff<0 ) {
	// we extend the entering cluster
	Eigen::Map< Eigen::Vector3f > endpos( _entering_qcluster.back().xyz.data() );
	Eigen::Vector3f extpos = endpos + (extsign*steplen*float(istep))*pcavec;
	// make hit
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = extpos(i);
	qpt.tick = extpos(0)/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200.0;
	qpt.pixeladc = steplen;
	qpt.fromplaneid = -1;
	qpt.type = kExt;
	_entering_qcluster.emplace_back( std::move(qpt) );
	
	if ( extpos(0) < -50 || extpos(1)>117+20 || extpos(1)<-117-20 || extpos(2)<-20 || extpos(2)>1056 ) {
	  // limit this adjustment
	  break;
	}

	// update the hypothesis
	_entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster, 0.0 );
	_maxdiff = _entering_hypo[_maxch] - _maxch_pe;
	// std::cout << "extpos[0]=" << extpos[0] << " maxdiff=" << _maxdiff
	// 	  << "  _hypomax=" << _entering_hypo[_maxch] << " " << " _datamax=" << _maxch_pe << std::endl;
	istep++;
      }
    }
    //entering extension finished
    
    // set final hypo
    _entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster, 0.0 );
    
  }

  void FlashMatchCandidate::ExtendExitingEnd() {
    // we have to find which end we think is entering (cosmic assumptoin)
    // we pick the top-most end
    _exiting_qcluster.clear();
    _exiting_hypo.resize(32,0.0);
    float steplen = 1.0; //
	
    // extend entering, choose its end, we use the opposite
    // we extend thrpugh the tpc
    // if it improves the shape match (CDF maxdist), we keep
    // else we destroy
    // if helps, and we go through the anode, we extend further
    
    // extend the points, to the edge of the detector, only if it improves the shape match
    // we need a direction: pca. for now just use the current pca
    Eigen::Vector3f centerpos( _core->_pca_core.getAvePosition()[0]-_xoffset,
			       _core->_pca_core.getAvePosition()[1],
			       _core->_pca_core.getAvePosition()[2] );
    Eigen::Vector3f pcavec( _core->_pca_core.getEigenVectors()[0][0],
			    _core->_pca_core.getEigenVectors()[1][0],
			    _core->_pca_core.getEigenVectors()[2][0] );

    if ( pcavec.norm()<0.01 ) {
      return;
    }
    
    // determine sign of pca-axis
    float extsign = ( (_botend-centerpos).dot(pcavec) > 0 ) ? 1.0 : -1.0;

    bool indet = true;    
    bool intpc = true;
    bool isanode = false;
    int istep = 0;
    while (indet) {
      Eigen::Vector3f currentpos = _botend + (extsign*steplen*float(istep))*pcavec;
      // check the position
      if ( currentpos(0)<0 ) {
	// went through the anode!
	isanode = true;
	intpc   = false;
      }
      else if ( currentpos(0)>256 || currentpos(1)>117.0 || currentpos(1)<-117 || currentpos(2)<0 || currentpos(2)>1036.0 ) {
	intpc = false;
	isanode = false;
      }

      if ( currentpos(0)<0 || currentpos(0)>256 || currentpos(1)<-137 || currentpos(1)>137 || currentpos(2)<-20 || currentpos(2)>1056.0 ) {
	indet = false;
	break;
      }
      
      if ( indet ) {
	// create the hit
	//std::cout << "create extended hit@ " << currentpos.transpose() << std::endl;
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = currentpos(i);
	qpt.tick   = (currentpos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5) + 3200.0;
	qpt.pixeladc = steplen; 
	qpt.fromplaneid = -1;
	if ( intpc )
	  qpt.type = kGapFill; // we need it to set the light yield according to in-tpc gap fill levels
	else
	  qpt.type = kExt; // we need it to set the light yield according to in-tpc gap fill levels
	_exiting_qcluster.emplace_back( std::move(qpt) );
      }
      istep++;
    }

    // shape tests
    float maxdist_orig = 0.;
    float maxdist_wext = 0.;
    // need hypo
    _exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
    // orig hypo
    float orig_cdf = 0.;
    float ext_cdf = 0.;
    float data_cdf = 0.;
    std::vector<float> orig_v(32,0.0);
    std::vector<float> ext_v(32,0.0);
    std::vector<float> data_v(32,0.0);    
    for (int ich=0; ich<32; ich++) {
      float origpe = _corehypo[ich]*_corehypo.tot;
      float extpe  = origpe + _exiting_hypo[ich]*_exiting_hypo.tot;
      float datape = (*_flashdata)[ich] * _flashdata->tot;
      orig_cdf += origpe;
      ext_cdf  += extpe;
      data_cdf += datape;
      orig_v[ich] = orig_cdf;
      ext_v[ich]  = ext_cdf;
      data_v[ich] = data_cdf;
    }

    if ( orig_cdf>0 && ext_cdf>0 && data_cdf>0 ) {
    
      for (int ich=0; ich<32; ich++) {
	float dist_orig  = fabs(orig_v[ich]/orig_cdf-data_v[ich]/data_cdf);
	float dist_extpe = fabs(ext_v[ich]/ext_cdf-data_v[ich]/data_cdf);
	if ( dist_orig>maxdist_orig )
	  maxdist_orig = dist_orig;
	if ( dist_extpe>maxdist_wext )
	  maxdist_wext = dist_extpe;
      }
      
    }
    else {
      if ( orig_cdf==0 )
	maxdist_orig = 1.0;
      if ( ext_cdf==0 )
	maxdist_wext = 1.0;
    }
    
    if ( maxdist_wext > maxdist_orig ) {
      // worse with extension
      _exiting_hypo.clear();
      _exiting_hypo.resize(32,0);
      _exiting_qcluster.clear();
      return;
    }

    if ( isanode ) {
      // we pierced the anode.
      // we keep stepping until the maxpe is matched (stop if we alreay are over)
      // to do this, we need hypothesis to have been formed
      _exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
      float _maxdiff = _exiting_hypo[_maxch] - _maxch_pe;
      //std::cout << "Is ANODE: extend EXITING to match maxpmt" << std::endl;
      istep = 0;
      while ( _maxdiff<0 ) {
	// we extend the exiting cluster
	Eigen::Map< Eigen::Vector3f > endpos( _exiting_qcluster.back().xyz.data() );
	Eigen::Vector3f extpos = endpos + (extsign*steplen*float(istep))*pcavec;
	// make hit
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = extpos(i);
	qpt.tick = extpos(0)/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200.0;
	qpt.pixeladc = steplen;
	qpt.fromplaneid = -1;
	qpt.type = kExt;
	_exiting_qcluster.emplace_back( std::move(qpt) );
	
	if ( extpos(0) < -50 || extpos(1)>137 || extpos(1)<-137 || extpos(2)<-20 || extpos(2)>1058 ) {
	  // limit this adjustment
	  break;
	}
	
	// update the hypothesis
	_exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
	_maxdiff = _exiting_hypo[_maxch] - _maxch_pe;
	// std::cout << "extpos[0]=" << extpos[0] << " maxdiff=" << _maxdiff
	// 	  << "  _hypomax=" << _exiting_hypo[_maxch] << " " << " _datamax=" << _maxch_pe << std::endl;
	istep++;
      }
    }
    //exiting extension finished
    
    // set final value
    _exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
    
  }

  FlashHypo_t FlashMatchCandidate::getHypothesis( bool withextensions, bool suppresscosmicdisc, float cosmicdiscthresh ) {
    
    FlashHypo_t out;
    out.clusteridx = _cluster->idx;
    out.flashidx   = _flashdata->idx;
    out.tot = 0.;
    out.tot_intpc = 0.;
    out.tot_outtpc = 0.;

    float intpc_tot  = 0.;
    float outtpc_tot = 0.;
    out.resize(32,0);
    for (int ich=0; ich<32; ich++) {
      out[ich]  += _corehypo[ich]*_corehypo.tot;
      intpc_tot += _corehypo[ich]*_corehypo.tot;
      if ( withextensions) {
	out[ich]   += _entering_hypo[ich]*_entering_hypo.tot;
	out[ich]   += _exiting_hypo[ich]*_entering_hypo.tot;
	outtpc_tot += _entering_hypo[ich]*_entering_hypo.tot;
	outtpc_tot += _exiting_hypo[ich]*_entering_hypo.tot;
      }
      out[ich]  += _gapfill_hypo[ich]*_gapfill_hypo.tot;
      intpc_tot += _gapfill_hypo[ich]*_gapfill_hypo.tot;
    }

    float disccosmic_removed = 0.;
    for (int ich=0; ich<32; ich++) {
      if ( suppresscosmicdisc && out[ich] < cosmicdiscthresh ) {
	disccosmic_removed += out[ich];
	out[ich] = 0.;
      }
      out.tot += out[ich];
    }
    out.tot_intpc -= disccosmic_removed;
    
    // finally, norm
    if ( out.tot>0 ) {
      for (int ich=0; ich<32; ich++) {
	out[ich] /= out.tot;
      }
    }
    
    return out;
  }
  
  void FlashMatchCandidate::addMCTrackInfo( const std::vector<larlite::mctrack>& mctrack_v ) {

    _mctrack_v = &mctrack_v;
    
    const ::larutil::TimeService* tsv = ::larutil::TimeService::GetME(false);

    int   nmatch        = 0;
    int   best_trackid  =  -1;      
    int   best_trackidx =  -1;      
    float best_dtick    = 1e9;
    std::vector< int > mctrackid_matches;  // mctrackid
    std::vector< int > mctrackidx_matches; // vector index
    
    int imctrack=-1;    
    for (auto& mct : *_mctrack_v ) {
      imctrack++;

      float track_tick = mct.Start().T()*1.0e-3/0.5 + 3200;
      float flash_time = tsv->OpticalG4Time2TDC( mct.Start().T() );
      
      float dtick = fabs(_flashdata->tpc_tick - track_tick);
      if ( dtick < 5 ) { // space charge?
	nmatch++;
	mctrackidx_matches.push_back( imctrack );
	mctrackid_matches.push_back( mct.TrackID() );
	if ( dtick < best_dtick ) {
	  best_dtick = dtick;
	  best_trackidx = imctrack;
	  best_trackid  = mct.TrackID();
	}
      }

      if ( _cluster->mctrackid==mct.TrackID() ) {
	_cluster_mctrackidx = imctrack;
	_cluster_mctrackid  = mct.TrackID();
	_cluster_mctrack    = &mct;
      }
    }//end of mctrack loop

    // for now, just use the closest match
    _flash_mctrackid  = best_trackid;
    _flash_mctrackidx = best_trackidx;
    if ( best_trackidx>=0 ) {
      _flash_mctrack    = &( _mctrack_v->at( best_trackidx ) );
    }

    //std::cout << "[FlashMatchCandidate::addMCTrackInfo][DEBUG] flashtruth_mctrack=" << _flash_mctrackid << " clustertruth_mctrackid=" << _cluster_mctrackid << std::endl;
  }

  bool FlashMatchCandidate::isTruthMatch() {
    if ( _flash_mctrackid>=0 && _cluster_mctrackid>=0  && _flash_mctrackid==_cluster_mctrackid )
      return true;
    return false;
  }
  
  void FlashMatchCandidate::dumpMatchImage() {
    
    // ===================================================
    // Dump images for debug
    // Each image is for a flash
    //  we plot compatible flashes (according to compat matrix,
    //    so it could be fit to the data)
    //  we also plot the best chi2 and best maxdist match
    //  finally, we plot the truth-matched hypothesis
    // ===================================================

    std::cout << "[FlashMatchCandidate::dumpMatchImage][DEBUG] start" << std::endl;
    
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

    // detector boundary shapes
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
    if ( _hasevstatus ) {
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
    }

    // pmt and channel markers
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
      chmarkers_v[ich] = new TText(xyz[2]-5.0,xyz[1]-5.0,pmtname);
      chmarkers_v[ich]->SetTextSize(0.04);
    }


    // get data histogram
    // -------------------
    TH1D hdata("hdata","",32,0,32);
    float norm = _flashdata->tot;
    for (int ipmt=0; ipmt<32; ipmt++)
      hdata.SetBinContent( ipmt+1, (*_flashdata)[ipmt]*norm );
    hdata.SetLineWidth(4);
    hdata.SetLineColor(kBlack);
    float max = hdata.GetMaximum();

    // get index
    int iclust = _cluster->idx;
    int iflash = _flashdata->idx;
    
    // build hypothesis histogram
    // different components as well
    char hname[20];
    sprintf( hname, "htot_%d", iclust);
    TH1D* hhypotot = new TH1D(hname, "", 32, 0, 32 );
    hhypotot->SetLineWidth(2.0);
    hhypotot->SetLineColor(kBlue);

    sprintf( hname, "hcore_%d", iclust);
    TH1D* hhypocore = new TH1D(hname, "", 32, 0, 32 );
    hhypocore->SetLineColor(kRed);
    hhypocore->SetLineWidth(1.0);

    sprintf( hname, "hentering_%d", iclust);
    TH1D* hhypoenter = new TH1D(hname, "", 32, 0, 32 );
    hhypoenter->SetLineColor(kCyan);
    hhypoenter->SetLineWidth(1.0);
    
    sprintf( hname, "hexiting_%d", iclust);
    TH1D* hhypoexit = new TH1D(hname, "", 32, 0, 32 );
    hhypoexit->SetLineColor(kMagenta);
    hhypoexit->SetLineWidth(1.0);

    sprintf( hname, "hgap_%d", iclust);
    TH1D* hhypogap = new TH1D(hname, "", 32, 0, 32 );
    hhypogap->SetLineColor(kOrange);
    hhypogap->SetLineWidth(1.0);

    std::vector< TH1D* > hclust_v = {hhypocore,hhypoenter,hhypoexit,hhypogap,hhypotot};
    for (int ich=0; ich<32; ich++ ) {
      hhypocore->SetBinContent(ich+1,_corehypo[ich]*_corehypo.tot);
      hhypoenter->SetBinContent(ich+1,_entering_hypo[ich]*_entering_hypo.tot);
      hhypoexit->SetBinContent(ich+1,_exiting_hypo[ich]*_exiting_hypo.tot);
      hhypogap->SetBinContent (ich+1, _gapfill_hypo[ich]*_gapfill_hypo.tot );
      float tot = _corehypo[ich]*_corehypo.tot;
      tot += _gapfill_hypo[ich]*_gapfill_hypo.tot;
      tot += _entering_hypo[ich]*_entering_hypo.tot;
      tot += _exiting_hypo[ich]*_exiting_hypo.tot;
      hhypotot->SetBinContent( ich+1, tot );
    }
    
    // pe ratio of hypothesis
    float hypo_norm = hhypotot->Integral();
    float peratio = fabs(hypo_norm/norm-1.0);
    
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
      
    // draw pmt data markers
    std::vector<TEllipse*> datamarkers_v(32,0);
    std::vector<TEllipse*> hypomarkers_v(32,0);
    //std::vector<TEllipse*> truthmarkers_v(32,0);
      
    for (int ich=0; ich<32; ich++) {
      int opdet = geo->OpDetFromOpChannel(ich);
      double xyz[3];
      geo->GetOpChannelPosition( ich, xyz );
      float pe = (*_flashdata)[ich]*norm;
      if ( pe>10 )
	pe = 10 + (pe-10)*0.10;
      float radius = ( pe>50 ) ? 50 : pe;
      datamarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
      datamarkers_v[ich]->SetFillColor(kRed);
      
      float hypope = (hhypotot->GetBinContent(ich+1));
      if ( hypope>10 )
	hypope = 10 + (hypope-10)*0.10;
      radius = ( hypope>50 ) ? 50 : hypope;
      hypomarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
      hypomarkers_v[ich]->SetFillColor(kGreen+2);
    }
    // if ( truthmatch_hypo ) {
    //   float truthpe = (truthmatch_hypo->at(ich)*truthmatch_hypo->tot);
    //   if ( truthpe>10 )
    // 	    truthpe = 10 + (truthpe-10)*0.10;
    //   float radius = ( truthpe>50 ) ? 50 : truthpe;
    //   truthmarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
    //   truthmarkers_v[ich]->SetFillStyle(0);
    //   truthmarkers_v[ich]->SetLineColor(kMagenta);
    //   truthmarkers_v[ich]->SetLineWidth(3);
    // }	

    // projections for truthmatch cluster
    // TGraph* truthclust_zy[ kNumQTypes ]= {nullptr};
    // TGraph* truthclust_xy[ kNumQTypes ]= {nullptr};
    // int ntruthpts[ kNumQTypes ] = {0};
    // if ( qtruth && qtruth->size()>0 ) {
    //   //std::cout << "qtruth[" << flashdata_v[iflash].truthmatched_clusteridx << "] npoints: " << qtruth->size() << std::endl;
    //   for (int iqt=0; iqt<kNumQTypes; iqt++) {
    // 	truthclust_zy[iqt] = new TGraph(qtruth->size());
    // 	truthclust_xy[iqt] = new TGraph(qtruth->size());
    //   }
    //   float xoffset = (flashdata_v[iflash].tpc_tick-3200)*0.5*driftv;
    //   for (int ipt=0; ipt<(int)qtruth->size(); ipt++) {
    // 	const QPoint_t& truthq = (*qtruth)[ipt];
    // 	truthclust_zy[ truthq.type ]->SetPoint(ntruthpts[truthq.type],truthq.xyz[2], truthq.xyz[1] );
    // 	truthclust_xy[ truthq.type ]->SetPoint(ntruthpts[truthq.type],truthq.xyz[0]-xoffset, truthq.xyz[1] );
    // 	ntruthpts[truthq.type]++;
    //   }
    // 	//std::cout << "qtruth[0] = " << qtruth->at(ipt).xyz[0]-xoffset << " (w/ offset=" << xoffset << ")" << std::endl;
    //   for ( int iqt=0; iqt<kNumQTypes; iqt++ ) {
    // 	truthclust_zy[iqt]->Set( ntruthpts[iqt] );
    // 	truthclust_xy[iqt]->Set( ntruthpts[iqt] );
	
    // 	truthclust_zy[iqt]->Set( ntruthpts[iqt] );
    // 	truthclust_xy[iqt]->Set( ntruthpts[iqt] );
	
    // 	truthclust_zy[iqt]->SetMarkerSize(0.3);
    // 	truthclust_zy[iqt]->SetMarkerStyle( 20 );
	
    // 	truthclust_xy[iqt]->SetMarkerSize(0.3);
    // 	truthclust_xy[iqt]->SetMarkerStyle( 20 );
    //   }
    //   truthclust_zy[kGapFill]->SetMarkerColor(kRed);	
    //   truthclust_xy[kGapFill]->SetMarkerColor(kRed);
    //   truthclust_zy[kExt]->SetMarkerColor(kGreen+3);	
    //   truthclust_xy[kExt]->SetMarkerColor(kGreen+3);
    //   truthclust_zy[kNonCore]->SetMarkerColor(kYellow+2);
    //   truthclust_xy[kNonCore]->SetMarkerColor(kYellow+2);

    // mc-track that is truth matched to flash
    TGraph* mctrack_trueflash_zy = nullptr;
    TGraph* mctrack_trueflash_xy = nullptr;      
    if ( _flash_mctrack ) {
      mctrack_trueflash_zy = new TGraph( _flash_mctrack->size() );
      mctrack_trueflash_xy = new TGraph( _flash_mctrack->size() );	
      for (int istep=0; istep<(int)_flash_mctrack->size(); istep++) {
	mctrack_trueflash_zy->SetPoint(istep, (*_flash_mctrack)[istep].Z(), (*_flash_mctrack)[istep].Y() );
	mctrack_trueflash_xy->SetPoint(istep, (*_flash_mctrack)[istep].X(), (*_flash_mctrack)[istep].Y() );
      }
      mctrack_trueflash_zy->SetLineColor(kBlue);
      mctrack_trueflash_zy->SetLineWidth(1);
      mctrack_trueflash_xy->SetLineColor(kBlue);
      mctrack_trueflash_xy->SetLineWidth(1);
    }

    TGraph* mctrack_trueclust_zy = nullptr;
    TGraph* mctrack_trueclust_xy = nullptr;    
    if ( _cluster_mctrack ) {
      mctrack_trueclust_zy = new TGraph( _cluster_mctrack->size() );
      mctrack_trueclust_xy = new TGraph( _cluster_mctrack->size() );	
      for (int istep=0; istep<(int)_cluster_mctrack->size(); istep++) {
	mctrack_trueclust_zy->SetPoint(istep, (*_cluster_mctrack)[istep].Z(), (*_cluster_mctrack)[istep].Y() );
	mctrack_trueclust_xy->SetPoint(istep, (*_cluster_mctrack)[istep].X(), (*_cluster_mctrack)[istep].Y() );
      }
      mctrack_trueclust_zy->SetLineColor(46);
      mctrack_trueclust_zy->SetLineWidth(1);
      mctrack_trueclust_xy->SetLineColor(46);
      mctrack_trueclust_xy->SetLineWidth(1);
    }
      

    // make graphs for hypothesis cluster
    // ----------------------------------
    TGraph* clust_zy[4] = {nullptr};
    TGraph* clust_xy[4] = {nullptr};
    TGraph* mctrack_hypo_zy = nullptr;
    TGraph* mctrack_hypo_xy = nullptr;
    Int_t clust_colors[4] = { kRed, kOrange, kCyan, kMagenta };
	
    const QCluster_t* clustertypes[4] = { &_offset_qcluster, &_offset_qgap, &_entering_qcluster, &_exiting_qcluster };
    for (int i=0; i<4; i++) {

      
      const QCluster_t* qc = clustertypes[i];
      if (qc==nullptr || qc->size()==0)
	continue;

      std::cout << "make tgraph for type[" << i << "] with " << qc->size() << " points" << std::endl;
      
      clust_zy[i] = new TGraph(qc->size());
      clust_xy[i] = new TGraph(qc->size());

      
      clust_zy[i]->SetMarkerSize(0.3);
      clust_xy[i]->SetMarkerSize(0.3);
      
      clust_zy[i]->SetMarkerStyle(20);
      clust_xy[i]->SetMarkerStyle(20);
      
      for (int ipt=0; ipt<(int)qc->size(); ipt++) {
	const QPoint_t& qpt = (*qc)[ipt];
	clust_zy[i]->SetPoint(ipt,qpt.xyz[2], qpt.xyz[1] );
	clust_xy[i]->SetPoint(ipt,qpt.xyz[0], qpt.xyz[1] );
      }

      clust_zy[i]->SetMarkerColor( clust_colors[i] );
      clust_xy[i]->SetMarkerColor( clust_colors[i] );      
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
      
    // if ( truthclust_zy[kCore] ) {
    //   for (int i=0; i<kNumQTypes; i++) 
    // 	truthclust_zy[i]->Draw("P");
    // }
    if ( mctrack_trueflash_zy )
      mctrack_trueflash_zy->Draw("L");

    for (int ich=0; ich<32; ich++)
      chmarkers_v[ich]->Draw();
    
      
    // XY: DATA
    // --------
    dataxy.cd();
    bgxy.Draw();
    boxxy.Draw();
    // if ( truthclust_xy[kCore] ) {
    //   for (int i=0; i<kNumQTypes; i++)
    // 	truthclust_xy[i]->Draw("P");
    // }
    if ( mctrack_trueflash_xy )    
      mctrack_trueflash_xy->Draw("L");
      
    // YZ-2D plots: hypothesis
    // -----------------------
    hypoyz.cd();
    bg.Draw();
    boxzy.Draw();
    for ( auto& pbadchbox : zy_deadregions ) {
      pbadchbox->Draw();
    }      

    for (int ich=0; ich<32; ich++) {
      hypomarkers_v[ich]->Draw();
      pmtmarkers_v[ich]->Draw();
      // if ( truthmatch_hypo )
      //    truthmarkers_v[ich]->Draw();
    }

    // cluster points
    for (int i=0; i<4; i++) {
      if ( clust_zy[i] )
	clust_zy[i]->Draw("P");
    }
    if ( mctrack_trueclust_zy )    
      mctrack_trueclust_zy->Draw("L");

    for (int ich=0; ich<32; ich++)
      chmarkers_v[ich]->Draw();
                
    hypoxy.cd();
    bgxy.Draw();
    boxxy.Draw();

    // cluster points
    for (int i=0; i<4; i++) {
      if ( clust_xy[i] )
	clust_xy[i]->Draw("P");
    }

    if ( mctrack_trueclust_xy )    
      mctrack_trueclust_xy->Draw("L");    
    
    // finally hist pad
    histpad.cd();
    hdata.Draw("hist");
    for (int ihist=0; ihist<(int)hclust_v.size(); ihist++) {
      hclust_v[ihist]->Draw("hist same");
    }
    
    // if ( tophistidx<hclust_v.size() )
    //   hclust_v[tophistidx]->Draw("hist same");


    // // text summary
    // char ztruth[100];
    // if ( truthmatch_idx<0 )
    // 	sprintf( ztruth,"No truth-match");
    // else
    // 	sprintf( ztruth,"Truth-match idx (green): %d",truthmatch_idx );
    
    // char ztruthscores[100];
    // if ( truthmatch_idx<0 )
    // 	sprintf( ztruthscores, "Truth-Chi2=NA  Truth-Maxdist=NA" );
    // else
    // 	sprintf( ztruthscores, "Truth-Chi2=%.2f  Truth-Maxdist=%.2f peratio=%.2f", truthmatch_chi2, truthmatch_maxdist, truthmatch_peratio );
    
    // char zbestchi[100];
    // if ( bestchi2_idx>=0 )
    // 	sprintf( zbestchi, "Best Chi2 idx (cyan): %d  Chi2=%.1f peratio=%.2f", bestchi2_idx, bestchi2, bestchi2_peratio );
    // else
    // 	sprintf( zbestchi, "No chi2 match" );
    
    // char zbestmaxdist[100];
    // if ( bestmaxdist_idx>=0 )
    // 	sprintf( zbestmaxdist, "Best maxdist idx (blue): %d  maxdist=%.2f peratio=%.2f", bestmaxdist_idx, bestmaxdist, bestmaxdist_peratio );
    // else
    // 	sprintf( zbestmaxdist, "No maxdist match" );
    
    // char zbestfmatch[100];
    // if ( usefmatch )
    // 	sprintf( zbestfmatch, "Best fmatch index: %d fmatch=%.2f", bestfmatch_idx, matchscore_best );
    
    // TText ttruth(0.6,0.85,ztruth);
    // TText ttruthscore(0.6,0.80,ztruthscores);
    // TText tbestchi(0.6,0.75,zbestchi);
    // TText tbestmaxdist(0.6,0.70,zbestmaxdist);
    // TText* tbestfmatch = nullptr;
    // if ( usefmatch )
    // 	tbestfmatch = new TText(0.6,0.65,zbestfmatch);
    
    // ttruth.SetNDC(true);
    // ttruth.Draw();
    // ttruthscore.SetNDC(true);      
    // ttruthscore.Draw();
    // tbestchi.SetNDC(true);
    // tbestchi.Draw();
    // tbestmaxdist.SetNDC(true);
    // tbestmaxdist.Draw();
    // if ( usefmatch ) {
    // 	tbestfmatch->SetNDC(true);
    // 	tbestfmatch->Draw();
    // }
    
    c2d.Update();
    char cname[100];
    sprintf(cname,"hflashmatchcand_flashid%02d_clusterid%02d.png",iflash,iclust);
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
    for (int i=0; i<4; i++) {
    //   if ( truthclust_zy[i] )
    // 	delete truthclust_zy[i];
    //   if ( truthclust_xy[i] )
    // 	delete truthclust_xy[i];
      if ( clust_zy[i] )
    	delete clust_zy[i];
      if ( clust_xy[i] )
    	delete clust_xy[i];
    // }
    // if (mctrack_data)
    //   delete mctrack_data;
    // if ( mctrack_data_xy )
    //   delete mctrack_data_xy;
    
    // if ( mctrack_hypo_zy && _iclust!=truthmatch_idx ) {
    //   delete mctrack_hypo_zy;
    //   delete mctrack_hypo_xy;
    }

    if ( mctrack_trueflash_zy ) {
      delete mctrack_trueflash_zy;
      delete mctrack_trueflash_xy;
    }
    if ( mctrack_trueclust_zy ) {
      delete mctrack_trueclust_zy;
      delete mctrack_trueclust_xy;
    }
      
    for (int ic=0; ic<(int)hclust_v.size(); ic++) {
      delete hclust_v[ic];
    }
    hclust_v.clear();      
    
    // delete tbestfmatch;
    // tbestfmatch = nullptr;
    
  }//end of dumpimage

}
