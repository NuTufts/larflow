#include "FlashMatchCandidate.h"

#include <Eigen/Dense>

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
    _core(&qcoredata)
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

    // we build the core hypo
    _corehypo = buildFlashHypothesis( *_flashdata, _core->_core );

    // we identify the max pmt and its value -- we will use this to tune extensions
    _maxch = 0;
    _maxch_pe = 0;
    for (int ich=0; ich<(int)_flashdata->size(); ich++) {
      if ( _flashdata->at(ich)>_maxch_pe ) {
	_maxch = ich;
	_maxch_pe = _flashdata->at(ich);
      }
    }
  }

  
  void FlashMatchCandidate::ExtendEnteringEnd() {
    // we have to find which end we think is entering (cosmic assumptoin)
    // we pick the top-most end
    
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
	avepos[1][i] += core[(int)nptscore-ipt].xyz[i]/float(npoint_ave);
      avepos[0][3] += core[ipt].tick/float(npoint_ave);
      avepos[1][3] += core[(int)nptscore-ipt].tick/float(npoint_ave);			   
    }
    avepos[0][0] -= _xoffset;
    avepos[1][0] -= _xoffset;    
    
    // choose the top-most end
    _usefront = 0;
    Eigen::Vector3f topend(0,0,0);
    float toptick = 0;
    if ( avepos[0][1]>avepos[1][1] ) {
      for (int i=0; i<3; i++) topend(i) = avepos[0][i];
      _usefront = 1;
      toptick = avepos[0][3];
    }
    else {
      for (int i=0; i<3; i++) topend(i) = avepos[1][i];
      _usefront = 0;
      toptick = avepos[1][3];
    }

    // extend the points, to the edge of the detector
    // we need a direction: pca. for now just use the current pca
    Eigen::Vector3f centerpos( _core->_pca_core.getAvePosition()[0]-_xoffset,
			       _core->_pca_core.getAvePosition()[1],
			       _core->_pca_core.getAvePosition()[2] );
    Eigen::Vector3f pcavec( _core->_pca_core.getEigenVectors()[0][0],
			    _core->_pca_core.getEigenVectors()[1][0],
			    _core->_pca_core.getEigenVectors()[2][0] );
    // determine sign of pca-axis
    float extsign = ( (topend-centerpos).dot(pcavec) > 0 ) ? 1.0 : -1.0;

    bool intpc = true;
    bool isanode = false;
    float steplen = 3.0; // 
    _entering_qcluster.clear();
    while (intpc) {
      Eigen::Vector3f currentpos = topend + extsign*steplen*pcavec;
      // check the position
      if ( currentpos(0)<0 ) {
	// went through the anode!
	isanode = true;
	intpc   = false;
      }
      else if ( currentpos(0)>250 || currentpos(1)>120.0 || currentpos(1)<-120 ) {
	intpc = false;
      }
      
      if ( intpc ) {
	// create the hit
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = currentpos(i);
	qpt.tick   = (currentpos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5) + 3200.0;
	qpt.pixeladc = steplen; 
	qpt.fromplaneid = -1;
	qpt.type = kGapFill; // we need it to set the light yield according to in-tpc gap fill levels
	_entering_qcluster.emplace_back( std::move(qpt) );
      }
      
    }

    if ( isanode ) {
      // we pierced the anode.
      // we keep stepping until the maxpe is matched (stop if we alreay are over)
      // to do this, we need hypothesis to have been formed
      _entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster );
      float _maxdiff = _entering_hypo[_maxch] - _maxch_pe;
      while ( _maxdiff<0 ) {
	// we extend the entering cluster
	Eigen::Map< Eigen::Vector3f > endpos( _entering_qcluster.back().xyz.data() );
	Eigen::Vector3f extpos = endpos + (extsign*steplen)*pcavec;
	// make hit
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = extpos(i);
	qpt.tick = extpos(0)/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200.0;
	qpt.pixeladc = steplen;
	qpt.fromplaneid = -1;
	qpt.type = kExt;
	_entering_qcluster.emplace_back( std::move(qpt) );

	if ( extpos(0) < -20 ) {
	  // limit this adjustment
	  break;
	}

	// update the hypothesis
	_entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster );
	_maxdiff = _entering_hypo[_maxch] - _maxch_pe;
      }
    }
    //entering extension finished
  }

  void FlashMatchCandidate::ExtendExitingEnd() {
    // now the exiting end.
    // we only keep extending if it improves the match
    
  }

  FlashHypo_t FlashMatchCandidate::buildFlashHypothesis( const FlashData_t& flashdata, const QCluster_t&  qcluster ) {
    
    // each cluster builds a hypothesis for each compatible flash
    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float pixval2photons = (2.2/40)*0.3*40000*0.5*0.01; // [mip mev/cm]/(adc/MeV)*[pixwidth cm]*[phot/MeV]*[pe/phot] this is a WAG!!!
    const float gapfill_len2adc  = (60.0/0.3); // adc value per pixel for mip going 0.3 cm through pixel, factor of 2 for no field        
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
      xyz[0] = qhit.xyz[0]; // assumes I've already removed the xoffset
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
  
  
}
