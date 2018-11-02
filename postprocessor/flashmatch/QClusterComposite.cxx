#include "QClusterComposite.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"


namespace larflow {

  QClusterComposite::QClusterComposite( const QCluster_t& source_cluster ) :
    _cluster(&source_cluster),
    _core(source_cluster),
    _posfront(0,0,0),
    _posback(0,0,0)
  {

    _fExtStepLen = 3.0;
    
    // impicit is that the core is built

    // extend the points, to the edge of the detector
    // we need a direction: pca. for now just use the current pca from the core
    _centerpos = Eigen::Vector3f( _core._pca_core.getAvePosition()[0],
				  _core._pca_core.getAvePosition()[1],
				  _core._pca_core.getAvePosition()[2] );
    _pcavec = Eigen::Vector3f( _core._pca_core.getEigenVectors()[0][0],
			       _core._pca_core.getEigenVectors()[1][0],
			       _core._pca_core.getEigenVectors()[2][0] );

    bool pcaok = true;
    for ( int i=0; i<3; i++) {
      if ( std::isnan(_pcavec(i)) )
	pcaok = false;
    }

    if ( pcaok ) {

      Eigen::ParametrizedLine< float, 3 > pcline( _centerpos, _pcavec );
      
      
      // we mark the end of the clusters as the projection of the last core point onto the 1st pc
      Eigen::Map< Eigen::Vector3f > ptfront( _core._core.front().xyz.data() );
      Eigen::Map< Eigen::Vector3f > ptback(  _core._core.back().xyz.data() );
      
      // projection    
      _posfront = pcline.projection( ptfront );
      _posback  = pcline.projection( ptback );
      
      // extension
      ExtendEnteringEnd();
      ExtendExitingEnd();

    }
    
  }

  void QClusterComposite::ExtendEnteringEnd() {
    
    // we have to find which end we think is entering (cosmic assumptoin)
    // we pick the top-most end
    _entering_qcluster.clear();
	
    // choose the top-most end
    bool frontistop = (_posfront(1)>=_posback(1) );
    
    Eigen::Vector3f topend = ( frontistop ) ? _posfront : _posback;
    Eigen::Vector3f botend = ( frontistop ) ? _posback  : _posfront;
    
    // determine sign of pca-axis
    float extsign = ( (topend-_centerpos).dot(_pcavec) > 0 ) ? 1.0 : -1.0;

    // how long do we need to extend?
    float xoffset = 0;
    if ( _pcavec(0) > 0 ) {
      // we shift the closest end to 0, back to the origin
      xoffset = -botend(0);
    }
    else {
      // we shift the closest end to the cathode back to the cathode
      xoffset = 256.0 - botend(0);
    }

    bool indet = true;
    int istep = 0;
    while ( indet )  {
      Eigen::Vector3f currentpos = topend + (extsign*_fExtStepLen*(float(istep)+0.5))*_pcavec;
      currentpos(0) += xoffset;

      if ( currentpos(0)>256 || currentpos(0)<-50 ||
	   currentpos(1)>137.0 || currentpos(1)<-137
	   || currentpos(2)<-20 || currentpos(2)>1056.0 ) {
	indet = false;
      }
      
      if ( indet ) {
	// create the hit
	//std::cout << "create extended hit@ " << currentpos.transpose() << " pcavec=" << pcavec.transpose() << std::endl;
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = currentpos(i);
	qpt.xyz[0] -= xoffset;  // remove in the final position
	qpt.tick   = (qpt.xyz[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5) + 3200.0;
	qpt.pixeladc = _fExtStepLen; 
	qpt.fromplaneid = -1;
	qpt.type = kExt;
	_entering_qcluster.emplace_back( std::move(qpt) );
      }
      istep++;
    }//end of while loop
    
    //entering extension finished
    return;
  }

  void QClusterComposite::ExtendExitingEnd() {

    // we have to find which end we think is entering (cosmic assumptoin)
    // we pick the top-most end
    _exiting_qcluster.clear();
	
    // choose the bottom-most end
    bool frontisbot = (_posfront(1)<=_posback(1));
    
    Eigen::Vector3f botend = ( frontisbot ) ? _posfront  : _posback;
    Eigen::Vector3f topend = ( frontisbot ) ? _posback : _posfront;
    
    // determine sign of pca-axis
    float extsign = ( (botend-_centerpos).dot(_pcavec) > 0 ) ? 1.0 : -1.0;
    
    // how long do we need to extend?
    float xoffset = 0;
    if ( _pcavec(0) > 0 ) {
      // we shift the closest end to 0, back to the origin
      xoffset = -topend(0);
    }
    else {
      // we shift the closest end to the cathode back to the cathode
      xoffset = 256.0-topend(0);
    }
    
    bool indet = true;
    int istep = 0;
    while ( indet )  {
      Eigen::Vector3f currentpos = botend + (extsign*_fExtStepLen*(float(istep)+0.5))*_pcavec;
      currentpos(0) += xoffset;

      if ( currentpos(0)>256 || currentpos(0)<-50 ||
	   currentpos(1)>137.0 || currentpos(1)<-137
	   || currentpos(2)<-20 || currentpos(2)>1056.0 ) {
	indet = false;
      }
      
      if ( indet ) {
	// create the hit
	//std::cout << "create extended hit@ " << currentpos.transpose() << " pcavec=" << pcavec.transpose() << std::endl;
	QPoint_t qpt;
	qpt.xyz.resize(3,0);
	for (int i=0; i<3; i++) qpt.xyz[i] = currentpos(i);
	qpt.xyz[0] -= xoffset;
	qpt.tick   = (qpt.xyz[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5) + 3200.0;
	qpt.pixeladc = _fExtStepLen; 
	qpt.fromplaneid = -1;
	qpt.type = kExt;
	_exiting_qcluster.emplace_back( std::move(qpt) );
      }
      istep++;
    }//end of while loop

    //entering extension finished
    return;
  }

  void QClusterComposite::generateFlashCompositeHypo( const FlashData_t& flash ) {
    
    // we use the hypothesis to define the offset
    float xoffset = (flash.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();

    // we generate the four componet hypotheses: core, gap, entering, exiting

    const larutil::Geometry* geo = larutil::Geometry::GetME();
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float pixval2photons = (2.2/40)*0.3*40000*0.5*0.01; // [mip mev/cm]/(adc/MeV)*[pixwidth cm]*[phot/MeV]*[pe/phot] this is a WAG!!!
    const float gapfill_len2adc  = (80.0/0.3); // adc value per pixel for mip going 0.3 cm through pixel
    const float outoftpc_len2adc = 2.0*gapfill_len2adc; // adc value per pixel for mip going 0.3 cm through pixel, factor of 2 for no field

    const QCluster_t* qclusters[4] = { &_core._core, &_core._gapfill_qcluster, &_entering_qcluster, &_exiting_qcluster };
    
    FlashHypo_t hypo;
    hypo.resize(npmts,0.0);
    hypo.clusteridx = _cluster->idx; 
    hypo.flashidx   = flash.idx;
    hypo.tot_intpc = 0.;
    hypo.tot_outtpc = 0.;
    float norm = 0.0;
    for ( int i=0; i<4; i++) {
      // add different contributions
      const QCluster_t& qcluster = *(qclusters[i]);
      for ( size_t ihit=0; ihit<qcluster.size(); ihit++ ) {
	const QPoint_t& qhit = qcluster[ihit];
	double xyz[3];
	xyz[0] = qhit.xyz[0] - xoffset;
	xyz[1] = qhit.xyz[1];
	xyz[2] = qhit.xyz[2];
      
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
	      throw std::runtime_error("[QClusterComposite::buildFlashCompositeHypo][ERROR] unrecognized qpoint type");
	    }
	    
	    hypo[ich] += pe;
	    norm += pe;
	  }
	}
	else if ( vis->size()>0 && vis->size()!=npmts ) {
	  throw std::runtime_error("[QClusterComposite::buildFlashCompositeHypo][ERROR] unexpected visibility size");
	}

      }//end of hit loop
      
    }//end of component loops
	
    // normalize
    hypo.tot = norm;
    if ( norm>0 ) {
      for (size_t ich=0; ich<hypo.size(); ich++)
	hypo[ich] /= norm;
    }
    
  }

  std::vector< TGraph > QClusterComposite::getTGraphs( float xoffset ) const {
    const QCluster_t* qclusters[4] = { &_core._core, &_core._gapfill_qcluster, &_entering_qcluster, &_exiting_qcluster };

    std::vector< TGraph > graph_v;
    for (int i=0; i<4; i++) {
      const QCluster_t& cluster = *(qclusters[i]);
      TGraph comp_xy( cluster.size() );
      TGraph comp_zy( cluster.size() );      
      for ( size_t i=0; i<cluster.size(); i++ ) {
	const QPoint_t& qpt = cluster[i];
	comp_xy.SetPoint( i, qpt.xyz[0]-xoffset, qpt.xyz[1] );
	comp_zy.SetPoint( i, qpt.xyz[2], qpt.xyz[1] );
      }
      graph_v.emplace_back( std::move(comp_zy) );
      graph_v.emplace_back( std::move(comp_xy) );      
    }

    return graph_v;
  }
  
  //   float steplen = 1.0; //
	
  //   // extend entering, choose its end, we use the opposite
  //   // we extend thrpugh the tpc
  //   // if it improves the shape match (CDF maxdist), we keep
  //   // else we destroy
  //   // if helps, and we go through the anode, we extend further
    
  //   // extend the points, to the edge of the detector, only if it improves the shape match
  //   // we need a direction: pca. for now just use the current pca
  //   Eigen::Vector3f centerpos( _core->_pca_core.getAvePosition()[0]-_xoffset,
  // 			       _core->_pca_core.getAvePosition()[1],
  // 			       _core->_pca_core.getAvePosition()[2] );
  //   Eigen::Vector3f pcavec( _core->_pca_core.getEigenVectors()[0][0],
  // 			    _core->_pca_core.getEigenVectors()[1][0],
  // 			    _core->_pca_core.getEigenVectors()[2][0] );

  //   if ( pcavec.norm()<0.5 ) {
  //     return;
  //   }

  //   for (int i=0; i<3; i++) {
  //     if ( std::isnan(pcavec(i)) )
  // 	return;
  //   }
    
  //   // determine sign of pca-axis
  //   float extsign = ( (_botend-centerpos).dot(pcavec) > 0 ) ? 1.0 : -1.0;

  //   bool indet = true;    
  //   bool intpc = true;
  //   bool isanode = false;
  //   int istep = 0;
  //   while (indet) {
  //     Eigen::Vector3f currentpos = _botend + (extsign*steplen*float(istep))*pcavec;
  //     // check the position
  //     if ( currentpos(0)<0 ) {
  // 	// went through the anode!
  // 	isanode = true;
  // 	intpc   = false;
  //     }
  //     else if ( currentpos(0)>256 || currentpos(1)>117.0 || currentpos(1)<-117 || currentpos(2)<0 || currentpos(2)>1036.0 ) {
  // 	intpc = false;
  // 	isanode = false;
  //     }

  //     if ( currentpos(0)<0 || currentpos(0)>256 || currentpos(1)<-137 || currentpos(1)>137 || currentpos(2)<-20 || currentpos(2)>1056.0 ) {
  // 	indet = false;
  // 	break;
  //     }
      
  //     if ( indet ) {
  // 	// create the hit
  // 	//std::cout << "create extended hit@ " << currentpos.transpose() << " pcavec=" << pcavec.transpose() << std::endl;	
  // 	QPoint_t qpt;
  // 	qpt.xyz.resize(3,0);
  // 	for (int i=0; i<3; i++) qpt.xyz[i] = currentpos(i);
  // 	qpt.tick   = (currentpos[0]/larutil::LArProperties::GetME()->DriftVelocity()/0.5) + 3200.0;
  // 	qpt.pixeladc = steplen; 
  // 	qpt.fromplaneid = -1;
  // 	if ( intpc )
  // 	  qpt.type = kGapFill; // we need it to set the light yield according to in-tpc gap fill levels
  // 	else
  // 	  qpt.type = kExt; // we need it to set the light yield according to in-tpc gap fill levels
  // 	_exiting_qcluster.emplace_back( std::move(qpt) );
  //     }
  //     istep++;
  //   }

  //   // shape tests
  //   float maxdist_orig = 0.;
  //   float maxdist_wext = 0.;
  //   // need hypo
  //   _exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
  //   // orig hypo
  //   float orig_cdf = 0.;
  //   float ext_cdf = 0.;
  //   float data_cdf = 0.;
  //   std::vector<float> orig_v(32,0.0);
  //   std::vector<float> ext_v(32,0.0);
  //   std::vector<float> data_v(32,0.0);    
  //   for (int ich=0; ich<32; ich++) {
  //     float origpe = _corehypo[ich]*_corehypo.tot;
  //     float extpe  = origpe + _exiting_hypo[ich]*_exiting_hypo.tot;
  //     float datape = (*_flashdata)[ich] * _flashdata->tot;
  //     orig_cdf += origpe;
  //     ext_cdf  += extpe;
  //     data_cdf += datape;
  //     orig_v[ich] = orig_cdf;
  //     ext_v[ich]  = ext_cdf;
  //     data_v[ich] = data_cdf;
  //   }

  //   if ( orig_cdf>0 && ext_cdf>0 && data_cdf>0 ) {
    
  //     for (int ich=0; ich<32; ich++) {
  // 	float dist_orig  = fabs(orig_v[ich]/orig_cdf-data_v[ich]/data_cdf);
  // 	float dist_extpe = fabs(ext_v[ich]/ext_cdf-data_v[ich]/data_cdf);
  // 	if ( dist_orig>maxdist_orig )
  // 	  maxdist_orig = dist_orig;
  // 	if ( dist_extpe>maxdist_wext )
  // 	  maxdist_wext = dist_extpe;
  //     }
      
  //   }
  //   else {
  //     if ( orig_cdf==0 )
  // 	maxdist_orig = 1.0;
  //     if ( ext_cdf==0 )
  // 	maxdist_wext = 1.0;
  //   }
    
  //   if ( maxdist_wext > maxdist_orig ) {
  //     // worse with extension
  //     _exiting_hypo.clear();
  //     _exiting_hypo.resize(32,0);
  //     _exiting_qcluster.clear();
  //     return;
  //   }

  //   if ( isanode ) {
  //     // we pierced the anode.
  //     // we keep stepping until the maxpe is matched (stop if we alreay are over)
  //     // to do this, we need hypothesis to have been formed
  //     _exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
  //     float _maxdiff = _exiting_hypo[_maxch] - _maxch_pe;
  //     //std::cout << "Is ANODE: extend EXITING to match maxpmt" << std::endl;
  //     istep = 0;
  //     while ( _maxdiff<0 ) {
  // 	// we extend the exiting cluster
  // 	Eigen::Map< Eigen::Vector3f > endpos( _exiting_qcluster.back().xyz.data() );
  // 	Eigen::Vector3f extpos = endpos + (extsign*steplen)*pcavec;
  // 	// make hit
  // 	QPoint_t qpt;
  // 	qpt.xyz.resize(3,0);
  // 	for (int i=0; i<3; i++) qpt.xyz[i] = extpos(i);
  // 	qpt.tick = extpos(0)/larutil::LArProperties::GetME()->DriftVelocity()/0.5 + 3200.0;
  // 	qpt.pixeladc = steplen;
  // 	qpt.fromplaneid = -1;
  // 	qpt.type = kExt;
  // 	_exiting_qcluster.emplace_back( std::move(qpt) );
	
  // 	if ( extpos(0) < -50 || extpos(0)>1 || extpos(1)>137 || extpos(1)<-137 || extpos(2)<-20 || extpos(2)>1058 ) {
  // 	  // limit this adjustment
  // 	  break;
  // 	}
	
  // 	// update the hypothesis
  // 	_exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
  // 	_maxdiff = _exiting_hypo[_maxch] - _maxch_pe;
  // 	// std::cout << "extpos[0]=" << extpos[0] << " maxdiff=" << _maxdiff
  // 	// 	  << "  _hypomax=" << _exiting_hypo[_maxch] << " " << " _datamax=" << _maxch_pe << std::endl;
  // 	istep++;
  //     }
  //   }
  //   //exiting extension finished
    
  //   // set final value
  //   _exiting_hypo = buildFlashHypothesis( *_flashdata, _exiting_qcluster, 0.0 );
    
  // }
  
}
