#include "QClusterComposite.h"

// larlite
#include "LArUtil/Geometry.h"
#include "LArUtil/LArProperties.h"
#include "LArUtil/TimeService.h"
#include "SelectionTool/OpT0Finder/PhotonLibrary/PhotonVisibilityService.h"

#include "FlashMatchCandidate.h"

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
    // we want to shift so that vec has longest path out of detector
    float xoffset = 0;
    if ( extsign > 0 ) {
      // pca goes to topend
      if ( _pcavec(0)>0 ) // top end is in backhalf
	xoffset = -botend(0); // shift botend to anode
      else
	xoffset = 256.0 - botend(0);	// shift botend to cathode
    }
    else {
      // pca goes to botend
      if ( _pcavec(0)<0 ) // top end is in backhalf
	xoffset = -botend(0);
      else
	xoffset = 256.0 - botend(0);	      
    }

    bool indet = true;
    int istep = 0;
    while ( indet )  {
      Eigen::Vector3f currentpos = topend + (extsign*_fExtStepLen*(float(istep)+0.5))*_pcavec;
      currentpos(0) += xoffset;

      if ( currentpos(0)>280 || currentpos(0)<-50 ||
	   currentpos(1)>137.0 || currentpos(1)<-137
	   || currentpos(2)<-50 || currentpos(2)>1086.0 ) {
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
    if ( extsign > 0 ) {
      // pca goes to botend
      if ( _pcavec(0)>0 ) // top end is in backhalf
	xoffset = -topend(0); // shift botend to anode
      else
	xoffset = 256.0 - topend(0);	// shift botend to cathode
    }
    else {
      // pca goes to topend
      if ( _pcavec(0)<0 ) // top end is in backhalf
	xoffset = -topend(0);
      else
	xoffset = 256.0 - topend(0);	      
    }
    
    bool indet = true;
    int istep = 0;
    while ( indet )  {
      Eigen::Vector3f currentpos = botend + (extsign*_fExtStepLen*(float(istep)+0.5))*_pcavec;
      currentpos(0) += xoffset;

      if ( currentpos(0)>300 || currentpos(0)<-80 ||
	   currentpos(1)>137.0 || currentpos(1)<-137
	   || currentpos(2)<-50 || currentpos(2)>1086.0 ) {
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

  FlashCompositeHypo_t QClusterComposite::generateFlashCompositeHypo( const FlashData_t& flash, const bool apply_ext, const bool use_orig_qcluster ) const {
    
    // we use the hypothesis to define the offset
    float xoffset = (flash.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();

    // we generate the four componet hypotheses: core, gap, entering, exiting

    const larutil::Geometry* geo = larutil::Geometry::GetME();
    //const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV.root" );
    const phot::PhotonVisibilityService& photonlib = phot::PhotonVisibilityService::GetME( "uboone_photon_library_v6_70kV_EnhancedExtraTPCVis.root" );
    const larutil::LArProperties* larp = larutil::LArProperties::GetME();
    const float driftv = larp->DriftVelocity();
    const size_t npmts = 32;
    const float mev_per_pixval = 1.0/50.0; // WAG
    const float pixval2photons = mev_per_pixval*40000*0.25*0.5*0.01; // [ mev/adc]*[phot/MeV]*[early frac muon]*[field quenching]*[pe/phot] this is a WAG!!!
    const float gapfill_len2adc  = 3.5*(2.5/mev_per_pixval); // adc value per pixel for mip going 0.3 cm through pixel
    const float outoftpc_len2adc = 2.0*gapfill_len2adc; // adc value per pixel for mip going 0.3 cm through pixel, factor of 2 for no field

    const QCluster_t* qclusters[4] = { &_core._core, &_core._gapfill_qcluster, &_entering_qcluster, &_exiting_qcluster };
    if ( use_orig_qcluster ) {
      qclusters[0] = _cluster;
    }

    bool extend_using_maxpe = true; // otherwise using chi2
    
    FlashCompositeHypo_t hypo_composite;
    hypo_composite.nenter_used = 0;
    hypo_composite.nexit_used  = 0;

    float debug_totq = 0.;
    float debug_totintpc = 0.;
    
    // hypo for core+gaps and enter inside the det
    for ( int icomp=0; icomp<3; icomp++) {

      if ( !apply_ext && icomp==2 )
	break;

      const QCluster_t& qcluster = *(qclusters[icomp]);
      FlashHypo_t* h = nullptr;
      switch( icomp ) {
      case 0:
	h = &(hypo_composite.core);
	break;
      case 1:
	h = &(hypo_composite.gap);
	break;
      case 2:
	h = &(hypo_composite.enter);
	break;
      };
      
      for ( size_t ihit=0; ihit<qcluster.size(); ihit++ ) {
      	const QPoint_t& qhit = qcluster[ihit];
      	double xyz[3];
      	xyz[0] = qhit.xyz[0] - xoffset;
      	xyz[1] = qhit.xyz[1];
      	xyz[2] = qhit.xyz[2];
      
       	if ( xyz[0]>256.0 )
       	  continue; // i dont trust the hypotheses here

	if ( icomp==2 ) {
	  // ignore out of TPC for entering extension
	  if ( xyz[0]<0 || xyz[0]>256.0 || 
	       xyz[1]<-117.0 || xyz[1]>117.0 ||
	       xyz[2]<0 || xyz[2]>1036.0 ) {
	    break;
	  }
	  hypo_composite.nenter_used++;	  
	}
	
       	const std::vector<float>* vis = photonlib.GetAllVisibilities( xyz );
      
      	if ( vis && vis->size()==npmts) {
       	  for (int ich=0; ich<npmts; ich++) {
	    float q  = qhit.pixeladc;
	    float pe = 0.;
	    switch(icomp) {
	    case 0:
	      //core
	      pe = q*pixval2photons*(*vis)[ geo->OpDetFromOpChannel( ich ) ];
	      debug_totq += q;	      
	      break;
	    case 1:
	    case 2:
	      // gaps for inside tpc and entering extension inside the tpc
	      pe = q*gapfill_len2adc*(*vis)[geo->OpDetFromOpChannel(ich)];
	      break;
	    }
	    h->tot_intpc += pe;
	    h->tot       += pe;
	    (*h)[ich] += pe;
	  }
	}
       	else if ( vis->size()>0 && vis->size()!=npmts ) {
       	  throw std::runtime_error("[QClusterComposite::buildFlashCompositeHypo][ERROR] unexpected visibility size");
	}
	
      }//end of hit loop
      
    }/// end of (core,gap,enter-in-tpc) loop

    //std::cout << "debug: core q=" << debug_totq << std::endl;

    if (!apply_ext)
      return hypo_composite;

    // entering portion outside of the tpc
    // we use the entire contribution until we get to the detector, then we continue to add if we improve the maxdist
    FlashHypo_t pre_enter_outside = hypo_composite.makeHypo();
    // compare
    float current_maxdist = FlashMatchCandidate::getMaxDist( flash, pre_enter_outside );
    float current_maxperatio = calcMaxPEratio( flash, pre_enter_outside );    
    const QCluster_t& qenter = *(qclusters[2]);
    //std::cout << "debug: pre-enter peratio=" << current_maxperatio << " (enter-outtpc-start=" << hypo_composite.nenter_used << " enter size=" << qenter.size() << ")" << std::endl;    
    float nenter_pe = 0.;
    for ( int ihit=hypo_composite.nenter_used; ihit<(int)qenter.size(); ihit++ ) {
      const QPoint_t& qhit = qenter[ihit];
      double xyz[3];
      xyz[0] = qhit.xyz[0] - xoffset;
      xyz[1] = qhit.xyz[1];
      xyz[2] = qhit.xyz[2];

      if ( xyz[0]>256 )
	break;
      
      const std::vector<float>* vis = photonlib.GetAllVisibilities( xyz );      
      
      std::vector<float> dpe_v(32,0);
      float dpe_tot = 0.;
      if ( vis==NULL || vis->size()==0 ) {
	for (size_t ich=0; ich<32; ich++) {
	  float q  = qhit.pixeladc;
	  float pe = q*outoftpc_len2adc*(*vis)[geo->OpDetFromOpChannel(ich)];
	  pre_enter_outside[ich] += pe;
	  pre_enter_outside.tot += pe;
	  pre_enter_outside.tot_outtpc += pe;
	  dpe_v[ich] = pe;
	  dpe_tot += pe;
	}
      }
      
      if ( !extend_using_maxpe ) {
	// use max dist to extend
	float maxdist = FlashMatchCandidate::getMaxDist( flash, pre_enter_outside, false );
	if ( maxdist-0.05 > current_maxdist ) {
	  break;
	}
      }
      else {
	// use peratio of max pmt in data flash
	// good proxy for extension into anode?
	float maxperatio = calcMaxPEratio( flash, pre_enter_outside );
	//std::cout << "debug: continue-enter. new point=" << fabs(maxperatio) << " vs. " << fabs(current_maxperatio) << std::endl;	
	if ( fabs(maxperatio) < fabs(current_maxperatio) ) {
	  current_maxperatio = maxperatio;
	}
	else {
	  // stop entering extension
	  break;
	}
      }
      
      // else, update the composite
      for ( size_t ich=0; ich<32; ich++ ) {
	hypo_composite.enter[ich] += dpe_v[ich];
      }
      hypo_composite.tot += dpe_tot;
      hypo_composite.tot_outtpc += dpe_tot;
      hypo_composite.nenter_used++;
      nenter_pe += dpe_tot;
    }
    
    // std::cout << "[QClusterComposite::generateFlashCompositeHypo][ERROR] "
    // 	      << " nenter_used=" << nenter_used << " of qexit.size=" << qenter.size()
    // 	      << " padded=" << nenter_pe
    // 	      << std::endl;

    
    // now the exiting extension
    // only extend if we improve the comparison
    FlashHypo_t pre_exit_outside = hypo_composite.makeHypo();
    float pe_added = 0;
    current_maxdist = FlashMatchCandidate::getMaxDist( flash, pre_exit_outside );
    current_maxperatio = calcMaxPEratio( flash, pre_exit_outside );
    const QCluster_t& qexit = *(qclusters[3]);
    for ( int ihit=0; ihit<(int)qexit.size(); ihit++ ) {
      const QPoint_t& qhit = qexit[ihit];
      double xyz[3];
      xyz[0] = qhit.xyz[0] - xoffset;
      xyz[1] = qhit.xyz[1];
      xyz[2] = qhit.xyz[2];

      if ( xyz[0]>256 )
	break;

      bool intpc = true;
      if ( xyz[0]<0 || xyz[0]>256.0 || 
	   xyz[1]<-117.0 || xyz[1]>117.0 ||
	   xyz[2]<0 || xyz[2]>1036.0 ) {
	intpc = false;
      }
      
      const std::vector<float>* vis = photonlib.GetAllVisibilities( xyz );
      std::vector<float> dpe_v(32,0);
      float dpe_tot = 0.;      

      if ( vis && vis->size()==npmts) {
	for (size_t ich=0; ich<32; ich++) {
	  float q  = qhit.pixeladc;
	  float pe = 0;
	  if ( intpc )
	    pe = q*gapfill_len2adc*(*vis)[geo->OpDetFromOpChannel(ich)];
	  else
	    pe = q*outoftpc_len2adc*(*vis)[geo->OpDetFromOpChannel(ich)]; // fudge
	  pre_exit_outside[ich] += pe;
	  pre_exit_outside.tot += pe;
	  pre_exit_outside.tot_outtpc += pe;
	  dpe_v[ich] = pe;
	  dpe_tot += pe;
	  pe_added += pe;
	}
      }//end of vis loop
    

      if ( !extend_using_maxpe ) {
	float maxdist = FlashMatchCandidate::getMaxDist( flash, pre_exit_outside, false );
	if ( maxdist-0.05 > current_maxdist ) {
	  // stop
	  break;
	}
      }
      else {
	float maxperatio = calcMaxPEratio( flash, pre_exit_outside );
	if ( fabs(maxperatio) < fabs(current_maxperatio ) ) {
	  current_maxperatio = maxperatio;
	}
	else {
	  break;
	}
      }
      
      // else, update the composite
      for ( size_t ich=0; ich<32; ich++ ) {
	hypo_composite.exit[ich] += dpe_v[ich];
      }
      hypo_composite.tot += dpe_tot;
      if ( intpc )
	hypo_composite.tot_intpc += dpe_tot;
      else
	hypo_composite.tot_outtpc += dpe_tot;
      
      hypo_composite.nexit_used++;
    }

    // std::cout << "[QClusterComposite::generateFlashCompositeHypo][DEBUG] "
    // 	      << " nexit_used=" << nexit_used << " of qexit.size=" << qexit.size()
    // 	      << " padded=" << pe_added
    // 	      << std::endl;
    
    return hypo_composite;
    
  }

  std::vector< TGraph > QClusterComposite::getTGraphsAndHypotheses( const FlashData_t& flash, std::vector<TH1F*>& hypo_v ) const {

    float xoffset = (flash.tpc_tick-3200)*0.5*larutil::LArProperties::GetME()->DriftVelocity();
    //std::cout << "[QClusterComposite::getTGraphsAndHypotheses] xoffset=" << xoffset << std::endl;

    // first build hypotheses (so we can figure out extension lengths)
    FlashCompositeHypo_t hypo_wext  = generateFlashCompositeHypo( flash, true,  false );
    FlashCompositeHypo_t hypo_noext = generateFlashCompositeHypo( flash, false, false );
    //std::cout << "debug hypo-w/ext core="  << hypo_wext.core.tot << std::endl;
    //std::cout << "debug hypo-no/ext core=" << hypo_noext.core.tot << std::endl;    
    
    if ( hypo_v.size()!=2 )
      hypo_v.resize(2,nullptr);
    for ( int i=0; i<2; i++ ) {
      if ( hypo_v[i]==nullptr ) {
	// create one
	if ( i==0 )
	  hypo_v[i] = new TH1F("hhypo_qclustercomp_wext","",32,0,32);
	else if (i==1)
	  hypo_v[i] = new TH1F("hhypo_qclustercomp_noext","",32,0,32);	
      }
      else {
	// reset
	hypo_v[i]->Reset();
      }
    }
    // fill the hypo plots
    for ( int i=0; i<32; i++) {
      hypo_v[0]->SetBinContent(i+1,hypo_wext.PE(i));
      hypo_v[1]->SetBinContent(i+1,hypo_noext.PE(i));
    }
    
    const QCluster_t* qclusters[4] = { &_core._core, &_core._gapfill_qcluster, &_entering_qcluster, &_exiting_qcluster };

    std::vector< TGraph > graph_v;
    for (int i=0; i<4; i++) {
      const QCluster_t& cluster = *(qclusters[i]);
      int nhits = cluster.size();
      if ( i==2 )
	nhits = hypo_wext.nenter_used;
      else if ( i==3 )
	nhits = hypo_wext.nexit_used;
      TGraph comp_xy( nhits );
      TGraph comp_zy( nhits );      
      for ( int i=0; i<nhits; i++ ) {
	const QPoint_t& qpt = cluster[i];
	comp_xy.SetPoint( i, qpt.xyz[0]-xoffset, qpt.xyz[1] );
	comp_zy.SetPoint( i, qpt.xyz[2], qpt.xyz[1] );
      }
      graph_v.emplace_back( std::move(comp_zy) );
      graph_v.emplace_back( std::move(comp_xy) );      
    }

    return graph_v;
  }

  float QClusterComposite::calcMaxPEratio( const FlashData_t& flash, const FlashHypo_t& hypo ) const {
    int idx_maxpmt_data = 0;
    float pe_maxpmt_data = 0;
    for (int i=0; i<32; i++) {
      float pe = flash[i]*flash.tot;
      if ( pe>pe_maxpmt_data ) {
	idx_maxpmt_data = i;
	pe_maxpmt_data = pe;
      }
    }

    //std::cout << "debug: data pe-maxpmt=" << pe_maxpmt_data << std::endl;
    
    float pe_maxpmt_hypo = hypo[idx_maxpmt_data];
    //std::cout << "debug: hypo pe-maxpmt=" << pe_maxpmt_hypo << std::endl;    
    if ( pe_maxpmt_hypo==0 && pe_maxpmt_data==0 )
      return 1;
    else if ( pe_maxpmt_hypo==0 )
      return 0;
    
    float peratio = fabs( pe_maxpmt_hypo-pe_maxpmt_data)/pe_maxpmt_data;
    return peratio;
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
