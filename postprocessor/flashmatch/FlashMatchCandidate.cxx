#include "FlashMatchCandidate.h"

// eigen
#include <Eigen/Dense>

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
    _evstatus(nullptr)
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
    _corehypo = buildFlashHypothesis( *_flashdata, _offset_qcluster, 0.0  );

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


  FlashHypo_t FlashMatchCandidate::buildFlashHypothesis( const FlashData_t& flashdata, const QCluster_t&  qcluster, float xoffset ) {
    
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
      _entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster, 0.0 );
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
	_entering_hypo = buildFlashHypothesis( *_flashdata, _entering_qcluster, 0.0 );
	_maxdiff = _entering_hypo[_maxch] - _maxch_pe;
      }
    }
    //entering extension finished
  }

  void FlashMatchCandidate::ExtendExitingEnd() {
    // now the exiting end.
    // we only keep extending if it improves the match
    
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
    hhypotot->SetLineColor(kBlack);

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
    hhypogap->SetLineColor(kOrange+3);
    hhypogap->SetLineWidth(1.0);

    std::vector< TH1D* > hclust_v = {hhypocore,hhypoenter,hhypoexit,hhypogap,hhypotot};
    for (int ich=0; ich<32; ich++ ) {
      hhypocore->SetBinContent(ich+1,_corehypo[ich]*_corehypo.tot);
      hhypoenter->SetBinContent(ich+1,_entering_hypo[ich]*_entering_hypo.tot);
      hhypotot->SetBinContent( ich+1, _corehypo[ich]*_corehypo.tot+_entering_hypo[ich]*_entering_hypo.tot );
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

      float hypope = (hhypotot->Integral());
      if ( hypope>10 )
	hypope = 10 + (hypope-10)*0.10;
      radius = ( hypope>50 ) ? 50 : hypope;
      hypomarkers_v[ich] = new TEllipse(xyz[2],xyz[1],radius,radius);
      hypomarkers_v[ich]->SetFillColor(kOrange);

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

      // // mc-track: of truth match if it exists
      // TGraph* mctrack_data    = nullptr;
      // TGraph* mctrack_data_xy = nullptr;      
      // if ( mctrack_match ) {
      // 	const larlite::mctrack& mct = (*_mctrack_v)[ _mctrackid2index[flashdata_v[iflash].mctrackid] ];
      // 	std::cout << "matchedmctrack[" << flashdata_v[iflash].mctrackid << "] npoints: " << mct.size() << std::endl;	
      // 	mctrack_data    = new TGraph( mct.size() );
      // 	mctrack_data_xy = new TGraph( mct.size() );	
      // 	for (int istep=0; istep<(int)mct.size(); istep++) {
      // 	  mctrack_data->SetPoint(istep, mct[istep].Z(), mct[istep].Y() );
      // 	  mctrack_data_xy->SetPoint(istep, mct[istep].X(), mct[istep].Y() );
      // 	}
      // 	mctrack_data->SetLineColor(kBlue);
      // 	mctrack_data->SetLineWidth(1);
      // 	mctrack_data_xy->SetLineColor(kBlue);
      // 	mctrack_data_xy->SetLineWidth(1);
      // }

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
    // if ( mctrack_data )
    //   mctrack_data->Draw("L");

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
    // if ( mctrack_data_xy )
    //   mctrack_data_xy->Draw("L");
      
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
      chmarkers_v[ich]->Draw();
      // if ( truthmatch_hypo )
      // 	truthmarkers_v[ich]->Draw();
    }
      
    TGraph* bestmatchclust_zy[kNumQTypes] = {nullptr};
    TGraph* bestmatchclust_xy[kNumQTypes] = {nullptr};
    TGraph* mctrack_hypo_zy = nullptr;
    TGraph* mctrack_hypo_xy = nullptr;      
	
    // if ( bestmatch_iclust>=0 ) {
    //   // if we have a best match
      
    //   // make graph for hypothesis cluster
    //   int nbestpts[kNumQTypes] = {0};
    //   QCluster_t* qc = &(_qcluster_v[bestmatch_iclust]);
    // 	//std::cout << "qc[" << iclust << "] npoints: " << qc->size() << std::endl;
    //   for (int i=0; i<kNumQTypes; i++) {
    // 	bestmatchclust_zy[i] = new TGraph(qc->size());
    // 	bestmatchclust_xy[i] = new TGraph(qc->size());
	
    // 	bestmatchclust_zy[i]->SetMarkerSize(0.3);
    // 	bestmatchclust_xy[i]->SetMarkerSize(0.3);
	
    // 	bestmatchclust_zy[i]->SetMarkerStyle(20);
    // 	bestmatchclust_xy[i]->SetMarkerStyle(20);
	
    //   }
    //   float xoffset = (flashdata_v[iflash].tpc_tick-3200)*0.5*driftv;	
    //   for (int ipt=0; ipt<(int)qc->size(); ipt++) {
    // 	const QPoint_t& bestq = (*qc)[ipt];
    // 	bestmatchclust_zy[bestq.type]->SetPoint(nbestpts[bestq.type],bestq.xyz[2], bestq.xyz[1] );
    // 	bestmatchclust_xy[bestq.type]->SetPoint(nbestpts[bestq.type],bestq.xyz[0]-xoffset, bestq.xyz[1] );
    // 	nbestpts[bestq.type]++;
    //   }
    //   for (int i=0; i<kNumQTypes; i++) {
    // 	bestmatchclust_zy[i]->Set(nbestpts[i]);
    // 	bestmatchclust_xy[i]->Set(nbestpts[i]);
    //   }
      
    //   bestmatchclust_zy[kGapFill]->SetMarkerColor(kRed);	
    //   bestmatchclust_xy[kGapFill]->SetMarkerColor(kRed);
      
    //   bestmatchclust_zy[kExt]->SetMarkerColor(kGreen+3);	
    //   bestmatchclust_xy[kExt]->SetMarkerColor(kGreen+3);
      
    //   bestmatchclust_zy[kNonCore]->SetMarkerColor(kYellow+2);
    //   bestmatchclust_xy[kNonCore]->SetMarkerColor(kYellow+2);
      
    //   for (int i=0; i<kNumQTypes; i++) {
    // 	bestmatchclust_zy[i]->Draw("P");
    //   }
      
    //   // make graph for mctruth for hypothesis
    //   if ( bestmatch_iclust==truthmatch_idx ) {
    // 	mctrack_hypo_zy = mctrack_data;
    // 	mctrack_hypo_xy = mctrack_data_xy;
    //   }
    // 	else {
    // 	  const larlite::mctrack& mct = (*_mctrack_v)[ _mctrackid2index[qc->mctrackid] ];
    // 	  mctrack_hypo_zy = new TGraph( mct.size() );
    // 	  mctrack_hypo_xy = new TGraph( mct.size() );	
    // 	  for (int istep=0; istep<(int)mct.size(); istep++) {
    // 	    mctrack_hypo_zy->SetPoint(istep, mct[istep].Z(), mct[istep].Y() );
    // 	    mctrack_hypo_xy->SetPoint(istep, mct[istep].X(), mct[istep].Y() );
    // 	  }
    // 	  mctrack_hypo_zy->SetLineColor(kMagenta);
    // 	  mctrack_hypo_zy->SetLineWidth(1);
    // 	  mctrack_hypo_xy->SetLineColor(kMagenta);
    // 	  mctrack_hypo_xy->SetLineWidth(1);
    // 	}
	
    //   }// if has best match

      
      // if ( mctrack_hypo_zy )
      // 	mctrack_hypo_zy->Draw("L");
      
    hypoxy.cd();
    bgxy.Draw();
    boxxy.Draw();
    // if ( bestmatch_iclust>=0 ) {
    //   for (int i=0; i<kNumQTypes; i++) {
    // 	bestmatchclust_xy[i]->Draw("P");
    //   }
    // }
    // if ( mctrack_hypo_xy )
    //   mctrack_hypo_xy->Draw("L");      

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
    sprintf(cname,"hflashmatchcand_flashid%d_clusterid.png",iflash,iclust);
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
    // for (int i=0; i<kNumQTypes; i++) {
    //   if ( truthclust_zy[i] )
    // 	delete truthclust_zy[i];
    //   if ( truthclust_xy[i] )
    // 	delete truthclust_xy[i];
    //   if ( bestmatchclust_zy[i] )
    // 	delete bestmatchclust_zy[i];
    //   if ( bestmatchclust_xy[i] )
    // 	delete bestmatchclust_xy[i];
    // }
    // if (mctrack_data)
    //   delete mctrack_data;
    // if ( mctrack_data_xy )
    //   delete mctrack_data_xy;
    
    // if ( mctrack_hypo_zy && bestmatch_iclust!=truthmatch_idx ) {
    //   delete mctrack_hypo_zy;
    //   delete mctrack_hypo_xy;
    // }
    
    for (int ic=0; ic<(int)hclust_v.size(); ic++) {
      delete hclust_v[ic];
    }
    hclust_v.clear();      
    
    // delete tbestfmatch;
    // tbestfmatch = nullptr;
    
    }//end of dump images
  
  
    
  }

}