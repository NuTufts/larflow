#include "LikelihoodProtonMuon.h"

#include <cstdlib>

#include "TFile.h"
#include "TSpline.h"
#include "TVector3.h"

namespace larflow {
namespace reco {

  LikelihoodProtonMuon::LikelihoodProtonMuon()
    : larcv::larcv_base("LikelihoodProtonMuon")
  {

    _q2adc = 93.0/2.2;

    std::string larflowdir = std::getenv("LARFLOW_BASEDIR");
    std::string splinefile = larflowdir + "/larflow/Reco/data/Proton_Muon_Range_dEdx_LAr_TSplines.root";

    _splinefile_rootfile = new TFile( splinefile.c_str(), "open" );
    _sMuonRange2dEdx = (TSpline3*)_splinefile_rootfile->Get("sMuonRange2dEdx");
    _sProtonRange2dEdx = (TSpline3*)_splinefile_rootfile->Get("sProtonRange2dEdx");  
    
  }

  LikelihoodProtonMuon::~LikelihoodProtonMuon()
  {
    _splinefile_rootfile->Close();
    _sMuonRange2dEdx = nullptr;
    _sProtonRange2dEdx = nullptr;
  }
  
  double LikelihoodProtonMuon::calculateLL( const larlite::track& track ) const
  {

    int npts = track.NumberTrajectoryPoints();
    double current_res_range = 0.;
    const TVector3* last_pt = &(track.LocationAtPoint(npts-1));
    
    // loop in verse order for residual range
    double totw = 0.;
    double totll = 0.;
    
    for (int ipt=npts-1; ipt>=0; ipt-- ) {

      std::vector<double> dqdx_v(4); // one for each plane, plus median value
      for (int p=0; p<4; p++) 
        dqdx_v[p] = track.DQdxAtPoint( ipt, (larlite::geo::View_t)p);
      double dqdx_med = dqdx_v[3];

      const TVector3* pt = &(track.LocationAtPoint(ipt));

      double steplen = (*pt - *last_pt).Mag();

      current_res_range += steplen;


      double res = (current_res_range==0) ? 0.15 : current_res_range;
      
      // calculate residual range
      // calculate likelihood
      double mu_dedx = _sMuonRange2dEdx->Eval(res);
      double mu_dedx_birks = _q2adc*mu_dedx/(1+mu_dedx*0.0486/0.273/1.38);
      double p_dedx = _sProtonRange2dEdx->Eval(res);
      double p_dedx_birks = _q2adc*p_dedx/(1+p_dedx*0.0486/0.273/1.38);
        
      double dmu = dqdx_med-mu_dedx_birks;
      double dp  = dqdx_med-p_dedx_birks;

      double llpt = -0.5*dmu*dmu/100.0 + 0.5*dp*dp/100.0;
      double w_dedx = (mu_dedx_birks-p_dedx_birks)*(mu_dedx_birks-p_dedx_birks);
      if ( dqdx_med>10.0 ) {
        totll += llpt*w_dedx;
        totw  += w_dedx;
      }
    }//end of point loop
    
    if ( totw>0 )
      totll /= totw;

    return totll;
  }
  
}
}
