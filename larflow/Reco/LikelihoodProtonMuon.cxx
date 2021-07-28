#include "LikelihoodProtonMuon.h"

#include <cstdlib>
#include <sstream>

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

  double LikelihoodProtonMuon::calculateLL( const larlite::track& track,
                                            bool reverse ) const
  {

    std::vector<float> startvtx(3,0);

    int npts = track.NumberTrajectoryPoints();
    int firstpt = 0;
    if ( reverse ) {
      firstpt = npts-1;
    }

    auto const& pt = track.LocationAtPoint(firstpt);
    for (int i=0; i<3; i++)
      startvtx[i] = pt[i];

    return calculateLL( track, startvtx );
    
  }
  
  std::vector<double> LikelihoodProtonMuon::calculateLLseparate( const larlite::track& track,
                                                                 const std::vector<float>& vertex ) const
  {

    int npts = track.NumberTrajectoryPoints();
    double current_res_range = 0.;

    // check if dq/dx calculated for track
    size_t ndqdx_pts = track.NumberdQdx((larlite::geo::View_t)0);
    if ( ndqdx_pts!=4 ) {
      std::stringstream ss;
      ss << "[larflow::reco::LikelihoodProtonMuon.L" << __LINE__ << "] "
         << "Track given does not have complete dqdx information" << std::endl;
      throw std::runtime_error(ss.str());
    }

    const TVector3* start_pt = &(track.LocationAtPoint(0));
    const TVector3* end_pt = &(track.LocationAtPoint(npts-1));
    float dist[2] = {0,0};
    for (int i=0; i<3; i++) {
      dist[0] += ( vertex[i]-(*start_pt)[i] )*( vertex[i]-(*start_pt)[i] );
      dist[1] += ( vertex[i]-(*end_pt)[i] )*( vertex[i]-(*end_pt)[i] );
    }

    int istart = (dist[0]<dist[1]) ? npts-1 : 0;
    int iend   = (dist[0]<dist[1]) ? 0 : npts-1;
    int dindex = (dist[0]<dist[1]) ? -1 : 1;
    
    // loop in verse order for residual range
    double totw = 0.;
    double totll = 0.;
    double tot_llproton = 0.;
    double tot_llmuon = 0;
    
    const TVector3* last_pt = &(track.LocationAtPoint(istart));

    int ipt = istart;
    while ( ipt!=iend ) {

      std::vector<double> dqdx_v(4); // one for each plane, plus median value
      for (int p=0; p<4; p++) 
        dqdx_v[p] = track.DQdxAtPoint( ipt, (larlite::geo::View_t)p);
      double dqdx_med = dqdx_v[3];

      const TVector3* pt = &(track.LocationAtPoint(ipt));

      double steplen = (*pt - *last_pt).Mag();

      current_res_range += steplen;
      ipt += dindex;

      last_pt = pt;

      if ( dqdx_med>10.0 ) {
        
        double res = (current_res_range<0.15) ? 0.15 : current_res_range;
      
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
        
        totll += llpt*w_dedx;
        tot_llproton += w_dedx*(0.5*dp*dp/100.0);
        tot_llmuon   += w_dedx*(0.5*dmu*dmu/100.0);
        totw  += w_dedx;
      }
    }//end of point loop
    
    if ( totw>0 ) {
      totll /= totw;
      tot_llproton /= totw;
      tot_llmuon   /= totw;
    }

    std::vector<double> result = { totll, tot_llproton, tot_llmuon, totw };
    return result;
  }

  double LikelihoodProtonMuon::calculateLL( const larlite::track& track,
                                                         const std::vector<float>& vertex ) const
  {
    return calculateLLseparate(track,vertex).at(0);
  }
  
  
}
}
