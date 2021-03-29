#include "NuSel1e1pEventSelection.h"

#include "ublarcvapp/ubdllee/dwall.h"

namespace larflow {
namespace reco {

  int NuSel1e1pEventSelection::runSelection( const larflow::reco::NuSelectionVariables& nusel,
                                             const larflow::reco::NuVertexCandidate& nuvtx )
  {

    // dwall-reco
    int reco_boundary = 0;
    float reco_dwall = ublarcvapp::dwall( nuvtx.pos, reco_boundary );

    // second shower size
    int nhits_second_shower = 0;
    if ( nuvtx.shower_v.size()>1 ) {
      std::vector<int> nhit_shower_v(nuvtx.shower_v.size(),0);
      for (size_t ishr=0; ishr<nuvtx.shower_v.size(); ishr++)
        nhit_shower_v[ishr] = (int)nuvtx.shower_v[ishr].size();
      std::sort( nhit_shower_v.begin(), nhit_shower_v.end() );
      nhits_second_shower = nhit_shower_v[1];
    }      

    // selection cuts
    std::vector<bool> vtx_pass( kNumCuts, false );
    vtx_pass[kMinShowerSize] = nusel.max_shower_nhits>500; // [2]
    vtx_pass[kNShowerProngs] = ( nusel.nshowers>0 && nusel.nshowers<=2 ); // [3]
    vtx_pass[kNTrackProngs]  = ( nusel.ntracks<=2 ); // [4]
    vtx_pass[kShowerGap]     = nusel.nplanes_connected>=2; // [5]
    vtx_pass[kTrackGap]      = (nusel.ntracks==0 || nusel.min_track_gap<3.0); // [6]
    vtx_pass[kMaxTrackLen]   = (nusel.ntracks==0 || nusel.max_track_length<300.0); // [7]
    vtx_pass[kSecondShower]  = (nhits_second_shower<100); // [8]
    vtx_pass[kVertexAct]     = (nusel.max_track_length>3.0 || nusel.vertex_charge_per_pixel>50.0); // [9]
    vtx_pass[kRecoFV]        = (reco_dwall>5.0); // [10]
    vtx_pass[kShowerLLCut]   = (nusel.largest_shower_avedqdx > 20.0 && nusel.largest_shower_avedqdx>20 ); // [11]
    vtx_pass[kWCPixel]       = (nusel.frac_allhits_on_cosmic<0.5); // [12]      
    vtx_pass[kHadronic]      = (nusel.max_proton_pid<40 && nusel.vertex_hip_fraction>0.05); // [13]
    vtx_pass[kAllCuts]       = true;

    // reco variable cuts only
    for ( int i=kMinShowerSize; i<kAllCuts; i++)
      vtx_pass[kAllCuts] = vtx_pass[kAllCuts] && vtx_pass[i];
      
    if ( vtx_pass[kAllCuts] )
      return 1;
    
    return 0;
  }
  
}
}
