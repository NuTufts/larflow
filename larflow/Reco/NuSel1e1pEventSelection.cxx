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

    std::vector<std::string> cutnames
      = {"minshower",  // 0
         "nshowerprongs", //1
         "ntrackprongs", //2
         "showergap", //3
         "trackgap", //4
         "maxtracklen", //5
         "secondshowernhits", //6
         "vertexact", //7
         "recofv", //8
         "showerll", //9
         "wcpixel", //10
         "hadronic", //11
         "allcuts"}; //12
         
    
    // selection cuts
    std::vector<bool> vtx_pass( kNumCuts, false );
    vtx_pass[kMinShowerSize] = nusel.max_shower_nhits>500; // [0]
    vtx_pass[kNShowerProngs] = ( nusel.nshowers>0 && nusel.nshowers<=2 ); // [1]
    vtx_pass[kNTrackProngs]  = ( nusel.ntracks<=2 ); // [2]
    vtx_pass[kShowerGap]     = nusel.nplanes_connected>=2; // [3]
    vtx_pass[kTrackGap]      = (nusel.ntracks==0 || nusel.min_track_gap<3.0); // [4]
    vtx_pass[kMaxTrackLen]   = (nusel.ntracks==0 || nusel.max_track_length<300.0); // [5]
    vtx_pass[kSecondShower]  = (nhits_second_shower<100); // [6]
    vtx_pass[kVertexAct]     = (nusel.max_track_length>3.0 || nusel.vertex_charge_per_pixel>50.0); // [7]
    vtx_pass[kRecoFV]        = (reco_dwall>5.0); // [8]
    vtx_pass[kShowerLLCut]   = (nusel.largest_shower_avedqdx > 20.0 && nusel.largest_shower_avedqdx>20 ); // [9]
    vtx_pass[kWCPixel]       = (nusel.frac_allhits_on_cosmic<0.5); // [10]      
    vtx_pass[kHadronic]      = (nusel.max_proton_pid<100 && nusel.vertex_hip_fraction>0.05) || (nusel.vertex_charge_per_pixel>50.0); // [11]
    vtx_pass[kAllCuts]       = true;

    // reco variable cuts only
    for ( int i=kMinShowerSize; i<kAllCuts; i++) {
      LARCV_DEBUG() << "  cut[" << i << "] " << cutnames[i] << ": pass=" << vtx_pass[i] << " all=" << vtx_pass[kAllCuts] << std::endl;
      vtx_pass[kAllCuts] = vtx_pass[kAllCuts] && vtx_pass[i];
    }
    LARCV_DEBUG() << "vertex dist2true: " << nusel.dist2truevtx << " cm" << std::endl;
      
    if ( vtx_pass[kAllCuts] )
      return 1;
    
    return 0;
  }
  
}
}
