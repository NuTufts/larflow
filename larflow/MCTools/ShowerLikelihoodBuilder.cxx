#include "ShowerLikelihoodBuilder.h"

#include "larcv/core/DataFormat/DataFormatTypes.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "DataFormat/larflow3dhit.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/mcshower.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"
#include "larflow/Reco/cluster_functions.h"
#include "larflow/Reco/geofuncs.h"

namespace larflow {
namespace reco {

  /**
   * @brief default constructor 
   */
  ShowerLikelihoodBuilder::ShowerLikelihoodBuilder()
  {
    _hll          = new TH2F("lfshower_ll","",2000, -10, 190, 1000, 0, 100 );
    _hll_weighted = new TH2F("lfshower_ll_weighted","",2000, -10, 190, 1000, 0, 100 );
  }

  ShowerLikelihoodBuilder::~ShowerLikelihoodBuilder()
  {
    // if ( _hll ) delete _hll;
    // _hll = nullptr;
  }

  /**
   * @brief process data for one event, retrieving data from larcv and larlite io managers
   *
   *
   * steps:
   *
   * first we need to assemble true triplet points of showers
   * @verbatim embed:rst:leading-asterisk
   *   * we start by masking out adc images by segment image, keeping shower pixels
   *   * then we pass the masked image into the triplet proposal algorithm, making true pixels
   *   * we filter out the proposals by true match + source pixel being on an electron (true ssnet label)
   * @endverbatim
   *
   * after ground truth points are made, we can build the calculations we want
   *
   * @param[in] iolcv LArCV IO manager
   * @param[in] ioll  larlite IO manager
   */
  void ShowerLikelihoodBuilder::process( larcv::IOManager& iolcv, larlite::storage_manager& ioll )
  {

    // fill profile histogram

    // break into clusters
    // truth ID the trunk cluster
    // save vars for the trunk verus non-trunk clutsters
    // 
    // _fillProfileHist( truehit_v, shower_dir_sce, shower_vtx_sce );


  }

  /**
   * @brief clear variables
   *
   */
  void ShowerLikelihoodBuilder::clear()
  {
  }
  
  /**
   * @brief Build a shower profile histogram using true shower hits.
   *
   * we assume all hits belong to the shower. 
   * note: this code was intended to run on single shower events, 
   * in order to build a proper
   * profile that we can use on multi-shower events.
   *
   * this method populates the member histograms, _hll_weighted and _hll.
   * to do: create profiles for different energies
   * 
   * @param[in] truehit_v Collection of true shower hits
   * @param[in] shower_dir Vector describing initial 3D shower direction
   * @param[in] shower_vtx Vector giving 3D shower start point/vertex
   * 
   */ 
  void ShowerLikelihoodBuilder::_fillProfileHist( const std::vector<larlite::larflow3dhit>& truehit_v,
                                                  std::vector<float>& shower_dir,
                                                  std::vector<float>& shower_vtx )
  {

    // get distance of point from pca-axis
    // http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html

    std::cout << "Fill hits for shower[ "
              << "dir=(" << shower_dir[0] << "," << shower_dir[1] << "," << shower_dir[2] << ") "
              << "vtx=(" << shower_vtx[0] << "," << shower_vtx[1] << "," << shower_vtx[2] << ") "
              << "] with nhits=" << truehit_v.size()
              << std::endl;

    std::vector<float> shower_end(3);
    std::vector<float> d3(3);
    float len3 = 1.0;
    for (int i=0; i<3; i++ ) {
      shower_end[i] = shower_vtx[i] + shower_dir[i];
      d3[i] = shower_dir[i];
    }
      
    for ( auto const& hit : truehit_v ) {
      
      std::vector<float> pt = { hit[0], hit[1], hit[2] };

      std::vector<float> d1(3);
      std::vector<float> d2(3);

      float len1 = 0.;
      for (int i=0; i<3; i++ ) {
        d1[i] = pt[i] - shower_vtx[i];
        d2[i] = pt[i] - shower_end[i];
        len1 += d1[i]*d1[i];
      }
      len1 = sqrt(len1);

      // cross-product
      std::vector<float> d1xd2(3);
      d1xd2[0] =  d1[1]*d2[2] - d1[2]*d2[1];
      d1xd2[1] = -d1[0]*d2[2] + d1[2]*d2[0];
      d1xd2[2] =  d1[0]*d2[1] - d1[1]*d2[0];
      float len1x2 = 0.;
      for ( int i=0; i<3; i++ ) {
        len1x2 += d1xd2[i]*d1xd2[i];
      }
      len1x2 = sqrt(len1x2);
      float rad  = len1x2/len3; // distance of point from PCA-axis

      float proj = 0.;
      for ( int i=0; i<3; i++ )
        proj += shower_dir[i]*d1[i];

      // std::cout << "  hit: (" << pt[0] <<  "," << pt[1] << "," << pt[2] << ") "
      //           << " dist=" << len1
      //           << " proj=" << proj
      //           << " rad=" << rad
      //           << std::endl;
      float w=1.;
      if ( rad>0 ) w = 1.0/rad;
      _hll_weighted->Fill( proj, rad, w );
      _hll->Fill( proj, rad );
    }
    
  }


}
}
