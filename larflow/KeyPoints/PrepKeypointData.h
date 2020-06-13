#ifndef __PREP_KEYPOINT_DATA_H__
#define __PREP_KEYPOINT_DATA_H__

/**
 * 
 * This class is responsible for making the training data for the key-point+larmatch task
 *
 * The job is to provide labels for the proposed matches for the network.
 * It can either remake proposals or it can make labels for old proposals.
 *
 * The cases we are trying to label:
 *
 * cosmic muon start and stop, primary ancestor only.
 * cosmic proton start and stop, primary ancestor only.
 * neutrino primary track start and stop
 * neutrino primary shower start and stop
 * cosmic shower start, primary ancestor only (hard)
 *
 * maybe, depending on quality of truth data
 * neutrino secondaries -- scattering secondaries
 *
 * The training output we are trying to get the network to make are:
 *  1) for each space point (i.e. triplet) proposed, scores (one for each keypoint type)
 *     which come from a gaussian between the 3D position of the space point
 *     and the closest true keypoint of that class. 
 *  2) If the closest true keypoint is greater than some distance, the score is set to 0.0
 *  3) The classes are { neutrino vertex, shower start, track ends }
 * 
 * 
 *
 */

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include <string>

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larflow/PrepFlowMatchData/PrepMatchTriplets.h"

#include "KPdata.h"

class TH1F;
class TH2D;

namespace larcv {
  class Image2D;
  class IOManager;
}

namespace larlite {
  class event_mctrack;
  class event_mcshower;
  class event_mctruth;
  class storage_manager;
}

namespace larutil {
  class SpaceChargeMicroBooNE;
}

namespace ublarcvapp {
namespace mctools {
  class MCPixelPGraph;
}
}

namespace larflow {
namespace keypoints {
  
  
  class PrepKeypointData {
  public:

    PrepKeypointData();
    virtual ~PrepKeypointData();

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );
    
    void process( const std::vector<larcv::Image2D>&    adc_v,
                  const std::vector<larcv::Image2D>&    badch_v,
                  const std::vector<larcv::Image2D>&    segment_v,
                  const std::vector<larcv::Image2D>&    instance_v,
                  const std::vector<larcv::Image2D>&    ancestor_v,
                  const larlite::event_mctrack&  mctrack_v,
                  const larlite::event_mcshower& mcshower_v,
                  const larlite::event_mctruth&  mctruth_v );

    void make_proposal_labels( const larflow::PrepMatchTriplets& match_proposals );
    
  protected:

    std::string _adc_image_treename; //< name of tree storing charge image2d
    
    // KPdata in KPdata.h
    // This class represents the true keypoints
    std::vector<KPdata> _kpd_v; 
    
    std::vector<KPdata>    
      getMuonEndpoints( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                        const std::vector<larcv::Image2D>& adc_v,
                        const larlite::event_mctrack& mctrack_v,
                        larutil::SpaceChargeMicroBooNE* psce );
    
    std::vector<KPdata>
      getShowerStarts( ublarcvapp::mctools::MCPixelPGraph& mcpg,
                       const std::vector<larcv::Image2D>& adc_v,
                       const larlite::event_mcshower& mcshower_v,
                       larutil::SpaceChargeMicroBooNE* psce );

    void _label_nu_keypoints( const larlite::event_mctruth& mctruth_v,
                              const std::vector<larcv::Image2D>& adc_v,
                              larutil::SpaceChargeMicroBooNE* psce,
                              std::vector<KPdata>& kpdata_v  );
    
    void filter_duplicates();

  public:

    void setADCimageTreeName(std::string treename) { _adc_image_treename=treename; };
    const std::vector<KPdata>& getKPdata() const { return _kpd_v; };
    void printKeypoints() const;
            
  public:
    
    PyObject* get_keypoint_array(int ikpclass ) const; //< get numpy array with true keypoint information
    PyObject* get_triplet_score_array( float sig ) const; //< get numpy array with target scores

    
  protected:

    // PROPOSAL LABELS
    // ----------------
    
    // this provides the labels for each triplet proposal made by
    // larflow::PrepMatchTriplets
    std::vector< std::vector<float> > _match_proposal_labels_v[3]; 

    // Ana Tree
    int _run;
    int _subrun;
    int _event;
    TTree* _label_tree; ///< tree for storing labels
    std::vector< std::vector<float> > _kppos_v[3]; //< container for each of the 3 classes
    //< we need to keep a list of primary pixels to limit the neutrino score field    
    //std::vector< std::set<std::pair<int,int> > >  _primarypixels_v; 
    
  public:
    
    void defineAnaTree();
    void writeAnaTree();
    void fillAnaTree() { if (_label_tree) _label_tree->Fill(); };

  public:

    // statistics variables
    TH1F* hdist[3];
    TH1F* hdpix[4];
    int _nclose;
    int _nfar;
    void writeHists();

    std::vector<TH2D> makeScoreImage( const int ikpclass, const float sigma,
                                      const std::string histname,
                                      const larflow::PrepMatchTriplets& tripmaker,
                                      const std::vector<larcv::Image2D>& adc_v ) const;
    
  private:
    
    static bool _setup_numpy;
    
  };

}
}

#endif
