#ifndef __LARFLOW_PREP_MCKEYPOINT_MAKER_H__
#define __LARFLOW_PREP_MCKEYPOINT_MAKER_H__

#include <vector>
#include <string>

#include "larcv/core/Base/larcv_base.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"

#include "MCKeypoint.h"

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
  class MCParticleGraph;
  class EventMCPixelLabels;
}
}

namespace HighFive {
  class File;
}

namespace larflow {
namespace prep {
  

  /**
   * @ingroup PrepFlowMatchData
   * @class MCKeypointMaker
   * @brief Make the training data for the key-point+larmatch task
   *
   *
   * The cases we are trying to label:
   * \verbatim embed:rst:leading-asterisks
   *  * track start: both cosmic and nu origin
   *  * track end: both cosmic and nu origin

   * \endverbatim
   *
   * maybe, depending on quality of truth data
   * neutrino secondaries -- scattering secondaries.
   *
   * The outputs we are trying to get the network to make are:
   * \verbatim embed:rst:leading-asterisks
   *  1. for each space point (i.e. triplet) proposed, scores (one for each keypoint type)
   *     which come from a gaussian between the 3D position of the space point
   *     and the closest true keypoint of that class. 
   *  2. If the closest true keypoint is greater than some distance, the score is set to 0.0
   *  3. The classes are { neutrino vertex, shower start, track ends }
   * \endverbatim
   * 
   *
   */  
  class MCKeypointMaker : public larcv::larcv_base {
    
  public:

    MCKeypointMaker();
    virtual ~MCKeypointMaker();

    void process( larcv::IOManager& iolcv,
                  larlite::storage_manager& ioll );
    
    void process( const std::vector<larcv::Image2D>&    adc_v,
                  const std::vector<larcv::Image2D>&    badch_v,
                  //const std::vector<larcv::Image2D>&    segment_v,
                  const larlite::event_mctrack&  mctrack_v,
                  const larlite::event_mcshower& mcshower_v,
                  const larlite::event_mctruth&  mctruth_v );

    void set_mcparticle_graph( ublarcvapp::mctools::MCParticleGraph* mcpg ) { _mcpg = mcpg; };
    void set_spacecharge_instance( larutil::SpaceChargeMicroBooNE* psce )   { _psce = psce; };

    //void make_proposal_labels( const larflow::prep::PrepMatchTriplets& match_proposals );
    void clear();

  protected:

    std::string _adc_image_treename; ///< name of tree storing charge image2d
    

    std::vector<MCKeypoint> _kpd_v; ///< info on true keypoints found using MC truth
    
    std::vector<MCKeypoint>    
      getMuonEndpoints( ublarcvapp::mctools::MCParticleGraph& mcpg,
                        const std::vector<larcv::Image2D>& adc_v,
                        const larlite::event_mctrack& mctrack_v,
                        larutil::SpaceChargeMicroBooNE* psce );
    
    std::vector<MCKeypoint>
      getShowerStarts( ublarcvapp::mctools::MCParticleGraph& mcpg,
                       const std::vector<larcv::Image2D>& adc_v,
                       const larlite::event_mcshower& mcshower_v,
                       larutil::SpaceChargeMicroBooNE* psce );

    std::vector<MCKeypoint>
      getNonMuonTrackStarts( ublarcvapp::mctools::MCParticleGraph& mcpg,
                              const std::vector<larcv::Image2D>& adc_v,
                              const larlite::event_mctrack& mctrack_v,
                              larutil::SpaceChargeMicroBooNE* psce );

    std::vector<MCKeypoint>
      label_nu_keypoints( const larlite::event_mctruth& mctruth_v,
                          const std::vector<larcv::Image2D>& img_v,
                          larutil::SpaceChargeMicroBooNE* psce );

    // void _label_nu_keypoints( const larlite::event_mctruth& mctruth_v,
    //                           const std::vector<larcv::Image2D>& adc_v,
    //                           larutil::SpaceChargeMicroBooNE* psce,
    //                           std::vector<MCKeypoint>& kpdata_v  );
    // void _move_floating_keypoints(  const larflow::prep::PrepMatchTriplets& match_proposals );

    void _adjust_photon_keypoints( 
      float edep_cluster_threshold,
      float edep_point_threshold,
      ublarcvapp::mctools::MCParticleGraph& mcpg,
      ublarcvapp::mctools::EventMCPixelLabels& pixel3d);

    void _copy_to_vectors();
    
    //void filter_duplicates();

    ublarcvapp::mctools::MCParticleGraph* _mcpg;
    larutil::SpaceChargeMicroBooNE* _psce;
    larlite::storage_manager* _ioll;
    larcv::IOManager* _iolcv;

  public:

    /**
     * @brief set the tree name used to get wire plane images
     * @param[in] treename Name of tree in ROOT file following `image2d_[treename]_tree`
     */
    void setADCimageTreeName(std::string treename) { _adc_image_treename=treename; };

    /**
     * @brief get the vector of true keypoints labeled in the image
     * @return vector of keypoints represented by MCKeypoint class
     */    
    const std::vector<MCKeypoint>& getMCKeypoint() const { return _kpd_v; };

    void printKeypoints() const;

    void export_as_hdf( std::string outfile );
    void save_entry_to_hdf( HighFive::File& file, std::string group_prefix_name );
            
  public:

    // PyObject* get_keypoint_array(int ikpclass ) const;
    // PyObject* get_triplet_score_array( float sig ) const;

    
  public:

    // PROPOSAL LABELS
    // ----------------
    
    std::vector< std::vector<float> > _match_proposal_labels_v[6]; ///< provides the labels for triplet proposals made by larflow::prep::PrepMatchTriplets

    // Ana Tree
    int _run; ///< run ID number of event
    int _subrun; ///< subrun ID number of event
    int _event;  ///< event ID number
    // //TTree* _label_tree; ///< ROOT TTree for storing labels
    std::vector< std::vector<float> > _kppos_v[6]; ///< container of true keypoint 3D positions in cm, for each of the 6 classes
    std::vector< std::vector<int> >   _kp_pdg_trackid_v[6]; ///< each entry maps (pdg, trackid) for truth meta-data matching
    // //< we need to keep a list of primary pixels to limit the neutrino score field    
    // //std::vector< std::set<std::pair<int,int> > >  _primarypixels_v;
    
  public:
    
    // void defineAnaTree();
    // void writeAnaTree();

    /** @brief if analysis tree has been created, fill data to tree for current event */
    //void fillAnaTree() { if (_label_tree) _label_tree->Fill(); };

  public:

    // // statistics variables
    // TH1F* hdist[3]; ///< histograms contaning distance to true keypoint for the spacepoints
    // TH1F* hdpix[4]; ///< Don't remember
    int _nclose;    ///< number of space point proposals within some radius of a true keypoint
    int _nfar;      ///< number of space point proposals further than some radius of a true keypoint
    // void writeHists();
    
  protected:

    double tpc_bounds[3][2];

  private:
    
    //static bool _setup_numpy; ///< flag to indicate if import_numpy() has been called
    
  };

}
}

#endif
