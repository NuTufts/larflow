#ifndef __SPLIT_HITS_BY_SSNET_H__
#define __SPLIT_HITS_BY_SSNET_H__

#include <Python.h>
#include "bytesobject.h"

#include <string>
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventSparseImage.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {

  /**
   * @ingroup Reco
   * @class SplitHitsBySSNet
   * @brief Classifies LArMatch 3D spacepoints as track or shower using SSNet scores
   *
   * This class takes 3D spacepoints from the LArMatch network and labels them
   * with track/shower classifications using SSNet (Sparse Submanifold Network)
   * scores. The SSNet network provides pixel-level track vs shower predictions
   * on 2D wire plane images, which are used to classify the 3D points.
   *
   * The algorithm:
   * 1. Projects each 3D spacepoint to 2D wire plane coordinates
   * 2. Looks up the SSNet shower score at the projected pixel location
   * 3. Stores the score in the hit's feature vector
   * 4. Optionally splits hits into track/shower categories based on threshold
   *
   * SSNet scores range from 0 (track-like) to 1 (shower-like). The default
   * threshold of 0.5 means hits with score > 0.5 are classified as showers.
   *
   * Usage patterns:
   * - Label only: Add SSNet scores to hits without splitting
   * - Split only: Separate pre-labeled hits into track/shower containers
   * - Label and split: Do both operations in one step
   *
   * The class can operate in standalone mode using the label/split methods,
   * or integrate with the LArCV/larlite I/O framework using process methods.
   */
  class SplitHitsBySSNet : public larcv::larcv_base {

  public:

    /**
     * @brief Default constructor with standard parameters
     * 
     * Initializes with commonly used default values:
     * - SSNet shower threshold: 0.5 (hits > 0.5 classified as shower)
     * - LArMatch quality threshold: 0.1 (minimum hit score to consider)
     * - SSNet image stem: "ubspurn_plane" (expects ubspurn_plane0, plane1, plane2)
     * - Wire image name: "wire" (ADC images for projection)
     * - Input hit tree: "larmatch" (LArMatch spacepoint container)
     * - Output stem: "ssnsetsplit" (creates ssnsetsplit_showerhit, ssnsetsplit_trackhit)
     */
    SplitHitsBySSNet()
      : larcv::larcv_base("SplitHitsBySSNet"),
      _score_threshold(0.5),
      _larmatch_threshold(0.1),
      _ssnet_stem_name("ubspurn_plane"),
      _adc_name("wire"),
      _input_larmatch_hit_tree_name("larmatch"),
      _output_larmatch_hit_stem_name("ssnsetsplit")
      {};

    /**
     * @brief Virtual destructor
     */
    virtual ~SplitHitsBySSNet() {};

    /**
     * @brief Label hits with SSNet scores and split into track/shower categories
     * 
     * Combined operation that both adds SSNet scores to hits and separates them
     * into track and shower containers based on the threshold.
     * 
     * @param ssnet_score_v Vector of SSNet score images (one per wire plane)
     * @param lfhit_v Input collection of 3D spacepoints to classify
     * @param ssnet_score_threshold Shower score threshold for classification (typically 0.5)
     * @param larmatch_score_threshold Minimum LArMatch quality to consider hit
     * @param accept_v Output container for hits classified as showers (score > threshold)
     * @param reject_v Output container for hits classified as tracks (score <= threshold)
     * 
     * This is the most commonly used method, combining labeling and splitting
     * in one efficient operation. Only hits with LArMatch score > larmatch_score_threshold
     * are processed.
     */
    void label_and_split( const std::vector<larcv::Image2D>& ssnet_score_v,
                          const larlite::event_larflow3dhit& lfhit_v,
                          const float ssnet_score_threshold,
                          const float larmatch_score_threshold,
                          std::vector<larlite::larflow3dhit>& accept_v,
                          std::vector<larlite::larflow3dhit>& reject_v );

    /**
     * @brief Add SSNet shower scores to hits without splitting
     * 
     * Labels each hit with its SSNet shower score by projecting to wire planes
     * and looking up the score at the projected pixel location. The score is
     * stored in the hit's renormed_shower_score field.
     * 
     * @param ssnet_score_v Vector of SSNet score images (one per wire plane)
     * @param larmatch_hit_v Input/output collection of hits to label (modified in place)
     * 
     * Use this when you want to add SSNet information but defer the track/shower
     * classification decision. The hits can be split later using split() or split_constinput().
     */
    void label( const std::vector<larcv::Image2D>& ssnet_score_v,
                larlite::event_larflow3dhit& larmatch_hit_v );

    /**
     * @brief Split pre-labeled hits into track and shower categories
     * 
     * Separates hits that already have SSNet scores into track and shower containers.
     * Assumes hits have been previously labeled with SSNet scores.
     * 
     * @param lfhit_v Input collection of pre-labeled hits (modified: hits may be removed)
     * @param ssnet_score_threshold Shower score threshold for classification
     * @param larmatch_score_threshold Minimum LArMatch quality to consider hit
     * @param accept_v Output container for shower hits (score > threshold)
     * @param reject_v Output container for track hits (score <= threshold)
     * 
     * Only hits with LArMatch score > larmatch_score_threshold are processed.
     * Processed hits are removed from the input container.
     */
    void split( larlite::event_larflow3dhit& lfhit_v,
                const float ssnet_score_threshold,
                const float larmatch_score_threshold,
                std::vector<larlite::larflow3dhit>& accept_v,
                std::vector<larlite::larflow3dhit>& reject_v );

    /**
     * @brief Split pre-labeled hits without modifying input container
     * 
     * Same as split() but does not modify the input container - hits are copied
     * to output containers instead of moved.
     * 
     * @param lfhit_v Input collection of pre-labeled hits (not modified)
     * @param ssnet_score_threshold Shower score threshold for classification
     * @param larmatch_score_threshold Minimum LArMatch quality to consider hit
     * @param accept_v Output container for shower hits (score > threshold)
     * @param reject_v Output container for track hits (score <= threshold)
     */
    void split_constinput( const larlite::event_larflow3dhit& lfhit_v,
                           const float ssnet_score_threshold,
                           const float larmatch_score_threshold,
                           std::vector<larlite::larflow3dhit>& accept_v,
                           std::vector<larlite::larflow3dhit>& reject_v );
    
    /**
     * @brief Process hits using I/O managers (labels and splits)
     * 
     * High-level interface that reads SSNet images and LArMatch hits from
     * I/O managers, performs labeling and splitting, then saves results
     * to output trees.
     * 
     * @param iolcv LArCV I/O manager (reads SSNet score images)
     * @param ioll larlite I/O manager (reads hits, writes track/shower hits)
     * 
     * This method:
     * 1. Loads SSNet score images using _ssnet_stem_name + plane index
     * 2. Loads LArMatch hits from _input_larmatch_hit_tree_name
     * 3. Labels and splits hits using class threshold parameters
     * 4. Saves shower hits to "_output_stem_showerhit" tree
     * 5. Saves track hits to "_output_stem_trackhit" tree
     */
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    
    /**
     * @brief Process hits to add SSNet labels only (no splitting)
     * 
     * Like process() but only adds SSNet scores to hits without creating
     * separate track/shower containers. Modified hits are saved back to
     * the original container.
     * 
     * @param iolcv LArCV I/O manager (reads SSNet score images)
     * @param ioll larlite I/O manager (reads and modifies hits)
     */
    void process_labelonly( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

    /**
     * @brief Process pre-labeled hits to split them only
     * 
     * Like process() but assumes hits already have SSNet scores and only
     * performs the splitting operation.
     * 
     * @param iolcv LArCV I/O manager (not used, kept for interface consistency)
     * @param ioll larlite I/O manager (reads labeled hits, writes track/shower hits)
     */
    void process_splitonly( larcv::IOManager& iolcv, larlite::storage_manager& ioll );      

    /**
     * @brief Python interface for track/shower labeling (experimental)
     * 
     * Provides Python interface for SSNet-based track/shower classification
     * using sparse representation. This is an experimental feature for
     * integration with Python-based analysis workflows.
     * 
     * @param adc_v Wire plane ADC images for projection
     * @param ssnet_score_v SSNet score images
     * @param adc_threshold Minimum ADC threshold for valid pixels
     * @param spacepoint_triplets Python array of 3D spacepoint coordinates
     * @param sparse_wireplane0 Sparse representation of plane 0
     * @param sparse_wireplane1 Sparse representation of plane 1  
     * @param sparse_wireplane2 Sparse representation of plane 2
     * @return PyObject* Python array with track/shower labels
     * 
     * @note This method requires proper Python/NumPy initialization
     */
    PyObject* make_trackshowerlabels_from2dssnet( 
              const std::vector<larcv::Image2D>& adc_v,
              const std::vector<larcv::Image2D>& ssnet_score_v,
              const float adc_threshold,
              PyObject* spacepoint_triplets, 
              PyObject* sparse_wireplane0,
              PyObject* sparse_wireplane1,
              PyObject* sparse_wireplane2 );
    
  protected:
    
    // Configuration parameters
    float _score_threshold;                      ///< SSNet shower score threshold above which hits are classified as shower (default: 0.5)
    float _larmatch_threshold;                   ///< Minimum LArMatch score for hits to be processed (default: 0.1)
    std::string _ssnet_stem_name;                ///< Stem name for SSNet score images (default: "ubspurn_plane", expects plane0/1/2 suffix)
    std::string _adc_name;                       ///< Name of ADC wire plane image tree (default: "wire")
    std::string _input_larmatch_hit_tree_name;   ///< Name of input LArMatch spacepoint tree (default: "larmatch")
    std::string _output_larmatch_hit_stem_name;  ///< Stem for output hit tree names (default: "ssnsetsplit", creates _showerhit/_trackhit)
    
  public:

    // Configuration setters
    
    /**
     * @brief Set SSNet shower score threshold for classification
     * @param thresh Threshold value (0.0-1.0). Hits with score > thresh become showers
     * 
     * Lower thresholds classify more hits as showers, higher thresholds are more selective.
     * Typical values: 0.3-0.7, with 0.5 being standard.
     */
    void set_ssnet_threshold( float thresh )    { _score_threshold=thresh; };

    /**
     * @brief Set minimum LArMatch quality threshold for processing hits
     * @param thresh Minimum LArMatch score (typically 0.1-0.5)
     * 
     * Only hits with LArMatch score above this threshold will be processed.
     * This filters out low-quality spacepoint predictions from the network.
     */
    void set_larmatch_threshold( float thresh ) { _larmatch_threshold=thresh; };

    /**
     * @brief Set stem name for SSNet score image trees
     * @param stem Base name for SSNet images (e.g., "ubspurn_plane" → "ubspurn_plane0", "ubspurn_plane1", "ubspurn_plane2")
     */
    void set_ssnet_tree_stem_name( std::string stem ) { _ssnet_stem_name=stem; };

    /**
     * @brief Set name of tree containing LArMatch spacepoints
     * @param hitname Name of larlite tree with larflow3dhit objects
     */
    void set_larmatch_tree_name( std::string hitname ) { _input_larmatch_hit_tree_name=hitname; };

    /**
     * @brief Set name of tree containing wire plane ADC images
     * @param name Name of LArCV tree with Image2D wire plane data
     */
    void set_adc_tree_name( std::string name ) { _adc_name=name; };

    /**
     * @brief Set stem name for output hit trees
     * @param stem Base name for output (e.g., "split" → "split_showerhit", "split_trackhit")
     */
    void set_output_tree_stem_name( std::string stem ) { _output_larmatch_hit_stem_name=stem; };

  protected:

    // Internal storage for split results
    std::vector<larlite::larflow3dhit>  _shower_hit_v; ///< Internal container for shower-classified hits (used by getter methods)
    std::vector<larlite::larflow3dhit>  _track_hit_v;  ///< Internal container for track-classified hits (used by getter methods)

  public:

    // Access to internal storage
    
    /**
     * @brief Get mutable reference to shower hit container
     * @return Reference to internal shower hit vector
     * 
     * Provides access to shower hits from last split operation.
     * Container is cleared at start of each split.
     */
    std::vector<larlite::larflow3dhit>& get_shower_hits() { return _shower_hit_v; };

    /**
     * @brief Get mutable reference to track hit container  
     * @return Reference to internal track hit vector
     * 
     * Provides access to track hits from last split operation.
     * Container is cleared at start of each split.
     */   
    std::vector<larlite::larflow3dhit>& get_track_hits()  { return _track_hit_v; };    

    /**
     * @brief Get const reference to shower hit container
     * @return Const reference to internal shower hit vector
     */    
    const std::vector<larlite::larflow3dhit>& get_shower_hits() const { return _shower_hit_v; };

    /**
     * @brief Get const reference to track hit container
     * @return Const reference to internal track hit vector
     */        
    const std::vector<larlite::larflow3dhit>& get_track_hits()  const { return _track_hit_v; };

  protected:

    /**
     * @brief Create track/shower images from sparse UResNet output
     * 
     * Helper function to convert sparse network outputs to dense 2D images
     * for lookup operations. Used when SSNet scores are stored in sparse format.
     * 
     * @param plane Wire plane index (0, 1, or 2)
     * @param adc Reference ADC image for geometry information
     * @param iolcv LArCV I/O manager to read sparse data
     * @param container Output container for dense images
     */
    void _make_trackshower_images_from_sparse_uresnet( const int plane,
						       const larcv::Image2D& adc,
						       larcv::IOManager& iolcv,
						       larcv::EventImage2D& container );
    

  private:

    static bool __setup_numpy;  ///< Flag tracking whether NumPy has been initialized for Python interface
    
  };
  
}
}

#endif
