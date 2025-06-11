#ifndef __CHOOSE_MAX_LARFLOW_HIT_H__
#define __CHOOSE_MAX_LARFLOW_HIT_H__


#include <string>
#include <map>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/DataFormat/EventImage2D.h"

#include "larlite/DataFormat/storage_manager.h"
#include "larlite/DataFormat/larflow3dhit.h"

namespace larflow {
namespace reco {
  
  /**
   * @ingroup Reco
   * @class ChooseMaxLArFlowHit
   * @brief Reduces LArMatch hit density by selecting highest-scoring hit per pixel location
   *
   * The LArMatch network can produce multiple 3D spacepoint candidates that project
   * to the same 2D pixel location on a wire plane. This creates redundant hits and
   * can lead to inefficient downstream processing. This class implements a hit
   * reduction algorithm that keeps only the highest-scoring hit per pixel per plane.
   *
   * ## Algorithm Overview
   *
   * The reduction operates in three phases, once for each wire plane:
   * 1. **Pixel Mapping**: Group all 3D hits by their 2D projection coordinates (row, col) on the source plane
   * 2. **Score Selection**: For each pixel location, select the hit with highest LArMatch score
   * 3. **Union Formation**: Combine selected hits from all three planes, avoiding duplicates
   *
   * ## Physics Motivation
   *
   * Multiple hits at the same pixel location typically arise from:
   * - Network uncertainty in 3D reconstruction
   * - Ambiguous wire intersections in complex events
   * - Noise or low-confidence predictions
   *
   * By keeping only the highest-scoring hit, we:
   * - Preserve the most reliable spacepoint reconstructions
   * - Reduce computational load for downstream algorithms
   * - Eliminate redundant information while maintaining reconstruction quality
   *
   * ## Key Features
   *
   * - **Per-plane operation**: Processes each wire plane independently to handle projection ambiguities
   * - **Score-based selection**: Uses LArMatch network confidence (track_score) as selection criterion
   * - **Coordinate mapping**: Projects 3D spacepoints to 2D using stored targetwire coordinates
   * - **Duplicate prevention**: Tracks used hits across planes to avoid double-counting
   * - **Configurable I/O**: Supports custom input/output tree names for flexible integration
   *
   * ## Example Usage
   *
   * Consider spacepoints projecting to Y-plane coordinates (row, col, score):
   * - Input: [(10, 1, 0.8), (10, 1, 0.3), (15, 3, 0.9), (15, 3, 0.6)]
   * - Output: [(10, 1, 0.8), (15, 3, 0.9)]  # Highest score per pixel kept
   *
   * ## Performance Considerations
   *
   * - Uses std::map for O(log n) pixel lookup operations
   * - Memory usage scales with number of unique pixel locations
   * - Processing time is O(n log m) where n = hits, m = unique pixels
   * - Typical reduction factors: 2-5x fewer hits depending on event complexity
   *
   * ## Integration Notes
   *
   * - Operates on larflow3dhit data structures from LArMatch network
   * - Requires wire plane images for coordinate transformation metadata
   * - Best used early in reconstruction pipeline before clustering
   * - Can significantly speed up DBSCAN and other spatial algorithms
   */  
  class ChooseMaxLArFlowHit : public larcv::larcv_base {

  public:

    /**
     * @brief Default constructor with standard parameter initialization
     *
     * Initializes the hit reduction algorithm with commonly used default values:
     * - Input tree: "larmatch" (raw LArMatch network spacepoints)
     * - Output tree: "maxlarmatch" (reduced hit collection)
     *
     * The default settings are suitable for most standard reconstruction workflows
     * where LArMatch output needs to be reduced before clustering.
     */
    ChooseMaxLArFlowHit()
      : larcv::larcv_base("ChooseMaxLArFlowHit"),
      _input_larflow3dhit_treename("larmatch"),
      _output_larflow3dhit_treename("maxlarmatch")
      {};

    /**
     * @brief Virtual destructor
     */
    virtual ~ChooseMaxLArFlowHit() {};

    /**
     * @brief Process hit reduction using I/O managers
     *
     * Main entry point that reads LArMatch spacepoints from the input tree,
     * applies the pixel-based hit reduction algorithm, and saves the reduced
     * hit collection to the output tree.
     *
     * Algorithm execution:
     * 1. Load wire plane images for coordinate transformation metadata
     * 2. Load input LArMatch spacepoints from configured tree
     * 3. For each wire plane, build pixel→hit mapping and select highest scores
     * 4. Combine hits from all planes, avoiding duplicates
     * 5. Save reduced hit collection to output tree
     *
     * @param[in] iolcv LArCV I/O manager (reads wire plane images for coordinate metadata)
     * @param[in] ioll  larlite I/O manager (reads input hits, writes reduced output hits)
     *
     * Performance: Typical reduction factors of 2-5x depending on event complexity.
     * Memory usage scales with number of unique pixel locations across all planes.
     */
    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );

  public:

    /**
     * @struct Pixel_t
     * @brief Represents a unique pixel location in wire plane coordinate space
     *
     * This structure encapsulates the 2D coordinates of a pixel within a specific
     * wire plane, providing a unique identifier for grouping 3D spacepoints that
     * project to the same 2D location.
     *
     * The coordinate system:
     * - **row**: Time dimension (drift direction), typically 0-6000 for MicroBooNE
     * - **col**: Wire dimension, wire number within the plane
     * - **plane**: Wire plane index (0=U, 1=V, 2=Y for MicroBooNE)
     *
     * Used as a key in std::map to associate pixel locations with lists of
     * spacepoint candidates, enabling efficient grouping and selection.
     */
    struct Pixel_t {
      int row;   ///< Row coordinate (time/drift dimension) in 2D wire plane image
      int col;   ///< Column coordinate (wire number) in 2D wire plane image  
      int plane; ///< Wire plane index (0=U, 1=V, 2=Y for MicroBooNE geometry)

      /**
       * @brief Lexicographic comparison operator for std::map ordering
       *
       * Defines total ordering by (plane, row, col) to enable use as map key.
       * Ordering prioritizes plane first, then row, then column for efficient
       * spatial locality in map traversal.
       *
       * @param rhs Right-hand side Pixel_t for comparison
       * @return true if this pixel is lexicographically less than rhs
       */
      bool operator<( const Pixel_t& rhs ) const {
        if ( plane<rhs.plane ) return true;
        else if ( plane==rhs.plane && row<rhs.row ) return true;
        else if ( plane==rhs.plane && row==rhs.row && col<rhs.col ) return true;
        return false;
      };
    };

  protected:
    
    /**
     * @brief Build pixel-to-spacepoint mapping for a specific wire plane
     *
     * This core algorithm creates a mapping from 2D pixel coordinates to the
     * 3D spacepoints that project to those locations. Used internally by
     * process() to group competing spacepoint candidates by pixel location.
     *
     * Algorithm:
     * 1. Iterate through all input spacepoints not yet marked as used
     * 2. Project each 3D point to 2D coordinates on the specified source plane
     * 3. Group spacepoint indices by their (row, col) pixel coordinates
     * 4. Store mapping in _srcpixel_to_spacepoint_m for later selection
     *
     * Coordinate transformation:
     * - Time: hit.tick → row using image metadata
     * - Wire: hit.targetwire[plane] → col directly (already in wire coordinates)
     *
     * @param[in] hit_v Input collection of 3D spacepoints to map
     * @param[in] img_v Wire plane images (used for coordinate transformation metadata)
     * @param[in] source_plane Wire plane index to use as projection source (0, 1, or 2)
     * @param[in] idx_used_v Tracking vector marking which hits have been used (prevents double-counting)
     *
     * @note This method modifies the internal _srcpixel_to_spacepoint_m mapping
     */
    void _make_pixelmap( const larlite::event_larflow3dhit& hit_v,
                         const std::vector<larcv::Image2D>& img_v,
                         const int source_plane,
                         std::vector<int>& idx_used_v );
    
    std::map< Pixel_t, std::vector<int> > _srcpixel_to_spacepoint_m; ///< Maps pixel coordinates to competing spacepoint indices

  protected:

    // Configuration parameters
    std::string _input_larflow3dhit_treename;  ///< Name of input tree containing raw LArMatch spacepoints (default: "larmatch")
    std::string _output_larflow3dhit_treename; ///< Name of output tree for reduced hit collection (default: "maxlarmatch")

  public:

    // Configuration setters

    /** 
     * @brief Set name of input tree containing LArMatch spacepoints
     * @param name Tree name for input hit collection (default: "larmatch")
     *
     * This tree should contain larflow3dhit objects from the LArMatch network,
     * typically representing all possible 3D spacepoint candidates before
     * any hit reduction or quality filtering.
     */
    void set_input_larflow3dhit_treename( std::string name )  { _input_larflow3dhit_treename=name; };

    /** 
     * @brief Set name of output tree for reduced hit collection
     * @param name Tree name for output hit collection (default: "maxlarmatch")
     *
     * The output tree will contain the reduced set of larflow3dhit objects,
     * with only the highest-scoring hit per pixel location retained. This
     * reduced collection is suitable for downstream clustering and tracking.
     */    
    void set_output_larflow3dhit_treename( std::string name ) { _output_larflow3dhit_treename=name; };

  };


}
}

#endif
