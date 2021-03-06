#ifndef __FLOW_MATCH_HIT_MAKER_H__
#define __FLOW_MATCH_HIT_MAKER_H__


#include <Python.h>
#include "bytesobject.h"
#include <vector>
#include <map>
#include <array>
#include "larcv/core/DataFormat/EventChStatus.h"
#include "core/DataFormat/larflow3dhit.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larcv/core/DataFormat/Image2D.h"


namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData 
   * @class FlowMatchHitMaker
   * @brief Processes output of LArMatch network and makes space point objects
   *
   * @author Taritree Wongjirad (taritree.wongjirad@tuts.edu)
   * @date $Data 2020/07/22 17:00$
   *
   *
   * Revision history
   * 2020/07/22: Added doxygen documentation. 
   * 
   *
   */  
  class FlowMatchHitMaker {
  public:
    
    FlowMatchHitMaker()
      : _match_score_threshold(0.5),
      has_ssnet_scores(false),
      has_kplabel_scores(false),
      has_paf(false)
        {};
    virtual ~FlowMatchHitMaker() {};

    /**
     * @struct match_t
     * @brief  internal struct storing network output for single spacepoint
     *
     * indexed by (U,V,Y) wire ids
     *
     */
    struct match_t {
      
      int Y; ///< Y-wire
      int U; ///< U-wire
      int V; ///< V-wire

      float YU; ///< match score between Y to U
      float YV; ///< match score between Y to V
      float UV; ///< match score between U to V
      float UY; ///< match score between U to Y
      float VU; ///< match score between V to U
      float VY; ///< match score between V to Y

      std::array<float,3> tyz; ///< tick, y, z coordinates
      std::vector<float> ssnet_scores; ///< ssnet scores
      std::vector<float> keypoint_scores; ///< keypoint scores (for each class)
      std::vector<float> paf; ///< particle affinity field, 3D vector
      int istruth; ///< indicates if a true (i.e. not ghost) space point. 1->true; 0->ghost
      
      match_t()
      : Y(0),U(0),V(0),
        YU(0),YV(0),UV(0),UY(0),VU(0),VY(0),
        keypoint_scores( std::vector<float>(3,0) ),
        paf( std::vector<float>(3,0) ),
        istruth(0)
      {};

      /** comparitor for sorting. based on plane wire ids going from plane Y,U,V. */
      bool operator<(const match_t& rhs) const {
        if (Y<rhs.Y) return true;
        else if ( Y==rhs.Y && U<rhs.U ) return true;
        else if ( Y==rhs.Y && U==rhs.U && V<rhs.V ) return true;
        return false;
      };

      /** set wire index number for a plane */
      void set_wire( int plane, int wire ) {
        if ( plane==0 )      U = wire;
        else if ( plane==1 ) V = wire;
        else if ( plane==2 ) Y = wire;
      };

      /** set flag indicating if true space point. 1->true; 0->ghost. */
      void set_istruth( int label) {
	istruth = label;
      };

      /** set larmatch score for provided flow between planes */
      void set_score ( int source_plane, int target_plane, float prob ) {
        if ( source_plane==2 ) {
          if ( target_plane==0 && prob>YU )      YU = prob;
          else if ( target_plane==1 && prob>YV ) YV = prob;
        }
        else if ( source_plane==1 ) {
          if ( target_plane==0 && prob>VU )      VU = prob;
          else if ( target_plane==2 && prob>VY ) VY = prob;
        }
        else if ( source_plane==0 ) {
          if ( target_plane==1 && prob>UV )      UV = prob;
          else if ( target_plane==2 && prob>UY ) UY = prob;
        };
      };

      /** get the scores for all of the flow directions */
      std::vector<float> get_scores() const {
        std::vector<float> s(6,0);
        s[0] = YU;
        s[1] = YV;
        s[2] = UV;
        s[3] = UY;
        s[4] = VU;
        s[5] = VY;
        return s;
      };
      
    };


    float _match_score_threshold;    ///< do not generate hits for space points below this threshold
    std::vector<match_t> _matches_v; ///< container for network output for each spacepoint
    std::map< std::vector<int>, int > _match_map; ///< map of (Y,U,V) triple to positin in matches_v
    bool has_ssnet_scores; ///< ssnet scores have been provided
    bool has_kplabel_scores; ///< keypoint scores have been provided
    bool has_paf; ///< particle affinity field directions have been provided

    /**
     * \brief reset state and clear member containers
     */
    void clear() {
      _matches_v.clear();
      _match_map.clear();
      has_ssnet_scores=false;
      has_kplabel_scores=false;
      has_paf=false; };
    int add_match_data( PyObject* pair_probs,
                        PyObject* source_sparseimg, PyObject* target_sparseimg,
                        PyObject* matchpairs,
                        const int source_plane, const int target_plane,
                        const larcv::ImageMeta& source_meta,
                        const std::vector<larcv::Image2D>& img_v,
                        const larcv::EventChStatus& ev_chstatus );

    /** \brief spacepoints with larmatch score below set value will not be stored */
    void set_score_threshold( float score ) { _match_score_threshold = score; }; 

    int add_triplet_match_data( PyObject* triple_probs,
                                PyObject* triplet_indices,
                                PyObject* sparseimg_u,
                                PyObject* sparseimg_v,
                                PyObject* sparseimg_y,
                                const std::vector< std::vector<float> >& pos_vv,
                                const std::vector<larcv::Image2D>& adc_v );

    int add_triplet_ssnet_scores( PyObject* triplet_indices,
                                  PyObject* sparseimg_u,
                                  PyObject* sparseimg_v,
                                  PyObject* sparseimg_y,
                                  const larcv::ImageMeta& meta,                                  
                                  PyObject* ssnet_scores );

    int add_triplet_keypoint_scores( PyObject* triplet_indices,
                                     PyObject* sparseimg_u,
                                     PyObject* sparseimg_v,
                                     PyObject* sparseimg_y,
                                     const larcv::ImageMeta& meta,                                     
                                     PyObject* kplabel_scores );

    int add_triplet_affinity_field( PyObject* triplet_indices,
                                    PyObject* imgu_sparseimg,
                                    PyObject* imgv_sparseimg,
                                    PyObject* imgy_sparseimg,
                                    const larcv::ImageMeta& meta,                                                      
                                    PyObject* paf_pred );    
    
    void make_hits( const larcv::EventChStatus& ev_chstatus,
                    std::vector<larlite::larflow3dhit>& hit_v )  const;

  };

}
}

#endif
