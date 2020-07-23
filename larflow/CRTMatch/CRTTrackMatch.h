#ifndef __LARFLOW_CRTTRACK_MATCH_H__
#define __LARFLOW_CRTTRACK_MATCH_H__

#include <vector>

#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larlite/core/DataFormat/storage_manager.h"
#include "larlite/core/DataFormat/crttrack.h"
#include "larlite/core/DataFormat/opflash.h"
#include "larlite/core/DataFormat/larflowcluster.h"
#include "larlite/core/LArUtil/SpaceChargeMicroBooNE.h"

#include "larflow/Reco/cluster_functions.h"

namespace larflow {
namespace crtmatch {

  /**
   * @ingroup CRTMatch
   * @class CRTTrackMatch
   * @brief Match CRTTrack objects to lines of charge in the image
   *
   * The `larlite::crttrack` object contains the (x,y,z) positions of two
   * CRT hits conincident at some time 't'. We can use the two spatial
   * coordinates to draw a line through the TPC and look for charge along that line.
   * The 't' can be used to match with an optical flash.
   * To match the straight-line track with an ionization track, 
   * we assume that no scattering has occured to change the path.
   * We take into account space charge effects that will distort the path of
   * the apparent path of the track.
   * We also account for potential uncertainty in the CRT hit positions by 
   * varying the hit positions to make the best fit of the predicted ionization trajectory
   * with one found in the image.
   *
   */  
  class CRTTrackMatch : public larcv::larcv_base {

  public:

    CRTTrackMatch();
    virtual ~CRTTrackMatch();

    void process( larcv::IOManager& iolcv, larlite::storage_manager& ioll );
    void set_max_iters( int tries ) { _max_iters = tries; };
    void set_col_neighborhood( int cols )   { _col_neighborhood = cols; };
    void set_max_fit_step_size( float cm )  { _max_fit_step_size = cm; };
    void set_max_last_step_size( float cm ) { _max_last_step_size = cm; };
    void make_debug_images( bool make_debug ) { _make_debug_images = make_debug; };

    void save_to_file( larlite::storage_manager& ioll, bool remove_if_no_flash=true );
    void clear_output_containers();

    /**
     * @brief internal struct to assemble info for CRT Track
     *
     */
    struct crttrack_t {

      int crt_track_index;                 ///< index in event container of larlite::crttrack object
      const larlite::crttrack* pcrttrack;  ///< pointer to original CRT track object

      std::vector< std::vector<double> >   hit_pos_vv;      ///< modified CRT hit positions (after fit)
      std::vector< std::vector< int > >    pixellist_vv[3]; ///< (row,col) of pixels near line for each plane (3 planes total)
      std::vector< float >                 pixelrad_vv[3];  ///< pixel radius from crt-track line for each plane (3 planes total)
      std::vector< std::vector< double > > pixelpos_vv;     ///< list of pixel positions near the crt-track line
      std::vector< std::vector<int> >      pixelcoord_vv;   ///< list of pixel coordinates (row,col)
      std::vector< float >                 totalq_v;        ///< total charge along path
      std::vector< float >                 toterr_v;        ///< total error
      float                                len_intpc_sce;   ///< path length of the reconstructed path inside the TPC
      float                                t0_usec;         ///< time relative to beam window start in microsecond of CRT track

      int opflash_index;                                    ///< index of matched opflash in event container
      const larlite::opflash*  opflash;                     ///< pointer to opflash object

      crttrack_t()
      : crt_track_index(-1),
        pcrttrack( nullptr ),
        totalq_v( {0,0,0} ),
        toterr_v( {0,0,0} ),
        len_intpc_sce(0.0)
      {};
      crttrack_t( int idx, const larlite::crttrack* ptrack )
      : crt_track_index(idx),
        pcrttrack( ptrack ),
        totalq_v( {0,0,0} ),
        toterr_v( {0,0,0} ),
        len_intpc_sce(0.0)
      {};
      
    };

    // methods
    crttrack_t _find_optimal_track( const larlite::crttrack& crt,
                                    const std::vector<larcv::Image2D>& adc_v );

    
    crttrack_t _collect_chargepixels_for_track( const std::vector<double>& hit1_pos,
                                                const std::vector<double>& hit2_pos,
                                                const float t0_usec,
                                                const std::vector<larcv::Image2D>& adc_v,
                                                const float max_step_size,
                                                const int col_neighborhood );
    
    bool _propose_corrected_track( const CRTTrackMatch::crttrack_t& old,
                                   std::vector< double >& hit1_new,
                                   std::vector< double >& hit2_new );

    bool _crt_intersections( larflow::reco::cluster_t& track_cluster,
                             const larlite::crttrack& orig_crttrack,
                             std::vector< std::vector<double> >& panel_pos_v,
                             std::vector< double >& dist2_original_hits );
    


    std::string _str( const CRTTrackMatch::crttrack_t& data );

    void _matchOpflashes( std::vector< const larlite::event_opflash* > flash_vv,
                          const std::vector<CRTTrackMatch::crttrack_t>& tracks_v,
                          std::vector< larlite::opflash >& matched_opflash_v );
    
    larlite::larflowcluster _crttrack2larflowcluster( const CRTTrackMatch::crttrack_t& fitdata );    

    larutil::SpaceChargeMicroBooNE* _sce;          ///< pointer to space charge correction class
    larutil::SpaceChargeMicroBooNE* _reverse_sce;  ///< pointer to reverse space charge correction class

    /// algorithm parameters
    
    int _max_iters;            ///< max attempts to improve CRT track fit
    int _col_neighborhood;     ///< window to search for charge around track
    float _max_fit_step_size;  ///< max step size when fittering
    float _max_last_step_size; ///< max step size for final fit
    float _max_dt_flash_crt;   /// < maximum time difference (usec) between opflash and crttrack
    std::string _adc_producer; ///< name of ADC image tree
    std::string _crttrack_producer; ///< name of CRT track producer
    std::vector<std::string> _opflash_producer_v;  ///< names of opflash producers to search for matching flashes
    bool _make_debug_images;   ///< if true, dump debugging pngs

  public:

    void setADCtreename( std::string treename ) { _adc_producer=treename; }; ///< Set Image2D tree to get wire images from.

  protected:
    
    /// products of algorithm
    
    std::vector< crttrack_t >        _fitted_crttrack_v;   ///< result of crt track fitter
    std::vector< larlite::opflash >  _matched_opflash_v;   ///< opflashes matched to tracks in _fitted_crttrack_v
    std::vector< larlite::crttrack > _modified_crttrack_v; ///< crttrack object with modified end points
    std::vector< larlite::larflowcluster > _cluster_v;     ///< collections of larflow3dhit close to the 3D line

  public:

    const std::vector< larlite::larflowcluster >& getClusters() { return _cluster_v; }; ///< get matched track clusters

  };
  
}
}

#endif
