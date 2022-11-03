#ifndef __FLOW_TRIPLES_H__
#define __FLOW_TRIPLES_H__


#include <map>
#include <vector>

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "TH2D.h"

namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData 
   * @class FlowTriples
   * @brief Generate and store (U,V,Y) wire combintations extracted from examining coincident ionization between two planes
   *
   * @author Taritree Wongjirad (taritree.wongjirad@tufts.edu)
   * @date $Data 2020/07/22 17:00$
   *
   * Revision history
   * 2020/07/22: Added doxygen documentation. 
   * 
   *
   */  
  class FlowTriples {

  public:

    /** 
     * @struct PixData_t
     *
     * @brief internal struct to represent to a pixel and provide sorting method
     *
     */
    struct PixData_t {
      int row; ///< row of pixel in image
      int col; ///< col of pixel in image
      float val; ///< value of pixel
      int idx;   ///< index in container
      
      PixData_t()
      : row(0),col(0),val(0.0),idx(0)
      {};

      /** @brief constructor with row, col, value 
       *  @param[in] r row of pixel
       *  @param[in] c col of pixel
       *  @param[in] v value of pixel
       */      
      PixData_t( int r, int c, float v)
      : row(r),col(c),val(v),idx(0) {};

      /** @brief comparator based on row then col then value */
      bool operator<( const PixData_t& rhs ) const {
        if (row<rhs.row) return true;
        if ( row==rhs.row ) {
          if ( col<rhs.col ) return true;
          if ( col==rhs.col ) {
            if ( val<rhs.val ) return true;
          }
        }
        return false;
      };
    };
    
    /** 
     * @struct CropPixData_t
     *
     * @brief internal struct to represent to a pixel in cropped image and provide sorting method
     *
     */
    struct CropPixData_t {
      int row; ///< row of pixel in cropped image
      int col; ///< col of pixel in cropped image
      int rawRow; ///< row of pixel in original image
      int rawCol; ///< col of pixel in original image
      float val; ///< value of pixel
      int idx;   ///< index in container
      
      CropPixData_t()
      : row(0),col(0),rawRow(0),rawCol(0),val(0.0),idx(0)
      {};

      /** @brief constructor with row, col, value 
       *  @param[in] r row of pixel
       *  @param[in] c col of pixel
       *  @param[in] v value of pixel
       */      
      CropPixData_t( int r, int c, int rr, int rc, float v)
      : row(r),col(c),rawRow(rr),rawCol(rc),val(v),idx(0) {};

      /** @brief comparator based on row then col then value */
      bool operator<( const CropPixData_t& rhs ) const {
        if (row<rhs.row) return true;
        if ( row==rhs.row ) {
          if ( col<rhs.col ) return true;
          if ( col==rhs.col ) {
            if ( val<rhs.val ) return true;
          }
        }
        return false;
      };
    };
    
    FlowTriples()
      : _source_plane(-1),
      _target_plane(-1),
      _other_plane(-1) {
    };
    
    FlowTriples( int source_plane, int target_plane,
                 const std::vector<larcv::Image2D>& adc_v,
                 const std::vector<larcv::Image2D>& badch_v,
                 float threshold, bool save_index );

    FlowTriples( int source, int target,
                 const std::vector<larcv::Image2D>& adc_v,
                 const std::vector<larcv::Image2D>& badch_v,
                 const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                 float threshold, bool save_index );
    
    
    virtual ~FlowTriples() {};

    /** @brief number of pixels in the source image */
    int nsourceIndices() const {
      if ( _source_plane==-1 ) return 0;
      return (int)_sparseimg_vv[_source_plane].size();
    };

    // retrieve candidate matches to source pixel via index
    //const std::vector<int>& getTargetIndices( int src_index ) const;
    //const std::vector<int>& getTruthVector( int src_index )   const;

    // retrieve candidate matches to source image via target index
    //const std::vector<int>& getTargetIndicesFromSourcePixel( int col, int row ) const;
    //const std::vector<int>& getTruthVectorFromSourcePixel( int col, int row ) const;

    static std::vector< std::vector<FlowTriples::PixData_t> >
      make_initial_sparse_image( const std::vector<larcv::Image2D>& adc_v, float threshold );

    static std::vector< std::vector<FlowTriples::PixData_t> >
      make_initial_sparse_prong_image( const std::vector<larcv::Image2D>& adc_v, 
                                       ublarcvapp::mctools::MCPixelPGraph& mcpg, 
                                       int trackid, float threshold );

    static std::vector< std::vector<FlowTriples::PixData_t> >
      make_cropped_initial_sparse_prong_image( const std::vector<larcv::Image2D>& adc_v, 
                                               ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                               larlite::storage_manager& ioll, 
                                               int trackid, float threshold,
                                               int rowSpan, int colSpan,
                                               bool shower=true );

    static std::vector< std::vector<FlowTriples::PixData_t> >
      make_cropped_initial_sparse_prong_image_wMask( const std::vector<larcv::Image2D>& adc_v, 
                                                     ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                     larlite::storage_manager& ioll, 
                                                     int trackid, float threshold,
                                                     int rowSpan, int colSpan,
                                                     bool shower=true );

    static std::vector< std::vector<FlowTriples::CropPixData_t> >
      make_cropped_initial_sparse_prong_image_reco( const std::vector<larcv::Image2D>& adc_v, 
                                                    const std::vector<larcv::Image2D>& thrumu_v,
                                                    const larlite::larflowcluster& prong,
                                                    const TVector3& cropCenter, 
                                                    float threshold, int rowSpan, int colSpan );

    /** @brief index of the source plane considered */
    int get_source_plane_index() { return _source_plane; };

    /** @brief index of the target plane considered */    
    int get_target_plane_index() { return _target_plane; };

    /** @brief index of the other (not source or target) plane considered */
    int get_other_plane_index()  { return _other_plane; };

    std::vector<TH2D> plot_triple_data( const std::vector<larcv::Image2D>& adc_v,
                                        const std::vector< std::vector<PixData_t> >& sparseimg_vv,                                        
                                        std::string hist_stem_name );
    
    std::vector<TH2D> plot_sparse_data( const std::vector<larcv::Image2D>& adc_v,
                                        const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                                        std::string hist_stem_name );

    std::vector<TH2D> plot_cropped_sparse_data( int rowSpan, int colSpan,
                                        const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                                        std::string hist_stem_name );

    std::vector<TH2D> plot_cropped_sparse_data( int rowSpan, int colSpan,
                                        const std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                                        std::string hist_stem_name );

    /** @brief get pixels in each plane that are dead */
    std::vector< std::vector<PixData_t> >& getDeadChToAdd() { return _deadch_to_add; };

    /** @brief get the combination of three wires with coincident charge seen */
    std::vector< std::vector<int> >&       getTriples() { return _triple_v; };
                                                            
  protected:

    int _source_plane; ///< index of the source plane considered
    int _target_plane; ///< index of the target plane considered
    int _other_plane;  ///< index of the other (non-source, non-target) plane considered

    std::vector< std::vector< PixData_t > > _sparseimg_vv;  ///< stores non-zero pixel information for each plane
    std::vector< std::vector<int> >         _triple_v;      ///< combination of three wire plane pixels with coincident charge
    std::vector< std::vector<PixData_t> >   _deadch_to_add; ///< list of dead channels in each plane

    void _makeTriples( int source, int target,
                       const std::vector<larcv::Image2D>& adc_v,
                       const std::vector<larcv::Image2D>& badch_v,
                       const std::vector< std::vector<PixData_t> >& sparseimg_vv,                                  
                       float threshold, bool save_index );
    
      
  };

}
}

#endif
