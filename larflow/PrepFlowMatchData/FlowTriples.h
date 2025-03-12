#ifndef __FLOW_TRIPLES_H__
#define __FLOW_TRIPLES_H__


#include <map>
#include <vector>

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "ublarcvapp/MCTools/MCPixelPGraph.h"
#include "TH2D.h"

#include "larflow/PrepFlowMatchData/PixData_t.h"
#include "larflow/PrepFlowMatchData/CropPixData_t.h"

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

#ifndef __CINT__
#ifndef __CLING__
    static std::vector< std::vector<larflow::prep::PixData_t> >
      make_initial_sparse_image( const std::vector<larcv::Image2D>& adc_v, float threshold );

    static std::vector< std::vector<larflow::prep::PixData_t> >
      make_cropped_initial_sparse_prong_image_truth( const std::vector<larcv::Image2D>& adc_v, 
                                                     ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                     larlite::storage_manager& ioll, 
                                                     int trackid, float threshold,
                                                     int rowSpan, int colSpan,
                                                     bool shower=true );

    static std::vector< std::vector<larflow::prep::CropPixData_t> >
      make_cropped_initial_sparse_prong_image_reco( const std::vector<larcv::Image2D>& adc_v, 
                                                    const std::vector<larcv::Image2D>& thrumu_v,
                                                    const larlite::larflowcluster& prong,
                                                    const TVector3& cropCenter, 
                                                    float threshold, int rowSpan, int colSpan );

    static std::vector< std::vector<larflow::prep::CropPixData_t> >
      make_cropped_initial_sparse_prong_image_reco_rmContextPart( const std::vector<larcv::Image2D>& adc_v, 
                                                                  const std::vector<larcv::Image2D>& thrumu_v,
                                                                  const larlite::larflowcluster& prong,
                                                                  const TVector3& cropCenter, 
                                                                  float threshold, int rowSpan, int colSpan,
                                                                  ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                                  int trackid_rm, int trackid_rm2=-1 );

    static std::vector< std::vector<larflow::prep::CropPixData_t> >
      make_cropped_initial_sparse_prong_image_reco_subContextPart( const std::vector<larcv::Image2D>& adc_v_reco,
                                                                   const std::vector<larcv::Image2D>& adc_v_sim, 
                                                                   const std::vector<larcv::Image2D>& thrumu_v,
                                                                   const larlite::larflowcluster& prong,
                                                                   const TVector3& cropCenter, 
                                                                   float threshold, int rowSpan, int colSpan,
                                                                   ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                                   int trackid_rm, int trackid_rm2=-1 );

    static std::vector< std::vector<larflow::prep::CropPixData_t> >
      make_cropped_initial_sparse_prong_image_reco_truthProngSub( const std::vector<larcv::Image2D>& adc_v_reco,
                                                                  const std::vector<larcv::Image2D>& adc_v_sim, 
                                                                  const std::vector<larcv::Image2D>& thrumu_v,
                                                                  const larlite::larflowcluster& prong,
                                                                  const TVector3& cropCenter, 
                                                                  float threshold, int rowSpan, int colSpan,
                                                                  ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                                                  int trackid_rm, int trackid_rm2=-1 );
#endif
#endif

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
#ifndef __CINT__
#ifndef __CLING__
    std::vector< std::vector<PixData_t> >& getDeadChToAdd() { return _deadch_to_add; };
#endif
#endif

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
    
      
  private:

    static void getRecoImageBounds( std::vector< std::vector<int> >& imgBounds,
                             const std::vector<larcv::Image2D>& adc_v, 
                             const larlite::larflowcluster& prong,
                             const TVector3& cropCenter, int rowSpan, int colSpan );

    static void fillProngImagesFromReco(std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                                 const float& threshold,
                                 const std::vector<larcv::Image2D>& adc_v,
                                 const std::vector<larcv::Image2D>& thrumu_v,
                                 const larlite::larflowcluster& prong,
                                 const std::vector< std::vector<int> >& imgBounds);

    static void fillProngImagesFromTruth(std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                                  const float& threshold,
                                  const std::vector<larcv::Image2D>& adc_v,
                                  const std::vector< std::vector<int> >& imgBounds);

    static void fillContextImages(std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                           const float& threshold,
                           const std::vector<larcv::Image2D>& adc_v,
                           const std::vector<larcv::Image2D>& thrumu_v,
                           const std::vector< std::vector<int> >& imgBounds);

    static void fillContextImages(std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                           const float& threshold,
                           const std::vector<larcv::Image2D>& adc_v,
                           const std::vector< std::vector<int> >& imgBounds);

    static void fillPartRmContextImages(std::vector< std::vector<CropPixData_t> >& sparseimg_vv,
                                 const float& threshold,
                                 const std::vector<larcv::Image2D>& adc_v,
                                 const std::vector<larcv::Image2D>& thrumu_v,
                                 const std::vector< std::vector<int> >& imgBounds,
                                 ublarcvapp::mctools::MCPixelPGraph& mcpg,
                                 const int& trackid_rm, const int& trackid_rm2=-1);


  };

}
}

#endif
