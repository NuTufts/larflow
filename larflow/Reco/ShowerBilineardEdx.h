#ifndef __LARFLOW_RECO_SHOWER_BILINEAR_DEDX_H__
#define __LARFLOW_RECO_SHOWER_BILINEAR_DEDX_H__

#include <vector>
#include <map>
#include "TGraph.h"
#include "TTree.h"
#include "TH2D.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "DataFormat/larflowcluster.h"
#include "DataFormat/pcaxis.h"
#include "DataFormat/track.h"
#include "DataFormat/mcshower.h"
#include "LArUtil/SpaceChargeMicroBooNE.h"

namespace larflow {
namespace reco {

  class ShowerBilineardEdx : public larcv::larcv_base {
  public:
    
    ShowerBilineardEdx();
    virtual ~ShowerBilineardEdx();

    struct Result_t {
      std::vector<float> trunk_dedx_planes;
      std::vector< std::vector<float> > dedx_curve_planes;
    };

    void processShower( larlite::larflowcluster& shower,
                        larlite::track& trunk,
                        larlite::pcaxis& pca,
                        const std::vector<larcv::Image2D>& adc_v );

    float aveBilinearCharge_with_grad( const larcv::Image2D& img,
                                       std::vector<float>& start3d,
                                       std::vector<float>& end3d,
                                       int npts,
                                       float avedQdx,
                                       std::vector<float>& grad );

    float colcoordinate_and_grad( const std::vector<float>& pos,
                                  const int plane,
                                  const larcv::ImageMeta& meta,
                                  std::vector<float>& grad );
    
    float rowcoordinate_and_grad( const std::vector<float>& pos,
                                  const larcv::ImageMeta& meta,
                                  std::vector<float>& grad );

    float bilinearPixelValue_and_grad( std::vector<float>& pos3d,
                                       const int plane,
                                       const larcv::Image2D& img,
                                       std::vector<float>& grad );
    
    std::vector<float> sumChargeAlongTrunk( const std::vector<float>& start3d,
                                            const std::vector<float>& end3d,
                                            const std::vector<larcv::Image2D>& img_v,
                                            const float threshold,
                                            const int dcol,
                                            const int drow );
    

    void bindVariablesToTree( TTree* outtree );

    TGraph makeSegdQdxGraphs(int plane);

    void clear();
    
    // for debug
    std::vector< std::vector<TGraph> > bilinear_path_vv;
    std::vector< float > _shower_dir;
    std::vector< float > _pixsum_dqdx_v;
    std::vector< float > _bilin_dqdx_v;

    // pixel lists, sorted by position on trunk
    struct TrunkPix_t {
      int row;
      int col;      
      float smin;
      float smax;
      float s;
      TrunkPix_t()
      : row(0),
        col(0),
        smin(0),
        smax(0),
        s(0)
      {};
      TrunkPix_t( int r, int c, float s1, float s2 )
      : row(r),
        col(c),
        smin(s1),
        smax(s2),
        s(s1)
      {};
      bool operator<( const TrunkPix_t&  rhs ) const {
        if (s<rhs.s)
          return true;
        return false;
      };
    };
    typedef std::vector<TrunkPix_t> TrunkPixList_t;
    
    typedef std::map< std::pair<int,int>, TrunkPix_t > TrunkPixMap_t;
    std::vector< TrunkPixMap_t > _visited_v;
    std::vector< TrunkPixList_t > _plane_trunkpix_v;
    void _createDistLabels( const std::vector<float>& start3d,
                            const std::vector<float>& end3d,
                            const std::vector<larcv::Image2D>& img_v,
                            const float threshold );
    void maskPixels( int plane, TH2D* hist );

    struct Seg_t {
      float s;
      float smin;
      float smax;
      int itp1;
      int itp2;
      int plane;            
      float endpt[2][3];
      float pixsum;
      float dqdx;
      float ds;
    };
    typedef std::vector<Seg_t> SegList_t;
    std::vector< SegList_t > _plane_seg_dedx_v;
    std::vector< std::vector<float> > _plane_dqdx_seg_v;
    std::vector< std::vector<float> > _plane_s_seg_v;        
    void _makeSegments( float starting_s );
    void _sumChargeAlongSegments( const std::vector<float>& start3d,
                                  const std::vector<float>& end3d,
                                  const std::vector<larcv::Image2D>& img_v,
                                  const float threshold,
                                  const int dcol, const int drow );

    std::vector<TH2D> _debug_crop_v;

    std::vector< std::vector<float> > _plane_electron_srange_v;
    std::vector< float > _plane_electron_dqdx_v;
    std::vector< float > _plane_electron_dx_v;
    
    std::vector< std::vector<float> > _plane_gamma_srange_v;    
    std::vector< std::vector<float> > _plane_gamma_dqdx_v;
    std::vector< std::vector<float> > _plane_gamma_dx_v;
    
    void _findRangedQdx( const std::vector<float>& start3d,
                         const std::vector<float>& end3d,
                         const std::vector<larcv::Image2D>& adc_v,
                         const float dqdx_max,
                         const float dqdx_threshold );
    float _sumChargeAlongOneSegment( ShowerBilineardEdx::Seg_t& seg,
                                     const int plane,
                                     const std::vector<larcv::Image2D>& img_v,
                                     const float threshold,
                                     const int dcol, const int drow );

    float _true_min_feat_dist;
    float _true_vertex_err_dist;
    float _true_dir_cos;
    void calcGoodShowerTaggingVariables( const larlite::larflowcluster& shower,
                                         const larlite::track& trunk,
                                         const larlite::pcaxis& pca,
                                         const std::vector<larcv::Image2D>& adc_v,
                                         const std::vector<larlite::mcshower>& mcshower_v );

  private:
    
    static int ndebugcount;
    static larutil::SpaceChargeMicroBooNE* _psce;
    
  };
  
}
}

#endif
