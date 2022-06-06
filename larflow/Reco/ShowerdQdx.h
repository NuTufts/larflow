#ifndef __LARFLOW_RECO_SHOWER_DQDX_H__
#define __LARFLOW_RECO_SHOWER_DQDX_H__

#include <vector>
#include <map>
#include "TGraph.h"
#include "TTree.h"
#include "TH2D.h"
#include "larcv/core/Base/larcv_base.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larlite/DataFormat/larflowcluster.h"
#include "larlite/DataFormat/pcaxis.h"
#include "larlite/DataFormat/track.h"
#include "larlite/DataFormat/mcshower.h"
#include "larlite/DataFormat/mctrack.h"
#include "larlite/LArUtil/SpaceChargeMicroBooNE.h"
#include "NuVertexCandidate.h"

namespace larflow {
namespace reco {

  /**
   * @class ShowerBilineardQdx
   * @ingroup Reco
   *
   * Calculate the dq/dx using the beginning of the shower trunk.
   *
   */
  class ShowerdQdx : public larcv::larcv_base {
  public:
    
    ShowerdQdx();
    virtual ~ShowerdQdx();

    void processShower( const larlite::larflowcluster& shower,
                        const larlite::track& trunk,
                        const larlite::pcaxis& pca,
                        const std::vector<larcv::Image2D>& adc_v,
                        const larflow::reco::NuVertexCandidate& nuvtx );

    void processMCShower( const larlite::mcshower& shower,
                          const std::vector<larcv::Image2D>& adc_v,
                          const larflow::reco::NuVertexCandidate& nuvtx );

    void clear();    
    

    float colcoordinate_and_grad( const std::vector<float>& pos,
                                  const int plane,
				  const int tpcid,
				  const int cryoid,
				  const int nplanes,
                                  const larcv::ImageMeta& meta,
                                  std::vector<float>& grad );
    
    float rowcoordinate_and_grad( const std::vector<float>& pos,
				  const int plane,
				  const int tpcid,
				  const int cryoid,
                                  const larcv::ImageMeta& meta,
                                  std::vector<float>& grad );
    
    std::vector<float> sumChargeAlongTrunk( const std::vector<float>& start3d,
                                            const std::vector<float>& end3d,
                                            const std::vector<larcv::Image2D>& img_v,
                                            const float threshold,
                                            const int dcol,
                                            const int drow,
					    const int tpcid, const int cryoid );
    

    void bindVariablesToTree( TTree* outtree );

    TGraph makeSegdQdxGraphs(int plane);

    larlite::track makeLarliteTrackdqdx(int plane);
    
    std::vector<larcv::Image2D> maskTrackPixels( const std::vector<const larcv::Image2D*>& adc_v,
                                                 const larlite::track& shower_trunk,
                                                 const larflow::reco::NuVertexCandidate& nuvtx,
						 const int tpcid, const int cryoid );


    
    std::vector< std::vector<TGraph> > trunk_tgraph_vv; ///< trunk projected into image planes, for debug
    std::vector< float > _shower_dir;    ///< direction of the last shower processed
    std::vector< float > _pixsum_dqdx_v; ///< dq/dx measured on each plane by summing pixels
    float _best_pixsum_dqdx;   ///< dq/dx from "best" plane, chosen using hueristic
    int   _best_pixsum_plane;  ///< plane from which _best_pixsum_dqdx came from
    float _best_pixsum_ortho;  ///< cosine between _shower_dir and the orthonormal vector of the plane _best_pixsum_dqdx came from
    
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
    }; ///< data for wire plane pixels through which a given shower drunk moves through
    typedef std::vector<TrunkPix_t> TrunkPixList_t; ///< a collection of TrunkPix_t
    typedef std::map< std::pair<int,int>, TrunkPix_t > TrunkPixMap_t; // provide map to look up pixel location to TrunkPix data
    std::vector< TrunkPixMap_t >    _visited_v; ///< pixel map for each plane
    std::vector< TrunkPixList_t >   _plane_trunkpix_v; ///< collection of TrunkPix_t for each plane
    void _createDistLabels( const std::vector<float>& start3d,
                            const std::vector<float>& end3d,
                            const std::vector<const larcv::Image2D*>& img_v,
                            const float threshold,
			    const int tpcid, const int cryoid );
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
    }; ///< line segment along the trunk over which we will measure dq/dx
    typedef std::vector<Seg_t> SegList_t; ///< collecton of segments
    std::vector< SegList_t > _plane_seg_dedx_v;           ///< collection of segments for each plane
    std::vector< std::vector<float> > _plane_dqdx_seg_v;  ///< the dq/dx measured by each segment for each plane's collection
    std::vector< std::vector<float> > _plane_s_seg_v;     ///< the distance of the segment from the start of the trunk
    void _makeSegments( const float starting_s, const float seg_size );
    void _sumChargeAlongSegments( const std::vector<float>& start3d,
                                  const std::vector<float>& end3d,
                                  const std::vector<larcv::Image2D>& img_v,
                                  const float threshold,
                                  const int dcol, const int drow,
				  const int tpcid, const int cryoid );

    std::vector<TH2D> _debug_crop_v;

    std::vector< std::vector<float> > _plane_electron_srange_v;
    std::vector< float > _plane_electron_dqdx_v;
    std::vector< float > _plane_electron_dx_v;
    std::vector< float > _plane_electron_mean_v;
    std::vector< float > _plane_electron_rms_v;
    std::vector< int >   _plane_electron_ngood_v;
    int _plane_electron_best;
    float _plane_electron_best_mean;
    float _plane_electron_best_rms;
    int _plane_electron_best_ngood;
    float _plane_electron_best_start;
    
    std::vector< std::vector<float> > _plane_gamma_srange_v;    
    std::vector< float > _plane_gamma_dqdx_v;
    std::vector< float > _plane_gamma_dx_v;
    std::vector< float > _plane_gamma_mean_v;
    std::vector< float > _plane_gamma_rms_v;
    std::vector< int >   _plane_gamma_ngood_v;        
    int _plane_gamma_best;
    float _plane_gamma_best_mean;
    float _plane_gamma_best_rms;
    int _plane_gamma_best_ngood;
    float _plane_gamma_best_start;
    
    void _findRangedQdx( const std::vector<float>& start3d,
                         const std::vector<float>& end3d,
                         const std::vector<larcv::Image2D>& adc_v,
                         const float dqdx_max,
                         const float dqdx_threshold,
                         std::vector<float>& plane_dqdx_v,
                         std::vector<float>& plane_dx_v,
                         std::vector< std::vector<float> >& plane_srange_v,
                         std::vector<float>& plane_mean_v,
                         std::vector<float>& plane_rms_v,
                         std::vector<int>& plane_ngood_v,
                         int& best_plane,
                         int& plane_max_ngood,
                         float& plane_best_dqdx,
                         float& plane_best_rms,
                         float& plane_best_start);
                         
                         
    float _sumChargeAlongOneSegment( ShowerdQdx::Seg_t& seg,
                                     const int plane,
                                     const std::vector<larcv::Image2D>& img_v,
                                     const float threshold,
                                     const int dcol, const int drow );

    float _true_min_feat_dist;
    float _true_vertex_err_dist;
    float _true_dir_cos;
    int _true_match_pdg;
    int _true_min_index;

    void calcGoodShowerTaggingVariables( const larlite::larflowcluster& shower,
                                         const larlite::track& trunk,
                                         const larlite::pcaxis& pca,
                                         const std::vector<const larcv::Image2D*>& adc_v,
                                         const std::vector<larlite::mcshower>& mcshower_v,
					 const int tpcid, const int cryoid );

    bool checkShowerTrunk( const std::vector<float>& start_pos,
                           const std::vector<float>& end_pos,
                           std::vector<float>& modstart3d,
                           std::vector<float>& modend3d,
                           std::vector<float>& shower_dir,
                           float& dist,
			   const int tpcid,
			   const int cryoid,
                           const std::vector<const larcv::Image2D*>& padc_v );
    
    void matchMCShowerAndProcess( const larlite::larflowcluster& reco_shower,
                                  const larlite::track& reco_shower_trunk,
                                  const larlite::pcaxis& reco_pca,
                                  const std::vector<larcv::Image2D>& adc_v,
                                  const larflow::reco::NuVertexCandidate& nuvtx,                                                    
                                  const std::vector<larlite::mcshower>& mcshower_v );

    float _true_max_primary_cos;
    void checkForOverlappingPrimary( const larlite::event_mctrack& ev_mctrack,
                                     const larlite::event_mcshower& ev_mcshower );    
    
  private:
    
    static int ndebugcount; ///< increment in order to give th2d unique names -- for debugging
    static larutil::SpaceChargeMicroBooNE* _psce; ///< utility to apply space charge to truth information
    
  };
  
}
}

#endif
