#ifndef __FLOW_TRIPLES_H__
#define __FLOW_TRIPLES_H__

#include <map>
#include <vector>

#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "TH2D.h"

namespace larflow {

  
  class FlowTriples {

  public:

    struct PixData_t {
      int row;
      int col;
      float val;
      int idx;
      PixData_t() {};
      PixData_t( int r, int c, float v)
      : row(r),col(c),val(v) {};
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

    int get_source_plane_index() { return _source_plane; };
    int get_target_plane_index() { return _target_plane; };
    int get_other_plane_index()  { return _other_plane; };

    std::vector<TH2D> plot_triple_data( const std::vector<larcv::Image2D>& adc_v,
                                        const std::vector< std::vector<PixData_t> >& sparseimg_vv,                                        
                                        std::string hist_stem_name );
    
    std::vector<TH2D> plot_sparse_data( const std::vector<larcv::Image2D>& adc_v,
                                        const std::vector< std::vector<PixData_t> >& sparseimg_vv,
                                        std::string hist_stem_name );

    std::vector< std::vector<PixData_t> >& getDeadChToAdd() { return _deadch_to_add; };
    std::vector< std::vector<int> >&       getTriples() { return _triple_v; };
                                                            
  protected:

    int _source_plane;
    int _target_plane;
    int _other_plane;

    std::vector< std::vector< PixData_t > > _sparseimg_vv;
    std::vector< std::vector<int> >         _triple_v;
    std::vector< std::vector<PixData_t> >   _deadch_to_add;

    void _makeTriples( int source, int target,
                       const std::vector<larcv::Image2D>& adc_v,
                       const std::vector<larcv::Image2D>& badch_v,
                       const std::vector< std::vector<PixData_t> >& sparseimg_vv,                                  
                       float threshold, bool save_index );
    
      
  };
    
}

#endif
