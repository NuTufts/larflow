#ifndef __LARFLOW_PREP_PREPLARMATCHDATA_H__
#define __LARFLOW_PREP_PREPLARMATCHDATA_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/Processor/ProcessBase.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "FlowTriples.h"
#include "TH2D.h"

namespace larflow {

  class PrepMatchTriplets  {
  public:

    PrepMatchTriplets()
      : _setup_numpy(false)
    {};
    virtual ~PrepMatchTriplets() {};

    
    void process( const std::vector<larcv::Image2D>& adc_v,
                  const std::vector<larcv::Image2D>& badch_v,
                  const float adc_threshold );

    void make_truth_vector( const std::vector<larcv::Image2D>& larflow_v );
    
    std::vector<TH2D> plot_sparse_images( std::string hist_stem_name );
                                          

    std::vector<TH2D> plot_truth_images( std::string hist_stem_name );

    /* std::vector<TH2D> plot_truth2plane_images( const std::vector<larcv::Image2D>& adc_v, */
    /*                                            std::string hist_stem_name ); */
    
    std::vector< larcv::ImageMeta >                       _imgmeta_v;
    std::vector< std::vector< FlowTriples::PixData_t > >  _sparseimg_vv;   ///< sparse representation of image
    std::vector< std::vector<int> >                       _triplet_v;      ///< set of sparseimage indices indicating candidate 3-plane match (U-index,V-index,Y-index,tick)
    std::vector< int >                                    _truth_v;        ///< indicates if index set in _triple_v is true match (1) or not (0)
    std::vector< std::vector<int> >                       _truth_2plane_v; ///< truth vectors for 2 plane flows. inner vector is 1/0 for all 2-plane flow dirs
    std::vector< float >                                  _weight_v;       ///< assigned weight for triplet
    std::vector< larflow::FlowDir_t >                     _flowdir_v;      ///< flow direction te triplet comes from


    // python/numpy functions, to help network interface
    PyObject* make_sparse_image( int plane );
    
    PyObject* make_2plane_match_array( larflow::FlowDir_t kdir,
                                       const int max_num_samples,
                                       const std::vector<int>& idx_v,
                                       const int start_idx,
                                       const bool withtruth,
                                       int& nsamples );

    PyObject* sample_2plane_matches( larflow::FlowDir_t kdir,
                                     const int& nsamples,
                                     int& nfilled,
                                     bool withtruth );

    PyObject* get_chunk_2plane_matches( larflow::FlowDir_t kdir,
                                        const int& start_index,
                                        const int& max_num_pairs,
                                        int& last_index,
                                        int& num_pairs_filled,
                                        bool with_truth );

    bool _setup_numpy;
    
  };

}

#endif
