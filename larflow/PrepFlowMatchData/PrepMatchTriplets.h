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
                  const float adc_threshold,
                  const bool check_wire_interection = false );

    void make_truth_vector( const std::vector<larcv::Image2D>& larflow_v );
    
    std::vector<TH2D> plot_sparse_images( std::string hist_stem_name );
                                          
    std::vector<TH2D> plot_truth_images( std::string hist_stem_name );

    std::vector< larcv::ImageMeta >                       _imgmeta_v;
    std::vector< std::vector< FlowTriples::PixData_t > >  _sparseimg_vv;   ///< sparse representation of image
    std::vector< std::vector<int> >                       _triplet_v;      ///< set of sparseimage indices indicating candidate 3-plane match (U-index,V-index,Y-index,tick)
    std::vector< int >                                    _truth_v;        ///< indicates if index set in _triple_v is true match (1) or not (0)
    std::vector< std::vector<int> >                       _truth_2plane_v; ///< truth vectors for 2 plane flows. inner vector is 1/0 for all 2-plane flow dirs
    std::vector< float >                                  _weight_v;       ///< assigned weight for triplet
    std::vector< larflow::FlowDir_t >                     _flowdir_v;      ///< flow direction te triplet comes from
    std::vector< float >                                  _triarea_v;      ///< area of triangle formed by the intersection of the 3 wires. measure of 3D consistency.
    std::vector< std::vector<float> >                     _pos_v;          ///< approx. 3d position of triplet

    // python/numpy functions, to help network interface
    PyObject* make_sparse_image( int plane );

    // 2 plane preparation
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

    // Triplet preparation
    PyObject* make_triplet_array( const int max_num_samples,
                                  const std::vector<int>& idx_v,
                                  const int start_idx,
                                  const bool withtruth,
                                  int& nsamples );
    
    PyObject* sample_triplet_matches( const int& num_max_samples, int& nfilled, bool withtruth );

    PyObject* get_chunk_triplet_matches( const int& start_index,
                                         const int& max_num_pairs,
                                         int& last_index,
                                         int& num_pairs_filled,
                                         bool with_truth );
    
    //std::vector<TH2D> plot_triplet_index_array( PyObject* np_index, PyObject* np_sparseimg, std::string hist_stem_name );

    bool _setup_numpy;
    
  };

}

#endif
