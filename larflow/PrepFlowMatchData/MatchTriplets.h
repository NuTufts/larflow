#ifndef __LARFLOW_PREP_MATCHTRIPLETS_H__
#define __LARFLOW_PREP_MATCHTRIPLETS_H__

#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include "larcv/core/DataFormat/ImageMeta.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "larflow/PrepFlowMatchData/FlowTriples.h"

namespace larflow {
namespace prep {

  class MatchTriplets {

  public:
    
    MatchTriplets()
      : _kshuffle_indices_when_sampling(false)
      {};
    virtual ~MatchTriplets() {};

    std::vector< larcv::ImageMeta >                       _imgmeta_v;       ///< image metas for the most recently processed event [elements is for planes]
    std::vector< std::vector< FlowTriples::PixData_t > >  _sparseimg_vv;    ///< sparse representation of image [elements is for planes]
    std::vector< std::vector<int> >                       _triplet_v;       ///< set of sparseimage indices indicating candidate 3-plane match (U-index,V-index,Y-index,tick)
    std::vector< std::vector<int> >                       _trip_cryo_tpc_v; ///< cryostat and tpc id for triplet [elements in triplet]
    std::vector< int >                                    _truth_v;         ///< indicates if index set in _triple_v is true match (1) or not (0)
    std::vector< std::vector<int> >                       _truth_2plane_v;  ///< truth vectors for 2 plane flows. inner vector is 1/0 for all 2-plane flow dirs [deprecate]
    std::vector< float >                                  _weight_v;        ///< assigned weight for triplet
    std::vector< larflow::FlowDir_t >                     _flowdir_v;       ///< flow direction te triplet comes from [deprecate]
    std::vector< float >                                  _triarea_v;       ///< area of triangle formed by the intersection of the 3 wires. measure of 3D consistency. [deprecate]
    std::vector< std::vector<float> >                     _pos_v;           ///< approx. 3d position of triplet
    std::vector< int >                                    _instance_id_v;   ///< instance ID label for each space point
    std::vector< int >                                    _ancestor_id_v;   ///< ancestor ID label for each space point
    std::vector< int >                                    _pdg_v;           ///< PDG label for each space point
    std::vector< int >                                    _origin_v;        ///< 0: unknown, 1:neutrino, 2:cosmic
    std::vector< int >                                    _match_span_v;    ///< distance from projection intersection to true intersection in wires

    void clear();

    bool _kshuffle_indices_when_sampling;    
    void setShuffleWhenSampling( bool shuffle ) { _kshuffle_indices_when_sampling = shuffle; }; ///< shuffle triplet index order when making numpy arrays

    // python/numpy functions, to help network interface
    PyObject* make_sparse_image( int plane ); // used in deploy/network data prep

    PyObject* get_all_triplet_data( const bool withtruth ); // used by deploy/network data prep

    PyObject* make_spacepoint_charge_array(); // used by deploy/network data prep
    

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
    
  private:

    static bool _setup_numpy;
    
  };    
  
}
}

#endif
