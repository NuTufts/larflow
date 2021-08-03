#ifndef __LARFLOW_PREP_PREPLARMATCHDATA_H__
#define __LARFLOW_PREP_PREPLARMATCHDATA_H__


#include <Python.h>
#include "bytesobject.h"

#include <vector>
#include "DataFormat/storage_manager.h"
#include "DataFormat/mcshower.h"
#include "larcv/core/DataFormat/Image2D.h"
#include "larcv/core/DataFormat/EventImage2D.h"
#include "larcv/core/DataFormat/EventChStatus.h"
#include "larcv/core/DataFormat/IOManager.h"
#include "larcv/core/Processor/ProcessBase.h"
#include "larflow/LArFlowConstants/LArFlowConstants.h"
#include "FlowTriples.h"
#include "TH2D.h"

namespace larflow {
namespace prep {

  /**
   * @ingroup PrepFlowMatchData 
   *
   * @class PrepMatchTriplets
   *
   * @brief Prepares potential spacepoints deriving from intersection of three wires.
   *
   * @author Taritree Wongjirad (taritree.wongjirad@tuts.edu)
   * @date $Data 2020/07/22 17:00$
   *
   * Revision history
   * 2020/07/22: Added doxygen documentation. 
   * 
   *
   */  
  class PrepMatchTriplets  {

  public:

    PrepMatchTriplets()
      : _kStopAtTripletMax(false),
      _kTripletLimit(1000000)
    {};
    virtual ~PrepMatchTriplets() {};

    
    void process( const std::vector<larcv::Image2D>& adc_v,
                  const std::vector<larcv::Image2D>& badch_v,
                  const float adc_threshold,
                  const bool check_wire_interection = false );
    
    void process( larcv::IOManager& iolcv,
                  std::string wire_producer,
                  std::string chstatus_producer,
                  const float adc_threshold=10.0,
                  const bool check_wire_intersection=false );

    void make_truth_vector( const std::vector<larcv::Image2D>& larflow_v );
    void make_instanceid_vector( const std::vector<larcv::Image2D>& instance_v );
    void make_ancestorid_vector( const std::vector<larcv::Image2D>& ancestor_v );    
    void make_segmentid_vector( const std::vector<larcv::Image2D>& segment_img_v,
                                const std::vector<larcv::Image2D>& adc_v );
    void process_truth_labels( larcv::IOManager& iolcv, larlite::storage_manager& ioll, std::string wire_producer="wire" );
    void setStopAtTripletMax( bool stop, int limit=1000000) { _kStopAtTripletMax = stop; _kTripletLimit = limit; };

    std::vector<int> get_triplet_imgcoord_rowcol( int idx_triplet );
    
    std::vector<TH2D> plot_sparse_images( std::string hist_stem_name );
                                          
    std::vector<TH2D> plot_truth_images( std::string hist_stem_name );

    std::vector< larcv::ImageMeta >                       _imgmeta_v;      ///< image metas for the most recently processed event
    std::vector< std::vector< FlowTriples::PixData_t > >  _sparseimg_vv;   ///< sparse representation of image
    std::vector< std::vector<int> >                       _triplet_v;      ///< set of sparseimage indices indicating candidate 3-plane match (U-index,V-index,Y-index,tick)
    std::vector< int >                                    _truth_v;        ///< indicates if index set in _triple_v is true match (1) or not (0)
    std::vector< std::vector<int> >                       _truth_2plane_v; ///< truth vectors for 2 plane flows. inner vector is 1/0 for all 2-plane flow dirs
    std::vector< float >                                  _weight_v;       ///< assigned weight for triplet
    std::vector< larflow::FlowDir_t >                     _flowdir_v;      ///< flow direction te triplet comes from
    std::vector< float >                                  _triarea_v;      ///< area of triangle formed by the intersection of the 3 wires. measure of 3D consistency.
    std::vector< std::vector<float> >                     _pos_v;          ///< approx. 3d position of triplet
    std::vector< int >                                    _instance_id_v;  ///< instance ID label for each space point
    std::vector< int >                                    _ancestor_id_v;  ///< ancestor ID label for each space point
    std::vector< int >                                    _pdg_v;          ///< PDG label for each space point
    std::vector< int >                                    _origin_v;       ///< 0: unknown, 1:neutrino, 2:cosmic
    void clear();

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

    PyObject* sample_hard_example_matches( const int& nsamples,
                                           const int& nhard_samples,
                                           PyObject* triplet_scores,                                                            
                                           int& nfilled,
                                           bool withtruth );

    // Truth Info
    PyObject* make_truthonly_triplet_ndarray();

    //std::vector<TH2D> plot_triplet_index_array( PyObject* np_index, PyObject* np_sparseimg, std::string hist_stem_name );

  protected:

    bool _kStopAtTripletMax;
    int  _kTripletLimit;

    // map from shower daughter IDs to mother IDs
    std::map<unsigned long, unsigned long> _shower_daughter2mother;
    void fill_daughter2mother_map( const std::vector<larlite::mcshower>& shower_v );

  private:
    
    static bool _setup_numpy; ///< true if numpy has been setup
    
  };

}
}

#endif
