#include "KeypointData.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include <set>
#include <stdexcept>

namespace larflow {
namespace keypoints {

  /**
   * static variable to track if numpy environment has been setup
   */
  bool KeypointData::_setup_numpy = false;  
  
  KeypointData::KeypointData()
    : tpcid(0),
      cryoid(0)
  {
  }

  /**
   * 
   * return a numpy array with keypoints for a given class
   *
   * @param[in] iclass larflow::KeyPoint_t value, indicating keypoint type
   * @return Numpy array with columns [tick,wire-U,wire-V,wire-Y,x,y,z,isshower,origin,pid]
   *
   */
  PyObject* KeypointData::get_keypoint_array( int iclass ) const
  {
    
    if ( !KeypointData::_setup_numpy ) {
      import_array1(0);
      KeypointData::_setup_numpy = true;
    }
    
    // first count the number of unique points
    std::set< std::vector<int> >    unique_coords;
    std::vector< std::vector<int> > kpd_index;
    int npts = 0;
    for ( size_t ikpd=0; ikpd<_kpd_v.size(); ikpd++ ) {
      auto const& kpd = _kpd_v[ikpd];
      if ( kpd.kptype!=(larflow::KeyPoint_t)iclass ) continue;
      if (kpd.imgcoord.size()>0) {
        if ( unique_coords.find( kpd.imgcoord )==unique_coords.end() ) {
          kpd_index.push_back( std::vector<int>{(int)ikpd,0} );
          unique_coords.insert( kpd.imgcoord );
          npts++;
        }
      }      
    }
    
    int nd = 2;
    npy_intp dims[] = { npts, 10 };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );
    
    size_t ipt = 0;
    for ( auto& kpdidx : kpd_index ) {
      
      auto const& kpd = _kpd_v[kpdidx[0]];
      
      if ( kpdidx[1]==0 ) {
        // start img coordinates
        for ( size_t i=0; i<4; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,i)) = (float)kpd.imgcoord[i];
        // 3D point
        for ( size_t i=0; i<3; i++ )
          *((float*)PyArray_GETPTR2(array,ipt,4+i)) = (float)kpd.keypt[i];
        // is shower
        *((float*)PyArray_GETPTR2(array,ipt,7)) = (float)kpd.is_shower;
        // origin
        *((float*)PyArray_GETPTR2(array,ipt,8)) = (float)kpd.origin;
        // PID
        *((float*)PyArray_GETPTR2(array,ipt,9)) = (float)kpd.pid;
        ipt++;
      }
    }// end of loop over keypointdata structs
    
    return (PyObject*)array;
  }


  /**
   * return a numpy array with keypoint class scores
   *
   * The score is calculated for each proposed spacepoint using a gaussian where the mean
   *  is the 3d position of the closest ground truth keypoint for the given class.
   * For any point sig*0.3 cm away from a ground truth keypoint, the score is set to zero.
   *
   * Assumes that `process` has already been run.
   *
   * @param[in] sig The sigma used in Gaussian to calculate keypoint class score
   * @return Numpy array with shape [num space points, 6 classes ]
   *
   */
  PyObject* KeypointData::get_triplet_score_array( float sig ) const
  {

    if ( !KeypointData::_setup_numpy ) {
      import_array1(0);
      KeypointData::_setup_numpy = true;
    }

    // get label info for each triplet proposal
    int npts = (int)_match_proposal_labels_v[0].size();
    for (size_t iclass=0; iclass<6; iclass++) {
      if ( _match_proposal_labels_v[iclass].size()!=npts ) {
        throw std::runtime_error("number of triplet labels/scores for each class does not match!");
      }
    }
    
    int nd = 2;
    npy_intp dims[] = { npts, 6 };
    PyArrayObject* array = (PyArrayObject*)PyArray_SimpleNew( nd, dims, NPY_FLOAT );

    size_t ipt = 0;
    for ( size_t ipt=0; ipt<npts; ipt++ ) {
      for (size_t iclass=0; iclass<6; iclass++) {
        
        auto const& label_v = _match_proposal_labels_v[iclass][ipt];
        
        if ( label_v[0]==0.0 ) {
          *((float*)PyArray_GETPTR2(array,ipt,iclass)) = 0.0;
        }
        else {
          float dist = 0.;
          for (int i=0; i<3; i++) dist += label_v[1+i]*label_v[1+i];
          *((float*)PyArray_GETPTR2(array,ipt,iclass)) = exp( -0.5*dist/(sig*sig) );
        }
      }
    }// end of loop over keypointdata structs
    
    return (PyObject*)array;
  }
  
  
}
}
