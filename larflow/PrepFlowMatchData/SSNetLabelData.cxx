#include "SSNetLabelData.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "PrepSSNetTriplet.h"


namespace larflow {
namespace prep {

  /**
   * static variable to track if numpy environment has been setup
   */
  bool SSNetLabelData::_setup_numpy = false;  
  
  SSNetLabelData::SSNetLabelData()
  {
  }
  
  /**
   * @brief make the ssnet numpy arrays 
   *
   * @param[in]  num_max_samples Max number of samples to return
   * @param[out] nfilled number of samples actually returned
   * @param[in]  withtruth if true, return flag indicating if true/good space point
   * @param[out] pos_match_index vector index in return samples for space points which are true/good
   * @param[in]  match_array numpy array containing indices to sparse image for each spacepoint
   * @param[out] ssnet_label numpy array containing ssnet class labels for each spacepoint
   * @param[out] ssnet_top_weight numpy array containing topological weight
   * @param[out] ssnet_class_weight numpy array containing class weights
   * @return always returns 0
   *
   */
  PyObject* SSNetLabelData::make_ssnet_arrays()  
  {

    if ( !SSNetLabelData::_setup_numpy ) {
      import_array1(0);
      SSNetLabelData::_setup_numpy = true;
    }    

    // make ssnet label array
    int nsamples = _ssnet_label_v.size();
    int ssnet_label_nd = 1;
    npy_intp ssnet_label_dims1[] = { (long)nsamples };
    npy_intp ssnet_label_dims2[] = { (long)nsamples };
    npy_intp ssnet_label_dims3[] = { (long)nsamples };


    PyArrayObject* ssnet_label        = (PyArrayObject*)PyArray_SimpleNew( ssnet_label_nd, ssnet_label_dims1, NPY_LONG );
    PyArrayObject* ssnet_top_weight   = (PyArrayObject*)PyArray_SimpleNew( ssnet_label_nd, ssnet_label_dims2, NPY_FLOAT );
    PyArrayObject* ssnet_class_weight = (PyArrayObject*)PyArray_SimpleNew( ssnet_label_nd, ssnet_label_dims3, NPY_FLOAT );

    PyObject *ssnet_label_t_key       = Py_BuildValue("s", "ssnet_label_t");
    PyObject *ssnet_topweight_t_key   = Py_BuildValue("s", "ssnet_topweight_t");
    PyObject *ssnet_classweight_t_key = Py_BuildValue("s", "ssnet_classweight_t");

    std::vector<int> nclass( larflow::prep::PrepSSNetTriplet::kNumClasses, 0 );
    //LARCV_DEBUG() << "make class labels and topological weight arrays. nelems=" << ssnet_label_dims1[0] << std::endl;
    
    int nbad_labels = 0;
    for ( int i=0; i<(int)ssnet_label_dims1[0]; i++ ) {

      
      int label = _ssnet_label_v.at( i );
      
      if (label<0 || label>=larflow::prep::PrepSSNetTriplet::kNumClasses) {
	std::stringstream msg;
	msg << "invalid class label=" << label << " from the Tree" << std::endl;
	//throw std::runtime_error( msg.str() );
	label = 0;
	nbad_labels++;
	//std::cout << msg.str() << std::endl;
      }
      nclass[label]++;
      
      *((long*)PyArray_GETPTR1(ssnet_label,i))       = (long)label; // class label
      
      float topo_weight = 1.0;
      for ( auto const& w : _boundary_weight_v[i] )
	topo_weight *= w;
      
      *((float*)PyArray_GETPTR1(ssnet_top_weight,i)) = (float)topo_weight;
    }
    
    //LARCV_DEBUG() << "make class balancing weights" << std::endl;
    
    // calculate class-balancing weights
    int ntot = (int)ssnet_label_dims1[0];
    std::vector<float> w_class( larflow::prep::PrepSSNetTriplet::kNumClasses, 0.0 );
    float w_norm  = 0.;
    for (int i=0; i<(int)nclass.size(); i++) {
      if ( nclass[i]>0 )
	w_class[i] = 1.0/float(nclass[i]);
      else
	w_class[i] = 0.0;
      w_norm += w_class[i];
    }
    if ( w_norm>0 ) {
      for (int i=0; i<(int)nclass.size(); i++)
	w_class[i] /= w_norm;
    }
    
    for ( int i=0; i<(int)ssnet_label_dims1[0]; i++ ) {
      long label = *((long*)PyArray_GETPTR1(ssnet_label,i));
      if (label<0 || label>=larflow::prep::PrepSSNetTriplet::kNumClasses) {
	std::stringstream msg;
	msg << "invalid class label=" << label << " from the Tree" << std::endl;
	throw std::runtime_error( msg.str() );
      }      
      *((float*)PyArray_GETPTR1(ssnet_class_weight,i)) = w_class[label];
    }
    
    //LARCV_DEBUG() << "Num bad labels: " << nbad_labels << std::endl;
    
    // Create and fill dictionary
    PyObject *d = PyDict_New();
    PyDict_SetItem(d, ssnet_label_t_key,       (PyObject*)ssnet_label);
    PyDict_SetItem(d, ssnet_topweight_t_key,   (PyObject*)ssnet_top_weight);
    PyDict_SetItem(d, ssnet_classweight_t_key, (PyObject*)ssnet_class_weight);        
    Py_DECREF( ssnet_label_t_key );
    Py_DECREF( ssnet_topweight_t_key );
    Py_DECREF( ssnet_classweight_t_key );    
    
    
    return d;
  }
  
  
}
}
