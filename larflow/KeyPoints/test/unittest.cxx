#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include  <string>
#include  <vector>

#include "larcv/core/PyUtil/load_pyutil.h"
#include "larflow/KeyPoints/LoaderKeypointData.h"


int main(int nargs, char** argv ) {

  //larcv::load_pyutil();
  Py_Initialize();   
  import_array1(0);

  std::vector<std::string> rootfiles_v = {"test.root"};
  larflow::keypoints::LoaderKeypointData loader(rootfiles_v);
  loader.set_verbosity( larcv::msg::kDEBUG );
  
  int ENTRY=0;
  unsigned long nbytes = loader.load_entry(ENTRY);
  std::cout << "number of bytes read: " << nbytes << std::endl;

  loader.triplet_v->at(0).setShuffleWhenSampling(false);
  PyObject* tripdata = loader.triplet_v->at(0).get_all_triplet_data( true );
  PyObject* spacepoints = loader.triplet_v->at(0).make_spacepoint_charge_array();
  int nfilled = 0;
  int ntriplets = loader.triplet_v->at(0)._triplet_v.size();

  PyObject* data = loader.triplet_v->at(0).make_triplet_ndarray(false);

  PyObject* kpdata = loader.sample_data( ntriplets, nfilled, true );

  Py_Finalize();
  
  return 0;
  
}
