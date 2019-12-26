//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::reco;
#pragma link C++ class larflow::reco::cluster_t+;
#pragma link C++ function larflow::reco::cluster_larflow3dhits+;
#pragma link C++ function larflow::reco::cluster_dump2jsonfile+;
#pragma link C++ function larflow::reco::cluster_splitbytrackshower+;
#pragma link C++ function larflow::reco::cluster_imageprojection+;
#pragma link C++ function larflow::reco::cluster_getcontours+;
#pragma link C++ class larflow::reco::PCACluster+;
#pragma link C++ class larflow::reco::CRTMatch+;
#pragma link C++ class larflow::reco::PyLArFlow+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wcharge+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wssnet+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wprob+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wdeadch+;
//ADD_NEW_CLASS ... do not change this line
#endif
