//
// cint script to generate libraries
// Declaire namespace & classes you defined
// #pragma statement: order matters! Google it ;)
//

#ifdef __CINT__
#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all namespaces;

//#pragma link C++ namespace larflow;
#pragma link C++ namespace larflow::reco;
#pragma link C++ function larflow::reco::pointLineDistance3f+;
#pragma link C++ function larflow::reco::pointLineDistance3d+;
#pragma link C++ function larflow::reco::pointRayProjection3f+;
#pragma link C++ function larflow::reco::pointLineDistance3d+;
#pragma link C++ class larflow::reco::ClusterFunctions+;
#pragma link C++ class larflow::reco::cluster_t+;
#pragma link C++ function larflow::reco::cluster_larflow3dhits+;
#pragma link C++ function larflow::reco::cluster_spacepoint_v+;
#pragma link C++ function larflow::reco::cluster_sdbscan_larflow3dhits+;
#pragma link C++ function larflow::reco::cluster_dbscan_vp_larflow3dhits+;
#pragma link C++ function larflow::reco::cluster_dump2jsonfile+;
#pragma link C++ function larflow::reco::cluster_splitbytrackshower+;
#pragma link C++ function larflow::reco::cluster_imageprojection+;
#pragma link C++ function larflow::reco::cluster_getcontours+;
#pragma link C++ class larflow::reco::PCACluster+;
#pragma link C++ class larflow::reco::ProjectionDefectSplitter+;
#pragma link C++ class larflow::reco::PyLArFlow+;
#pragma link C++ class larflow::reco::ShowerReco+;
#pragma link C++ class larflow::reco::VertexReco+;
#pragma link C++ class std::vector<larflow::reco::VertexReco::Candidate_t>+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wcharge+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wssnet+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wprob+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wdeadch+;
#pragma link C++ class larflow::reco::KPCluster+;
#pragma link C++ class larflow::reco::NuVertexCandidate+;
#pragma link C++ class larflow::reco::SplitHitsBySSNet+;
#pragma link C++ class larflow::reco::ChooseMaxLArFlowHit+;
#pragma link C++ class larflow::reco::DBScanLArMatchHits+;
#pragma link C++ class larflow::reco::KeypointReco+;
#pragma link C++ class larflow::reco::TrackReco2KP+;
#pragma link C++ class larflow::reco::TrackClusterBuilder+;
#pragma link C++ class larflow::reco::CosmicTrackBuilder+;
#pragma link C++ class larflow::reco::NuVertexMaker+;
#pragma link C++ class larflow::reco::KPSRecoManager+;
//ADD_NEW_CLASS ... do not change this line
#endif
