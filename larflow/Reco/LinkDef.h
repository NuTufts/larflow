/** @defgroup Reco Reco
 *
 * @brief Algorithms that use Network outputs to reconstruct the interactions in the TPC
 *
 *
 * cint script to generate libraries and python bindings.
 * Declare namespace & classes you defined
 * pragma statement: order matters! Google it ;)
 *
 */

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
#pragma link C++ function larflow::reco::cluster_sdbscan_spacepoints+;
#pragma link C++ function larflow::reco::cluster_dbscan_vp_larflow3dhits+;
#pragma link C++ function larflow::reco::cluster_dump2jsonfile+;
#pragma link C++ function larflow::reco::cluster_splitbytrackshower+;
#pragma link C++ function larflow::reco::cluster_imageprojection+;
#pragma link C++ function larflow::reco::cluster_getcontours+;
#pragma link C++ class larflow::reco::ProjectionDefectSplitter+;
#pragma link C++ class larflow::reco::PyLArFlow+;
#pragma link C++ class larflow::reco::ShowerReco+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wcharge+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wssnet+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wprob+;
#pragma link C++ function larflow::reco::as_ndarray_larflowcluster_wdeadch+;
#pragma link C++ class larflow::reco::KPCluster+;
#pragma link C++ class larflow::reco::NuVertexCandidate+;
#pragma link C++ class larflow::reco::SplitHitsBySSNet+;
#pragma link C++ class larflow::reco::SplitHitsByParticleSSNet+;
#pragma link C++ class larflow::reco::ChooseMaxLArFlowHit+;
#pragma link C++ class larflow::reco::DBScanLArMatchHits+;
#pragma link C++ class larflow::reco::KeypointReco+;
#pragma link C++ class larflow::reco::TrackClusterBuilder+;
#pragma link C++ class larflow::reco::CosmicTrackBuilder+;
#pragma link C++ class larflow::reco::TrackOTFit+;
#pragma link C++ class larflow::reco::NuVertexMaker+;
#pragma link C++ class larflow::reco::NuVertexActivityReco+;
#pragma link C++ class larflow::reco::NuTrackBuilder+;
#pragma link C++ class larflow::reco::VertexTrackTruthRecoInfo+;
#pragma link C++ class larflow::reco::TrackdQdx+;
#pragma link C++ class larflow::reco::NuSelectionVariables+;
#pragma link C++ class std::vector<larflow::reco::NuSelectionVariables>+;
#pragma link C++ class larflow::reco::LikelihoodProtonMuon+;
#pragma link C++ class larflow::reco::ShowerBilineardEdx+;
#pragma link C++ class larflow::reco::ShowerdQdx+;
#pragma link C++ class larflow::reco::CosmicProtonFinder+;
#pragma link C++ class larflow::reco::PerfectTruthNuReco+;
#pragma link C++ class larflow::reco::TrackForwardBackwardLL+;
#pragma link C++ class larflow::reco::NuTrackKinematics+;
#pragma link C++ class larflow::reco::NuShowerKinematics+;
#pragma link C++ class larflow::reco::NuSelProngVars+;
#pragma link C++ class larflow::reco::NuSelVertexVars+;
#pragma link C++ class std::vector<larlite::track>+;
#pragma link C++ class larflow::reco::NuSelShowerTrunkAna+;
#pragma link C++ class larflow::reco::NuSelTruthOnNuPixel+;
#pragma link C++ class larflow::reco::NuSelUnrecoCharge+;
#pragma link C++ class larflow::reco::NuSel1e1pEventSelection+;
#pragma link C++ class larflow::reco::NuSelCosmicTagger+;
#pragma link C++ class larflow::reco::KPSRecoManager+;
//ADD_NEW_CLASS ... do not change this line
#endif
