set(MODULE_NAME Reco )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

set( HEADERS cluster_functions.h
  geofuncs.h
  SplitHitsBySSNet.h
  SplitHitsByParticleSSNet.h
  ChooseMaxLArFlowHit.h
  DBScanLArMatchHits.h
  ProjectionDefectSplitter.h
  ShortProtonClusterReco.h
  PyLArFlow.h
  KPCluster.h KeypointReco.h EventKeypointReco.h
  KeypointFilterByClusterSize.h
  KeypointFilterByWCTagger.h
  TrackClusterBuilder.h
  CosmicTrackBuilder.h
  ShowerRecoKeypoint.h
  ShowerBilineardEdx.h
  ShowerdQdx.h
  ShowerTruthMetricsMaker.h
  NuVertexCandidate.h
  TrackOTFit.h
  ProjPixFitter.h
  ClusterBookKeeper.h
  NuVertexMaker.h
  NuVertexFitter.h
  CosmicVertexBuilder.h  
  NuTrackBuilder.h
  NuShowerBuilder.h
  NuVertexShowerReco.h
  NuVertexShowerTrunkCheck.h
  NuVertexActivityReco.h
  NuVertexAddSecondaries.h
  VetoHitClustering.h
  PerfectTruthNuReco.h
  TrackdQdx.h
  NuTrackdQdx.h
  ClusterImageMask.h
  NuSelectionVariables.h
  LikelihoodProtonMuon.h
  CosmicProtonFinder.h
  TrackForwardBackwardLL.h
  TrackFindBadConnection.h  
  NuTrackKinematics.h
  NuShowerKinematics.h
  PostNuCheckShowerTrunkOverlap.h
  NuSelProngVars.h
  NuSelVertexVars.h
  NuSelWCTaggerOverlap.h  
  NuSelShowerTrunkAna.h
  NuSelTruthOnNuPixel.h
  NuSelShowerGapAna2D.h
  NuSelUnrecoCharge.h
  NuSelCosmicTagger.h
  NuSel1e1pEventSelection.h
  TruthRecoMatching.h
  KPSRecoManager.h )

add_library( ${LIBNAME} SHARED cluster_functions.cxx
  geofuncs.cxx
  SplitHitsBySSNet.cxx
  SplitHitsByParticleSSNet.cxx
  ChooseMaxLArFlowHit.cxx  
  DBScanLArMatchHits.cxx
  ProjectionDefectSplitter.cxx
  ShortProtonClusterReco.cxx  
  PyLArFlow.cxx
  KPCluster.cxx
  KeypointReco.cxx EventKeypointReco.cxx
  KeypointFilterByClusterSize.cxx
  KeypointFilterByWCTagger.cxx
  ShowerRecoKeypoint.cxx
  ShowerBilineardEdx.cxx
  ShowerdQdx.cxx
  ShowerTruthMetricsMaker.cxx  
  TrackClusterBuilder.cxx
  CosmicTrackBuilder.cxx
  NuVertexCandidate.cxx
  ProjPixFitter.cxx      
  TrackOTFit.cxx
  ClusterBookKeeper.cxx
  NuVertexMaker.cxx
  NuVertexFitter.cxx
  CosmicVertexBuilder.cxx
  NuTrackBuilder.cxx
  NuShowerBuilder.cxx
  NuVertexShowerReco.cxx
  NuVertexShowerTrunkCheck.cxx  
  NuVertexActivityReco.cxx
  NuVertexAddSecondaries.cxx  
  VetoHitClustering.cxx  
  PostNuCheckShowerTrunkOverlap.cxx  
  PerfectTruthNuReco.cxx  
  TrackdQdx.cxx
  NuTrackdQdx.cxx  
  ClusterImageMask.cxx  
  NuSelectionVariables.cxx
  LikelihoodProtonMuon.cxx
  CosmicProtonFinder.cxx
  TrackForwardBackwardLL.cxx
  TrackFindBadConnection.cxx  
  NuTrackKinematics.cxx
  NuShowerKinematics.cxx  
  NuSelProngVars.cxx
  NuSelVertexVars.cxx
  NuSelShowerTrunkAna.cxx
  NuSelWCTaggerOverlap.cxx
  NuSelTruthOnNuPixel.cxx
  NuSelShowerGapAna2D.cxx
  NuSelUnrecoCharge.cxx
  NuSelCosmicTagger.cxx  
  NuSel1e1pEventSelection.cxx
  TruthRecoMatching.cxx
  KPSRecoManager.cxx )

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# DEFINE SOME VARS THAT MIGHT BE USEFUL FOR SUBMODULES
#message("python include: ${PYTHON_INCLUDE_PATH}")
get_target_property(xgboost_include_dirs xgboost::xgboost INTERFACE_INCLUDE_DIRECTORIES)
#message("xgboost include: ${xgboost_include_dirs}")

set( PUB_INC_DIRS  ${LARCV_INCLUDE_DIR}
  ${UBLARCVAPP_INCLUDE_DIR}
  ${LARLITE_INC_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${NUMPY_INCLUDES} )
set( PRI_INC_DIRS
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${CILANTRO_INC_DIR}
  ${OpenCV_INCLUDE_DIRS} )
set( PRIVATE_LIBS_USED "" )
if(xgboost_FOUND)
  set( PRI_INC_DIRS ${PRI_INC_DIRS} ${xgboost_include_dirs} )
  set( PRIVATE_LIBS_USED xgboost::xgboost )
endif()

# larcv libs used
set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor LArCVApp_dbscan LArCVApp_MCTools )

# ublarcvapp libs used
set(UBLARCVAPP_LIBS_USED LArCVApp_UBImageMod)

get_target_property(ALL_HEADERS ${LIBNAME} PUBLIC_HEADER)

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larflow/${MODULE_NAME}>
  ${PUB_INC_DIRS}
  PRIVATE
  ${PRI_INC_DIRS} )
target_link_libraries(${LIBNAME} PUBLIC
  ${UBLARCVAPP_LIBS_USED}
  ${LARCV_LIBS_USED}
  ${LARLITE_LIBS}
  ${PYTHON_LIBRARIES}
  PRIVATE
  ${PRIVATE_LIBS_USED} )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${UBLARCVAPP_INCLUDE_DIR} ${LARCV_INCLUDE_DIR} ${LARLITE_INC_DIRS} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} ${OpenCV_INCLUDE_DIRS} ${xgboost_include_dirs} )
ROOT_GENERATE_DICTIONARY(G__${LIBNAME} ${ALL_HEADERS} ${HEADERS} LINKDEF LinkDef.h)
target_sources( ${LIBNAME} PRIVATE G__${LIBNAME}.cxx )

# install libraries and headers
install(TARGETS ${LIBNAME}
  EXPORT larflow
  LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include/larflow/${MODULE_NAME})

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}.rootmap
  DESTINATION lib  )

add_subdirectory( bin )
