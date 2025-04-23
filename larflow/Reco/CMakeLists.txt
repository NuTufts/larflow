set(MODULE_NAME Reco )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

set( HEADERS
  KPCluster.h 
  NuSelectionVariables.h
  NuVertexCandidate.h
  # SplitHitsBySSNet.h
  # SplitHitsByParticleSSNet.h
  # ChooseMaxLArFlowHit.h
  # DBScanLArMatchHits.h
  # ProjectionDefectSplitter.h
  # ShortProtonClusterReco.h
  # PyLArFlow.h
  # KeypointReco.h 
  # EventKeypointReco.h
  # KeypointFilterByClusterSize.h
  # KeypointFilterByWCTagger.h
  # TrackClusterBuilder.h
  # CosmicTrackBuilder.h
  # ShowerRecoKeypoint.h
  # ShowerBilineardEdx.h
  # ShowerdQdx.h  
  # TrackOTFit.h
  # ProjPixFitter.h
  # ClusterBookKeeper.h
  # NuVertexMaker.h
  # NuVertexFitter.h
  # CosmicVertexBuilder.h  
  # NuTrackBuilder.h
  # NuShowerBuilder.h
  # NuVertexShowerReco.h
  # NuVertexShowerTrunkCheck.h
  # NuVertexActivityReco.h
  # NuVertexAddSecondaries.h
  # VetoHitClustering.h
  # PerfectTruthNuReco.h
  # TrackdQdx.h
  # NuTrackdQdx.h
  # ClusterImageMask.h
  # LikelihoodProtonMuon.h
  # CosmicProtonFinder.h
  # TrackForwardBackwardLL.h
  # TrackFindBadConnection.h  
  # NuTrackKinematics.h
  # NuShowerKinematics.h
  # PostNuCheckShowerTrunkOverlap.h
  # NuSelProngVars.h
  # NuSelVertexVars.h
  # NuSelWCTaggerOverlap.h  
  # NuSelShowerTrunkAna.h
  # NuSelTruthOnNuPixel.h
  # NuSelShowerGapAna2D.h
  # NuSelUnrecoCharge.h
  # NuSelCosmicTagger.h
  # NuSel1e1pEventSelection.h
  # TruthRecoMatching.h
  # KPSRecoManager.h 
)

add_library( ${LIBNAME} SHARED
  KPCluster.cxx
  NuSelectionVariables.cxx
  NuVertexCandidate.cxx
  # SplitHitsBySSNet.cxx
  # SplitHitsByParticleSSNet.cxx
  # ChooseMaxLArFlowHit.cxx  
  # DBScanLArMatchHits.cxx
  # ProjectionDefectSplitter.cxx
  # ShortProtonClusterReco.cxx  
  # PyLArFlow.cxx
  # KeypointReco.cxx 
  # EventKeypointReco.cxx
  # KeypointFilterByClusterSize.cxx
  # KeypointFilterByWCTagger.cxx
  # ShowerRecoKeypoint.cxx
  # ShowerBilineardEdx.cxx
  # ShowerdQdx.cxx  
  # TrackClusterBuilder.cxx
  # CosmicTrackBuilder.cxx
  # ProjPixFitter.cxx      
  # TrackOTFit.cxx
  # ClusterBookKeeper.cxx
  # NuVertexMaker.cxx
  # NuVertexFitter.cxx
  # CosmicVertexBuilder.cxx
  # NuTrackBuilder.cxx
  # NuShowerBuilder.cxx
  # NuVertexShowerReco.cxx
  # NuVertexShowerTrunkCheck.cxx  
  # NuVertexActivityReco.cxx
  # NuVertexAddSecondaries.cxx  
  # VetoHitClustering.cxx  
  # PostNuCheckShowerTrunkOverlap.cxx  
  # PerfectTruthNuReco.cxx  
  # TrackdQdx.cxx
  # NuTrackdQdx.cxx  
  # ClusterImageMask.cxx  
  # LikelihoodProtonMuon.cxx
  # CosmicProtonFinder.cxx
  # TrackForwardBackwardLL.cxx
  # TrackFindBadConnection.cxx  
  # NuTrackKinematics.cxx
  # NuShowerKinematics.cxx  
  # NuSelProngVars.cxx
  # NuSelVertexVars.cxx
  # NuSelShowerTrunkAna.cxx
  # NuSelWCTaggerOverlap.cxx
  # NuSelTruthOnNuPixel.cxx
  # NuSelShowerGapAna2D.cxx
  # NuSelUnrecoCharge.cxx
  # NuSelCosmicTagger.cxx  
  # NuSel1e1pEventSelection.cxx
  # TruthRecoMatching.cxx
  # KPSRecoManager.cxx 
)

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# DEFINE SOME VARS THAT MIGHT BE USEFUL FOR SUBMODULES
message("python include: ${PYTHON_INCLUDE_PATH}")

set( LARLITE_LIBS_USED larlite_DataFormat )
message( STATUS "larlite libs: ${LARLITE_LIBS_USED}" )

# larcv libs used
set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor LArCVApp_dbscan LArCVApp_MCTools )

# ublarcvapp libs used
set(UBLARCVAPP_LIBS_USED LArCVApp_UBImageMod)

get_target_property(ALL_HEADERS ${LIBNAME} PUBLIC_HEADER)

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larflow/${MODULE_NAME}>
  ${LARCV_INCLUDE_DIR}
  ${UBLARCVAPP_INCLUDE_DIR}
  ${LARLITE_INC_DIRS}
  ${PYTHON_INCLUDE_PATH}
  ${NUMPY_INCLUDES}
  PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${CILANTRO_INC_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${LARFLOW_JSON_INCLUDE_DIR}
)
target_link_libraries(${LIBNAME} PUBLIC
  larlite_DataFormat
  #${PYTHON_LIBRARIES}
  #${LARCV_LIBS_USED}  
  #${UBLARCVAPP_LIBS_USED}
  #LArFlow_RecoUtils
  #LArFlow_SCBoundary
)

include_directories( ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${UBLARCVAPP_INCLUDE_DIR} ${LARCV_INCLUDE_DIR} ${LARLITE_INC_DIRS} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} ${OpenCV_INCLUDE_DIRS} )
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
