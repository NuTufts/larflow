set(MODULE_NAME Reco )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

set( HEADERS cluster_functions.h PCACluster.h PyLArFlow.h ShowerReco.h MCPixelPGraph.h VertexReco.h )

add_library( ${LIBNAME} SHARED cluster_functions.cxx PCACluster.cxx PyLArFlow.cxx ShowerReco.cxx MCPixelPGraph.cxx VertexReco.cxx )

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# DEFINE SOME VARS THAT MIGHT BE USEFUL FOR SUBMODULES
message("python include: ${PYTHON_INCLUDE_PATH}")

# larcv libs used
set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor LArCVApp_dbscan)

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
  )
target_link_libraries(${LIBNAME} ${LARCV_LIBS_USED} ${LARLITE_LIBS} ${PYTHON_LIBRARIES} )

include_directories( ${PROJECT_SOURCE_DIR} ${LARCV_INCLUDE_DIR} ${LARLITE_INC_DIRS} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} ${OpenCV_INCLUDE_DIRS} )
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
