set(MODULE_NAME PrepFlowMatchData )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

set( HEADERS PrepFlowMatchData.hh )

add_library( ${LIBNAME} SHARED
  PrepFlowMatchData.cxx )

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# DEFINE SOME VARS THAT MIGHT BE USEFUL FOR SUBMODULES

# larcv libs used
set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor)

get_target_property(ALL_HEADERS ${LIBNAME} PUBLIC_HEADER)

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larflow/${MODULE_NAME}>
  ${LARCV_INCLUDE_DIR}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/mrcnnmatch
  ${PROJECT_SOURCE_DIR}
  )
target_link_libraries(${LIBNAME} ${LARCV_LIBS_USED})

include_directories( ${PROJECT_SOURCE_DIR} ${LARCV_INCLUDE_DIR} )
ROOT_GENERATE_DICTIONARY(G__${LIBNAME} ${ALL_HEADERS} LINKDEF LinkDef.h)
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
