set(MODULE_NAME KeyPoints )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

set( HEADERS KPdata.h PrepKeypointData.h )

add_library( ${LIBNAME} SHARED
  KPdata.cxx
  PrepKeypointData.cxx )

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")

# DEFINE SOME VARS THAT MIGHT BE USEFUL FOR SUBMODULES
message("python include: ${PYTHON_INCLUDE_PATH}")

# larcv libs used
set(LARCV_LIBS_USED LArCVCoreBase LArCVCoreDataFormat LArCVCoreProcessor)

# ublarcvapp libs used
set(UBLARCVAPP_LIBS_USED LArCVApp_UBImageMod LArCVApp_MCTools )

get_target_property(ALL_HEADERS ${LIBNAME} PUBLIC_HEADER)

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larflow/${MODULE_NAME}>
  ${LARCV_INCLUDE_DIR}
  ${LARLITE_INC_DIRS}
  ${UBLARCVAPP_INCLUDE_DIR}
  ${PYTHON_INCLUDE_PATH}
  ${NUMPY_INCLUDES}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/mrcnnmatch
  ${PROJECT_SOURCE_DIR}
  )
target_link_libraries(${LIBNAME} ${UBLARCVAPP_LIBS_USED} ${LARCV_LIBS_USED} ${LARLITE_LIBS} ${PYTHON_LIBRARIES} )

include_directories( ${PROJECT_SOURCE_DIR} ${LARCV_INCLUDE_DIR} ${LARLITE_INC_DIRS} ${UBLARCVAPP_INCLUDE_DIR} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDES} )
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
