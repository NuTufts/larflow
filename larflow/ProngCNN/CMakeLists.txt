set(MODULE_NAME ProngCNN )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

set( HEADERS 
  ProngCNNInterface.h
)

add_library( ${LIBNAME} SHARED
  ProngCNNInterface.cxx
)

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
get_target_property(ALL_HEADERS ${LIBNAME} PUBLIC_HEADER)

target_include_directories(${LIBNAME}
  PUBLIC
  $<INSTALL_INTERFACE:include/larflow/${MODULE_NAME}>
  ${LARCV_INCLUDE_DIR}
  ${LARLITE_INC_DIRS}
  ${TORCH_INCLUDE_DIRS}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
)
target_link_libraries(${LIBNAME}
PUBLIC
  LArCVCoreBase
  LArCVCoreDataFormat
  larlite_DataFormat
  LArPID_Data
  LArPID_Model
  LArPID_Interface
  ${TORCH_LIBRARIES}
  ${ROOT_LIBRARIES}
)

message( STATUS "larpid libraries: ${larpid_LIBRARIES}")
message( STATUS "root: ${ROOT_LIBRARIES}")

# Add larpid include directory explicitly
if(DEFINED ENV{LARPID_DIR})
  set(LARPID_INCLUDE_DIR "$ENV{LARPID_DIR}/include")
  message( STATUS "larpid include dir: ${LARPID_INCLUDE_DIR}")
  target_include_directories(${LIBNAME} PUBLIC ${LARPID_INCLUDE_DIR})
endif()
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

# add test subdirectory
add_subdirectory(test)
