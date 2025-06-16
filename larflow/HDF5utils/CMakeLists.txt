set(MODULE_NAME HDF5utils )

# DEFINE THE TARGET
set( LIBNAME LArFlow_${MODULE_NAME} )

# Find HDF5: we'll need this for this module
find_package(HDF5 REQUIRED COMPONENTS C CXX HL)

# Debug: Print HDF5 information
message(STATUS "HDF5 found: ${HDF5_FOUND}")
message(STATUS "HDF5 include dirs: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")
message(STATUS "HDF5 HL libraries: ${HDF5_HL_LIBRARIES}")
message(STATUS "HDF5 definitions: ${HDF5_DEFINITIONS}")

# Source files
set(SOURCES
    NuVertexCandidateHDF5Reader.cxx
)

# Header files  
set(HEADERS
    NuVertexCandidateHDF5Reader.h
)

# Create library
add_library( ${LIBNAME} SHARED ${SOURCES})

# create a property to hold our headers
set_target_properties(${LIBNAME} PROPERTIES PUBLIC_HEADER "${HEADERS}")
get_target_property(ALL_HEADERS ${LIBNAME} PUBLIC_HEADER)

# Include directories
target_include_directories( ${LIBNAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/larflow/${MODULE_NAME}>
    ${LARLITE_INC_DIRS}
    ${LARCV_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    PRIVATE
    ${PROJECT_SOURCE_DIR} )

# Link libraries
target_link_libraries( ${LIBNAME} PUBLIC
    larlite_DataFormat
    ${LARCV_LIBRARIES}
    ${ROOT_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    LArFlow_Reco )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}
    ${LARLITE_INC_DIRS}
    ${LARCV_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS} )
ROOT_GENERATE_DICTIONARY(G__${LIBNAME} ${ALL_HEADERS} LINKDEF LinkDef.h)
target_sources( ${LIBNAME} PRIVATE G__${LIBNAME}.cxx )
  

# Set properties
set_target_properties(${LIBNAME} PROPERTIES
    CXX_STANDARD ${CMAKE_CXX_STANDARD}
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# Install library
install(TARGETS ${LIBNAME}
    EXPORT larflow
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/larflow/${MODULE_NAME})

# install files needed for dictionary use in CINT/CLING/PYTHON
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}_rdict.pcm
  ${CMAKE_CURRENT_BINARY_DIR}/lib${LIBNAME}.rootmap
  DESTINATION lib  )

# Build test executable if requested
option(BUILD_HDF5_TESTS "Build HDF5 test programs" ON)

if(BUILD_HDF5_TESTS)
    add_executable(test_nuvertex_hdf5_reader test_nuvertex_hdf5_reader.cxx)
    
    target_link_libraries(test_nuvertex_hdf5_reader
        LArFlow_HDF5utils
        LArFlow_Reco
        ${LARLITE_LIBS}
        ${larcv_LIBRARIES}
        ${ROOT_LIBRARIES}
    )
    
    # HDF5 is already linked through LArFlow_HDF5utils
    target_include_directories(test_nuvertex_hdf5_reader PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${LARLITE_INC_DIRS}
        ${LARCV_INCLUDE_DIR}
    )
    
    set_target_properties(test_nuvertex_hdf5_reader PROPERTIES
        CXX_STANDARD ${CMAKE_CXX_STANDARD}
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    
    # Optionally install test executable
    install(TARGETS test_nuvertex_hdf5_reader
        RUNTIME DESTINATION bin
        OPTIONAL
    )
endif()


# # install libraries and headers
# install(TARGETS ${LIBNAME}
#   EXPORT larflow
#   LIBRARY DESTINATION lib
#   PUBLIC_HEADER DESTINATION include/larflow/${MODULE_NAME})


