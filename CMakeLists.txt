cmake_minimum_required(VERSION 3.10)
project(larflow VERSION 0.1)

set(CMAKE_BUILD_TYPE Debug)

#---Install location by default is in the build directory
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set (CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "${CMAKE_BUILD_DIR}" FORCE )
endif()

## Dependencies

#---CERN ROOT (required)
find_package(ROOT CONFIG REQUIRED)

#---LArCV (required)
set( larcv_DIR "$ENV{LARCV_LIBDIR}/cmake/larcv")
find_package( larcv REQUIRED )

#---ublarcvapp (required)
set( ublarcvapp_DIR "$ENV{UBLARCVAPP_LIBDIR}/cmake/ublarcvapp")
find_package( ublarcvapp REQUIRED )

#---OPENCV (required)
set(  OpenCV_DIR $ENV{OPENCV_LIBDIR}/../share/OpenCV )
find_package(OpenCV REQUIRED)

#---larbys GEO2D
# use it if env variables defined
if (DEFINED ENV{GEO2D_BASEDIR})
  set(LARCV_USE_GEO2D 1)

  # unless explicitly told no
  if (DEFINED ENV{USE_GEO2D} AND NOT $USE_GEO2D)
    set(LARCV_USE_GEO2D 0)
    message("DISABLING modules that need Geo2D")
  else()
    file( GLOB GEO2D_LIBS "$ENV{GEO2D_LIBDIR}/*.so" )
    set( GEO2D_INC_DIR "$ENV{GEO2D_INCDIR}" )
    set( GEO2D_LIB_DIR "$ENV{GEO2D_LIBDIR}" )
  endif()
else()
  set(LARCV_USE_GEO2D 0)
  message("DISABLING modules needing Geo2D")  
endif()


#---LArLite
set( LARLITE_INC_DIRS "$ENV{LARLITE_COREDIR}" "$ENV{LARLITE_BASEDIR}" "$ENV{LARLITE_USERDEVDIR}" "$ENV{LARLITE_BASEDIR}/../" )
set( LARLITE_LIB_DIR  "$ENV{LARLITE_LIBDIR}" )
set( HAS_LARLITE 1 )
file( GLOB LARLITE_LIBS "$ENV{LARLITE_LIBDIR}/libLArLite*.so" )

#---LArOpenCV
set( LAROPENCV_INC_DIR "$ENV{LAROPENCV_BASEDIR}" )
set( LAROPENCV_LIB_DIR "$ENV{LARLITE_LIBDIR}" )
file( GLOB LAROPENCV_LIBS "$ENV{LARLITE_LIBDIR}/libLArOpenCV_*.so" )

#---Python
find_package( PythonInterp 2.7 REQUIRED )
find_package( PythonLibs 2.7 REQUIRED )
execute_process( COMMAND python -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDES OUTPUT_STRIP_TRAILING_WHITESPACE )

#---Eigen
find_package(Eigen3 REQUIRED)

#---Cilantro
set( CILANTRO_INC_DIR "$ENV{CILANTRO_INC_DIR}" )

#--- glog
#find_package(glog)

## Setup

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

#---Define Variables for subdirectory cmakelists
set(LARFLOW_BASE_DIR  "${CMAKE_CURRENT_SOURCE_DIR}")

## Add Source

#---Add base of source
add_subdirectory(larflow)

#---Configuration export
include(CMakePackageConfigHelpers)
set(LARFLOW_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(LARFLOW_LIBRARY_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(TARGETS_EXPORT_NAME "larflowTargets")
configure_package_config_file("cmake/Config.cmake.in"
  "${CMAKE_BINARY_DIR}/cmake/larflowConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larflow"
  PATH_VARS LARFLOW_INCLUDE_DIR LARFLOW_LIBRARY_DIR LARFLOW_BASE_DIR)
write_basic_package_version_file("${CMAKE_BINARY_DIR}/cmake/larflowConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion)


#---Create info to include larflow in other cmake files
install(EXPORT larflow
        FILE larflowTargets.cmake 
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larflow")

#---Create info to include larflow in other cmake files
install(FILES "${CMAKE_BINARY_DIR}/cmake/larflowConfig.cmake"
  "${CMAKE_BINARY_DIR}/cmake/larflowConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/larflow")







