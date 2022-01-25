cmake_minimum_required(VERSION 2.8)
project(cilantro)

# Build setup

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-O3 -march=native -mtune=native")

# Packages

find_package(Eigen3 REQUIRED)

#find_package(Pangolin)
#if (Pangolin_FOUND)
#  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAS_PANGOLIN")
#  set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -DHAS_PANGOLIN")
#endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(OTHER_INCLUDES ${EIGEN3_INCLUDE_DIRS})
if (Pangolin_FOUND)
  set(OTHER_INCLUDES "${OTHER_INCLUDES} ${Pangolin_INCLUDE_DIRS}")
endif()

include_directories(include)
include_directories(${OTHER_INCLUDES})

# Build library

file(GLOB_RECURSE 3rd_src ${CMAKE_SOURCE_DIR}/src/3rd_party/*.c ${CMAKE_SOURCE_DIR}/src/3rd_party/*.cpp)
file(GLOB lib_include ${CMAKE_SOURCE_DIR}/include/cilantro/*.hpp)
file(GLOB lib_src ${CMAKE_SOURCE_DIR}/src/*.cpp)
if (NOT Pangolin_FOUND)
  list(REMOVE_ITEM lib_src     ${CMAKE_SOURCE_DIR}/src/common_renderables.cpp)
  list(REMOVE_ITEM lib_src     ${CMAKE_SOURCE_DIR}/src/renderable.cpp)
  list(REMOVE_ITEM lib_src     ${CMAKE_SOURCE_DIR}/src/visualizer.cpp)
  list(REMOVE_ITEM lib_src     ${CMAKE_SOURCE_DIR}/src/visualizer_handler.cpp)  
  list(REMOVE_ITEM lib_src     ${CMAKE_SOURCE_DIR}/src/image_viewer.cpp)
  
  list(REMOVE_ITEM lib_include ${CMAKE_SOURCE_DIR}/include/cilantro/image_viewer.h)
  list(REMOVE_ITEM lib_include ${CMAKE_SOURCE_DIR}/include/cilantro/visualizer.h)  
  list(REMOVE_ITEM lib_include ${CMAKE_SOURCE_DIR}/include/cilantro/visualizer_handle.h)
  list(REMOVE_ITEM lib_include ${CMAKE_SOURCE_DIR}/include/cilantro/common_renderables.h)
  list(REMOVE_ITEM lib_include ${CMAKE_SOURCE_DIR}/include/cilantro/renderable.h)
endif()

add_library(${PROJECT_NAME} SHARED ${3rd_src} ${lib_include} ${lib_src})
if (Pangolin_FOUND)
  target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARIES})
endif()

# Configure Doxyfile
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

# Build examples

option(BUILD_EXAMPLES "Build small example apps" ON)
if(BUILD_EXAMPLES)
  #file(GLOB example_files RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/examples/*.cpp)
  file(GLOB example_files RELATIVE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/examples/flat_convex_hull.cpp)
  if (NOT Pangolin_FOUND)
    list(REMOVE_ITEM example_files examples/visualizer.cpp)
  endif()
  foreach(example_file ${example_files})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    if (Pangolin_FOUND)
      target_link_libraries(${example_name} ${PROJECT_NAME} ${Pangolin_LIBRARIES})
    else()
      target_link_libraries(${example_name} ${PROJECT_NAME})
    endif()
  endforeach(example_file ${example_files})
endif()

# Package config files

file(REMOVE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")
export(TARGETS ${PROJECT_NAME} APPEND FILE "${CMAKE_BINARY_DIR}/${PROJECT_NAME}Targets.cmake")

set(CMAKECONFIG_INSTALL_DIR lib/cmake/${PROJECT_NAME})
file(RELATIVE_PATH REL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKECONFIG_INSTALL_DIR}" "${CMAKE_INSTALL_PREFIX}/include")

# Build tree config
set(EXPORT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
configure_file(${CMAKE_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in ${CMAKE_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE)

# Install tree config
set(EXPORT_INCLUDE_DIR "\${PROJECT_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(${CMAKE_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake @ONLY)

# Add package to CMake package registry for use from the build tree
option(EXPORT_${PROJECT_NAME} "Export ${PROJECT_NAME} package for use by other software" ON)
if(EXPORT_${PROJECT_NAME})
    export(PACKAGE ${PROJECT_NAME})
endif()

# Install target

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

install(FILES "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${PROJECT_NAME}Config.cmake" DESTINATION ${CMAKECONFIG_INSTALL_DIR})
install(EXPORT ${PROJECT_NAME}Targets DESTINATION ${CMAKECONFIG_INSTALL_DIR})

# Uninstall target

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in" "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" IMMEDIATE @ONLY)
add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
