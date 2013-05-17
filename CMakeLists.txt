################################################################################
# Fast user build settings

SET(IS_ARM TRUE)
SET(BUILD_FAST_LIBRARY TRUE)

################################################################################

SET(PROJECT_NAME fast)

PROJECT(${PROJECT_NAME})
CMAKE_MINIMUM_REQUIRED (VERSION 2.8)

SET (CMAKE_VERBOSE_MAKEFILE ON)

IF( NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE Release ) 
ENDIF()

SET(CMAKE_CXX_FLAGS "-Wall -Werror -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unknown-pragmas")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g")

IF(IS_ARM)
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -mfpu=neon -march=armv7-a")
ELSE()
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -mmmx -msse -msse -msse2 -msse3 -mssse3 -fomit-frame-pointer")
ENDIF()

# set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
 
# FAST library
SET(SOURCES  ${PROJECT_SOURCE_DIR}/src/fast_9.cpp
             ${PROJECT_SOURCE_DIR}/src/fast_10.cpp
             ${PROJECT_SOURCE_DIR}/src/fast_10_score.cpp
             ${PROJECT_SOURCE_DIR}/src/nonmax_3x3.cpp
             #${PROJECT_SOURCE_DIR}/src/nonmax_5x5.cpp
           )
IF(IS_ARM)
  LIST(APPEND SOURCES ${PROJECT_SOURCE_DIR}/src/faster_corner_9_neon.cpp)
ELSE()
  LIST(APPEND SOURCES ${PROJECT_SOURCE_DIR}/src/faster_corner_10_sse.cpp)
ENDIF()


ADD_LIBRARY( fast SHARED ${SOURCES} )
INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/include)

# Build a test program for the fast detector
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( test_fast ${PROJECT_SOURCE_DIR}/test/test.cpp )
TARGET_LINK_LIBRARIES( test_fast fast ${OpenCV_LIBS} )
