cmake_minimum_required(VERSION 3.16)
project(calltx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)

# ---- whisper.cpp (speech recognition engine, pure C/C++) ----
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
# GPU: configure with -DGGML_CUDA=ON (NVIDIA), -DGGML_VULKAN=ON, or Metal is on by default on macOS.

FetchContent_Declare(
  whisper
  GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git
  GIT_TAG        v1.7.5
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(whisper)

# ---- miniaudio (single header: WAV/MP3/FLAC decoding + resampling) ----
set(THIRD_PARTY_DIR ${CMAKE_BINARY_DIR}/third_party)
set(MINIAUDIO_H ${THIRD_PARTY_DIR}/miniaudio.h)
if(NOT EXISTS ${MINIAUDIO_H})
  message(STATUS "Downloading miniaudio.h")
  file(DOWNLOAD
    https://raw.githubusercontent.com/mackron/miniaudio/0.11.21/miniaudio.h
    ${MINIAUDIO_H} STATUS dl_status TLS_VERIFY ON)
  list(GET dl_status 0 dl_code)
  if(NOT dl_code EQUAL 0)
    message(FATAL_ERROR "Failed to download miniaudio.h: ${dl_status}")
  endif()
endif()

add_executable(calltx src/main.cpp src/audio.cpp)
target_include_directories(calltx PRIVATE ${THIRD_PARTY_DIR})
target_link_libraries(calltx PRIVATE whisper)
if(UNIX AND NOT APPLE)
  target_link_libraries(calltx PRIVATE pthread m dl)
endif()

enable_testing()
add_test(NAME calltx_help COMMAND calltx -h)
set_tests_properties(calltx_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: .*calltx")
