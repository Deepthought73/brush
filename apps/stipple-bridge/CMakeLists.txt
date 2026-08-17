if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CARGO_CMD cargo build -p stipple-bridge --verbose)
    set(TARGET_DIR "debug")
else ()
    set(CARGO_CMD cargo build -p stipple-bridge --release --verbose)
    set(TARGET_DIR "release")
endif ()

# If the user is cross compiling to a different target, update the location of the target directory
# *Note*: Make sure to install the target with rustup, as well as setting the proper environment variables, such as CC, HOST_CC, CARGO_TARGET_<triple>_LINKER, ...
# See https://doc.rust-lang.org/cargo/reference/environment-variables.html#configuration-environment-variables
if (DEFINED ENV{CARGO_BUILD_TARGET})
    set(TARGET_DIR "$ENV{CARGO_BUILD_TARGET}/${TARGET_DIR}")
endif()

if(ENABLE_LTO)
    set(RUST_FLAGS "-Clinker-plugin-lto -Clinker=${CMAKE_C_COMPILER} -Clink-arg=-fuse-ld=lld")
endif()

set(STIPPLE_BRIDGE_LIB "${CMAKE_CURRENT_BINARY_DIR}/${TARGET_DIR}/libstipple_bridge.a")
set(STIPPLE_BRIDGE_CXX "${CMAKE_CURRENT_BINARY_DIR}/stipple_bridge.cpp")
set(STIPPLE_BRIDGE_H   "${CMAKE_CURRENT_BINARY_DIR}/stipple_bridge.h")

# The C++ bridge headers under include/basalt/rt_splatter/ (ImageBridge,
# RectificationBridge, DepthProviderBridge) pull in basalt's own headers plus
# OpenCV/TBB/Eigen/Sophus/cereal. Rather than hand-listing each -I as new
# bridges need new headers, harvest the real include dirs from the targets
# basalt already links for this code, and hand them to cxx_build as one env
# var. NOTE: get_target_property only reads a target's *own*
# INTERFACE_INCLUDE_DIRECTORIES, it does not walk INTERFACE_LINK_LIBRARIES
# transitively (that resolution only happens internally when CMake generates
# a real compile command) -- so basalt::basalt-headers's own entry here does
# NOT pull in the Eigen/Sophus/cereal it links against; those leaf targets
# must be listed explicitly too.
set(STIPPLE_BRIDGE_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/include)
foreach(_target basalt::basalt-headers Eigen3::Eigen Sophus::Sophus cereal::cereal basalt::opencv TBB::tbb)
    if(TARGET ${_target})
        get_target_property(_target_includes ${_target} INTERFACE_INCLUDE_DIRECTORIES)
        if(_target_includes)
            list(APPEND STIPPLE_BRIDGE_INCLUDE_DIRS ${_target_includes})
        endif()
    endif()
endforeach()
list(REMOVE_DUPLICATES STIPPLE_BRIDGE_INCLUDE_DIRS)
# Join with ':' (not CMake's native ';') since that's what build.rs splits on,
# and a raw ';' would be ambiguous when passed through `cmake -E env`. Entries
# above are BUILD_INTERFACE/INSTALL_INTERFACE-wrapped generator expressions;
# they only evaluate at generate time (here, in the COMMAND below), and
# INSTALL_INTERFACE ones evaluate to empty during a normal build, leaving
# stray "::" in the joined string -- build.rs skips empty path segments to
# tolerate that.
string(REPLACE ";" ":" BASALT_INCLUDE_DIRS_JOINED "${STIPPLE_BRIDGE_INCLUDE_DIRS}")

add_custom_target(stipple_bridge_rust
    BYPRODUCTS ${STIPPLE_BRIDGE_LIB} ${STIPPLE_BRIDGE_CXX} ${STIPPLE_BRIDGE_H}
    COMMAND ${CMAKE_COMMAND} -E env
            CARGO_TARGET_DIR=${CMAKE_CURRENT_BINARY_DIR}
            RUSTFLAGS=${RUST_FLAGS}
            BASALT_INCLUDE_DIRS=${BASALT_INCLUDE_DIRS_JOINED}
            ${CARGO_CMD}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/stipple-bridge/src/lib.rs.cc
            ${STIPPLE_BRIDGE_CXX}
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge/stipple-bridge/src/lib.rs.h
            ${STIPPLE_BRIDGE_H}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    USES_TERMINAL
    VERBATIM
)
set_source_files_properties(${STIPPLE_BRIDGE_CXX} PROPERTIES GENERATED TRUE)

add_library(stipple_bridge STATIC ${STIPPLE_BRIDGE_CXX})
add_dependencies(stipple_bridge stipple_bridge_rust)
target_link_libraries(stipple_bridge PUBLIC pthread dl ${STIPPLE_BRIDGE_LIB})
# PUBLIC: consumers (e.g. basalt_vio, via rt_splatter.h) need rust/cxx.h too --
# cargo copies it to <CARGO_TARGET_DIR>/cxxbridge/rust/cxx.h, a sibling of the
# stipple_bridge.h copied into CMAKE_CURRENT_BINARY_DIR itself, not inside it.
# PRIVATE: only stipple_bridge.cpp itself (the generated glue, which #includes
# image_bridge.h/rectification_bridge.h/depth_provider_bridge.h directly) needs
# the harvested basalt/Eigen/Sophus/cereal/OpenCV/TBB dirs -- basalt_vio already
# gets those independently through its own basalt::opencv/basalt::basalt-headers
# links, so there's no need to re-expose them here.
target_include_directories(stipple_bridge
    PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/cxxbridge
    PRIVATE ${STIPPLE_BRIDGE_INCLUDE_DIRS}
)
