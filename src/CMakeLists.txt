add_executable(sfml_cuda
    main.cpp
    options.cpp
    scenes/full_screen_opengl.cpp
    # scenes/full_screen_opengl.cu
    scenes/pt.cpp
  "scenes/sampler_std.h" "scenes/sampler.h" "scenes/pt_math.h")
# target_compile_options(sfml_cuda PRIVATE                        
#     $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS} --expt-relaxed-constexpr>
# )
# set_target_properties(sfml_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(sfml_cuda SYSTEM PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(sfml_cuda PRIVATE project_options project_warnings ${CUDA_LIBRARIES})

target_link_libraries_system(
  sfml_cuda PRIVATE  
  CONAN_PKG::docopt.cpp
  CONAN_PKG::fmt
  CONAN_PKG::spdlog
  CONAN_PKG::imgui-sfml
  CONAN_PKG::glew
  CONAN_PKG::eigen
  CONAN_PKG::magic_enum
)

