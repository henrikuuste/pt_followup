#pragma once

#include "curand.h"
#include "curand_kernel.h"
#include "sampler.h"

struct SamplerCurand : public Sampler {
  curandState rng;

  CU_D void init(uint32_t seed, uint32_t offset) override {
    curand_init(seed * offset, 0, 0, &rng);
  }
  CU_D float get1D() override { return curand_uniform(&rng); }
  CU_D Vec2 get2D() override { return {get1D(), get1D()}; }
  CU_D Vec3 get3D() override { return {get1D(), get1D(), get1D()}; }
};