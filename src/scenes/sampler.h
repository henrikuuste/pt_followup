#pragma once

#include "../common.h"

struct Sampler {
  virtual CU_D ~Sampler(){};

  virtual CU_D void init(uint32_t seed, uint32_t offset) = 0;
  virtual CU_D float get1D()                             = 0;
  virtual CU_D Vec2 get2D()                              = 0;
  virtual CU_D Vec3 get3D()                              = 0;
};