#pragma once

#include "../common.h"

struct Sampler {
  virtual ~Sampler(){};

  virtual void init(uint32_t seed, uint32_t offset) = 0;
  virtual float get1D()                             = 0;
  virtual Vec2 get2D()                              = 0;
  virtual Vec3 get3D()                              = 0;
};