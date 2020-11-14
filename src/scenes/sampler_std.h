#pragma once

#include "sampler.h"
#include <random>

struct SamplerStd : public Sampler {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dis = std::uniform_real_distribution<float>{0.0f, 1.f};

  void init(uint32_t seed, uint32_t offset) override {
    std::seed_seq seq{seed, offset};
    gen.seed(seq);
  }
  float get1D() override { return dis(gen); }
  Vec2 get2D() override { return {dis(gen), dis(gen)}; }
  Vec3 get3D() override { return {dis(gen), dis(gen), dis(gen)}; }
};