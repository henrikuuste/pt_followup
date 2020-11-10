#pragma once

#include <spdlog/spdlog.h>

#include <Eigen/Dense>

#include <cstdint>
#include <map>
#include <string>

using Vec3     = Eigen::Vector3f;
using Vec2     = Eigen::Vector2f;
using Radiance = Vec3;
using Affine   = Eigen::Affine3f;
using AngAx    = Eigen::AngleAxisf;
using Quat     = Eigen::Quaternionf;

#define R_PI float(EIGEN_PI)
#define R_INVPI float(1.f / R_PI)
#define R_2PI float(2.f * EIGEN_PI)
#define EPSILON 1e-5f

using Color4 = Eigen::Matrix<uint8_t, 4, 1>;

struct Pixel {
  Vec2 xy;
  Color4 color;
};

enum DisplayMode { MODE_COLOR, MODE_NORMAL, MODE_DEPTH, MODE_COUNT };

inline const char *mode_strings[] = {
    "color",
    "normal",
    "depth",
};

struct AppContext {
  size_t frame          = 0;
  float dtime           = 0.f;
  float elapsed_seconds = 0.f;
  float renderProgress  = 0.f;
  float exposure        = 1.f;
  float gamma           = 2.4f;
  int max_depth         = 2;
  int samples           = 2;
  float far_plane       = 20.f;
  std::map<std::string, bool> features;
  DisplayMode mode = MODE_COLOR;
};
