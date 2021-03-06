#pragma once

#include <spdlog/spdlog.h>

#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include <Eigen/Dense>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <cstdint>
#include <map>
#include <string>

using Vec3     = Eigen::Vector3f;
using Vec2     = Eigen::Vector2f;
using Vec4     = Eigen::Matrix<float, 4, 1>;
using Radiance = Vec3;
using Affine   = Eigen::Affine3f;
using AngAx    = Eigen::AngleAxisf;
using Quat     = Eigen::Quaternionf;
using Mat4     = Eigen::Matrix<float, 4, 4>;
using Mat3     = Eigen::Matrix<float, 3, 3>;

#define R_PI float(EIGEN_PI)
#define R_INVPI float(1.f / R_PI)
#define R_2PI float(2.f * EIGEN_PI)
#define EPSILON 1e-5f

using Color4 = Eigen::Matrix<uint8_t, 4, 1>;

struct Pixel {
  Vec2 xy;
  Color4 color;
};

enum class DisplayMode : int { Color, Normal, Depth };

struct AppContext {
  size_t spp            = 0;
  float dtime           = 0.f;
  float elapsed_seconds = 0.f;
  float renderError     = 0.f;
  float exposure        = 1.f;
  float gamma           = 2.4f;
  int max_depth         = 4;
  int samples           = 1;
  float far_plane       = 20.f;
  bool request_reset    = true;
  bool enable_NEE       = true;
  // std::map<std::string, bool> features;
  DisplayMode mode = DisplayMode::Color;
};
