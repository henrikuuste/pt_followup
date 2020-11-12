#pragma once

#include "../common.h"
#include "pt_math.h"
#include "sampler.h"
#include <vector>


struct TraceContext {
  AppContext *app;
  Sampler *sampler;
  float sample1D() { return sampler->get1D(); }
  Vec2 sample2D() { return sampler->get2D(); }
  Vec3 sample3D() { return sampler->get3D(); }
};

struct MaterialSample {
  Radiance fr;
  Ray wi;
  float pdf;
};

struct Object;

struct Intersection {
  float distance       = -1.f;
  Object const *object = nullptr;
  Vec3 x;
  Vec3 n;
  operator bool() const { return valid(); }
  bool valid() const { return object != nullptr and distance > 0.f; }
  bool operator<(Intersection const &other) const {
    return valid() and (!other.valid() or distance < other.distance);
  }
};

struct Camera {
  Affine tr = Affine::Identity();
  float w, h;
  float fov = R_PI * .5f;
  Ray castRay(Vec2 const &coord, TraceContext &ctx) const;
};

struct Material {
  Radiance diffuse;
  Radiance emittance = Radiance::Zero();

  Radiance Le(Intersection const &i, Ray const &wo) const { return emittance; }
  MaterialSample sample(Intersection const &i, Ray const &wo, TraceContext &ctx) const;
};

struct Sphere {
  float radius = 1.f;
  Intersection intersect(Ray const &r, Object const *obj) const;
};

struct Plane {
  Vec3 normal = Vec3::UnitY();
  Intersection intersect(Ray const &r, Object const *obj) const;
};

struct Disc {
  Vec3 normal  = Vec3::UnitY();
  float radius = 1.f;
  Intersection intersect(Ray const &r, Object const *obj) const;
};

enum ObjectType {
  SPHERE,
  PLANE,
  DISC,
};

struct Object {
  Material mat;
  ObjectType type;
  Affine tr;
  Affine invTr;
  Quat rot;
  Quat invRot;

  union {
    Sphere sphere;
    Plane plane;
    Disc disc;
  };

  Object(Sphere const &obj, Material const &m, Affine const &t)
      : mat(m), type(SPHERE), sphere(obj) {
    setTransform(t);
  }
  Object(Plane const &obj, Material const &m, Affine const &t) : mat(m), type(PLANE), plane(obj) {
    setTransform(t);
  }
  Object(Disc const &obj, Material const &m, Affine const &t) : mat(m), type(DISC), disc(obj) {
    setTransform(t);
  }
  Object(Object const &o) : mat(o.mat), type(o.type) {
    setTransform(o.tr);
    if (type == SPHERE)
      sphere = o.sphere;
    else if (type == DISC)
      disc = o.disc;
    else
      plane = o.plane;
  }

  void setTransform(Affine const &t) {
    tr     = t;
    invTr  = t.inverse();
    rot    = tr.rotation();
    invRot = invTr.rotation();
  }

  Intersection intersect(Ray const &r) const;
};

struct Scene {
  std::vector<Object> objects;
  Intersection intersect(Ray const &r) const {
    Intersection ret;
    for (auto &o : objects) {
      auto test = o.intersect(r);
      if (test < ret)
        ret = test;
    }
    return ret;
  }
};

inline Color4 toSRGB(Radiance r, TraceContext &ctx) {
  r = (r * ctx.app->exposure).array().min(1.f).max(0.f).pow(1.f / ctx.app->gamma) * 255.f;
  Color4 c;
  c << r.cast<uint8_t>(), 255;
  return c;
}

inline bool shouldTerminate(Ray const &r, TraceContext &ctx) {
  return r.depth > ctx.app->max_depth;
}

Radiance trace(Scene const &scene, Ray const &wo, TraceContext &ctx);

struct PathTracer {
  std::vector<Radiance> radianceBuffer;

  void reset(Camera const &cam) { radianceBuffer.resize(cam.w * cam.h, Radiance::Zero()); }
  void render(Scene const &scene, Camera const &cam, AppContext &ctx, std::vector<Pixel> &image);
};