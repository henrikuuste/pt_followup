#pragma once

#include "../common.h"
#include "../cuda_memory.hpp"
#include "pt_math.h"
#include "sampler.h"
#include <ciso646>
#include <mutex>
#include <vector>

struct TraceContext {
  AppContext *app;
  Sampler *sampler;
  CU_D float sample1D() { return sampler->get1D(); }
  CU_D Vec2 sample2D() { return sampler->get2D(); }
  CU_D Vec3 sample3D() { return sampler->get3D(); }
};

struct MaterialSample {
  Radiance fr;
  Ray wi;
  float pdf;
};

struct ObjectSample {
  float pdf;
  Vec3 p;
  Vec3 n;
};

struct Object;

struct Intersection {
  float distance       = -1.f;
  Object const *object = nullptr;
  Vec3 x;
  Vec3 n;
  CU_HD operator bool() const { return valid(); }
  CU_HD bool valid() const { return object != nullptr and distance > 0.f; }
  CU_HD bool operator<(Intersection const &other) const {
    return valid() and (!other.valid() or distance < other.distance);
  }
};

struct Camera {
  Affine tr = Affine::Identity();
  float w, h;
  float fov = R_PI * .5f;
  CU_D Ray castRay(Vec2 const &coord, TraceContext &ctx) const;
  CU_HD void setCameraTf(Affine const &tf);
};

struct alignas(8) Material {
  enum MaterialType { DIFF, SPEC };
  Radiance diffuse;
  Radiance emittance = Radiance::Zero();
  MaterialType type  = DIFF;

  CU_HD Radiance Le([[maybe_unused]] Intersection const &i, [[maybe_unused]] Ray const &wo) const {
    return emittance;
  }
  CU_D MaterialSample sample(Intersection const &i, Ray const &wo, TraceContext &ctx) const;
};

struct Sphere {
  float radius = 1.f;
  CU_HD Intersection intersect(Ray const &r, Object const *obj) const;
  CU_D ObjectSample sample(Vec3 const &dir, TraceContext &ctx, Object const *obj) const;
};

struct Plane {
  Vec3 normal = Vec3::UnitY();
  CU_HD Intersection intersect(Ray const &r, Object const *obj) const;
  CU_D ObjectSample sample(Vec3 const &dir, TraceContext &ctx, Object const *obj) const;
};

struct Disc {
  Vec3 normal  = Vec3::UnitY();
  float radius = 1.f;
  CU_HD Intersection intersect(Ray const &r, Object const *obj) const;
  CU_D ObjectSample sample(Vec3 const &dir, TraceContext &ctx, Object const *obj) const;
};

enum ObjectType {
  SPHERE,
  PLANE,
  DISC,
};

struct alignas(16) Object {
  std::string name;
  Material mat;
  ObjectType type;
  alignas(16) Affine tr;
  alignas(16) Affine invTr;
  alignas(16) Quat rot;
  alignas(16) Quat invRot;
  bool hasScale = false;

  union {
    Sphere sphere;
    Plane plane;
    Disc disc;
  };

  Object(std::string_view n, Sphere const &obj, Material const &m, Affine const &t)
      : name(n), mat(m), type(SPHERE), sphere(obj) {
    setTransform(t);
  }
  Object(std::string_view n, Plane const &obj, Material const &m, Affine const &t)
      : name(n), mat(m), type(PLANE), plane(obj) {
    setTransform(t);
  }
  Object(std::string_view n, Disc const &obj, Material const &m, Affine const &t)
      : name(n), mat(m), type(DISC), disc(obj) {
    setTransform(t);
  }
  Object(Object const &o) : name(o.name), mat(o.mat), type(o.type) {
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
    if (not tr.rotation().isApprox(tr.linear())) {
      spdlog::info("{} has scale", name);
      hasScale = true;
    }
  }

  CU_HD Intersection intersect(Ray const &r) const;
  CU_D ObjectSample sample(Vec3 const &dir, TraceContext &ctx) const;
};

struct DeviceScene {
  cuda::raw_ptr<Object> objects;
  CU_HD Intersection intersect(Ray const &r, float dist = 0) const {
    Intersection ret;
    for (auto &o : objects) {
      auto test = o.intersect(r);
      if (test < ret) {
        ret = test;
        if (dist) {
          if (ret.distance + EPSILON < dist) {
            break;
          }
        }
      }
    }
    return ret;
  }
};

struct Scene {
  std::vector<Object> objects;
  operator DeviceScene() const {
    if (objects.size() != deviceObjects_.size())
      deviceObjects_.allocateManaged(objects.size());
    if (dirty_.load()) {
      spdlog::debug("Copying scene to GPU");
      CUDA_CALL(cudaMemcpy(deviceObjects_.get(), objects.data(), deviceObjects_.sizeBytes(),
                           cudaMemcpyHostToDevice));
      dirty_ = false;
    }
    return {deviceObjects_};
  }

private:
  mutable std::atomic_bool dirty_ = true;
  mutable cuda::owning_ptr<Object> deviceObjects_;
};

CU_HD inline Color4 toSRGB(Radiance r, TraceContext &ctx) {
  r = (r * ctx.app->exposure).array().min(1.f).max(0.f).pow(1.f / ctx.app->gamma) * 255.f;
  Color4 c;
  c << r.cast<uint8_t>(), 255;
  return c;
}

CU_HD inline bool shouldTerminate(Ray const &r, TraceContext &ctx) {
  return r.depth > ctx.app->max_depth;
}

CU_D Radiance sampleLights(Intersection const &hit, DeviceScene const &scene, TraceContext &ctx,
                           MaterialSample &ms);
CU_D Radiance trace(DeviceScene const &scene, Ray const &wo, TraceContext &ctx);
CU_D Vec3 uniformHemisphereSampling(TraceContext &ctx);
CU_D Vec3 cosineWeightedHemisphereSampling(TraceContext &ctx);
CU_D OrthonormalBasis onb(Vec3 const &dir);

struct PathTracer {
  cuda::owning_ptr<Radiance> radianceBuffer;
  std::mutex dataMutex;

  void reset(Camera const &cam) {
    if (radianceBuffer.size() != (cam.w * cam.h)) {
      std::unique_lock lk(dataMutex);
      radianceBuffer.allocateManaged(cam.w * cam.h);
    }
  }
  void render(Scene const &scene, Camera const &cam, AppContext &ctx, std::vector<Pixel> &image);
  void renderCuda(Scene const &scene, Camera const &cam, std::mutex &sceneMutex, AppContext &ctx,
                  cuda::raw_ptr<Pixel> image);
};