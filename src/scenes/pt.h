#pragma once

#include "../common.h"

#include <random>
#include <vector>

struct Ray {
  Vec3 origin;
  Vec3 dir;

  int depth = 1;
};

inline Ray operator*(Affine const &t, Ray const &r) {
  return {t * r.origin, t.linear() * r.dir, r.depth};
}

inline std::random_device rd;  // Will be used to obtain a seed for the random number engine
inline std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
inline std::uniform_real_distribution<float> dis(0.0f, 1.f);

struct Sampler {
  static float sample1D() { return dis(gen); }
  static Vec2 sample2D() { return {dis(gen), dis(gen)}; }
  static Vec3 sample3D() { return {dis(gen), dis(gen), dis(gen)}; }
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
  Ray castRay(Vec2 const &coord) const {
    Vec3 u   = Vec3::UnitX() * fov;
    Vec3 v   = Vec3::UnitY() * fov * h / w;
    Vec2 rnd = Sampler::sample2D();
    Vec3 d =
        u * ((coord.x() + rnd.x()) / w - .5) + v * ((coord.y() + rnd.y()) / h - .5) + Vec3::UnitZ();
    return {tr.translation(), tr.linear() * d.normalized()};
  }
};

struct Material {
  Radiance diffuse;
  Radiance emittance = Radiance::Zero();

  Radiance Le(Intersection const &i, Ray const &wo) const { return emittance; }
  MaterialSample sample(Intersection const &i, Ray const &wo) const {
    MaterialSample ms;
    ms.fr  = diffuse;
    Vec3 d = (Sampler::sample3D() * 2.f - Vec3::Ones()).normalized();
    if (d.dot(i.n) < 0)
      d = -d;
    ms.wi  = {i.x + i.n * EPSILON, d, wo.depth + 1};
    ms.pdf = 1.f;
    return ms;
  }
};

struct Sphere {
  float radius = 1.f;
  Intersection intersect(Ray const &r, Object const *obj) const {
    Vec3 L    = -r.origin;
    float tca = L.dot(r.dir);
    if (tca < 0)
      return {};
    float d2 = L.dot(L) - tca * tca;
    if (d2 > radius * radius)
      return {};
    float thc = std::sqrt(radius * radius - d2);
    float t0  = tca - thc;
    float t1  = tca + thc;
    if (t0 > t1)
      std::swap(t0, t1);

    if (t0 < 0) {
      t0 = t1; // if t0 is negative, let's use t1 instead
      if (t0 < 0)
        return {}; // both t0 and t1 are negative
    }

    Vec3 x = r.origin + r.dir * t0;
    return {t0, obj, x, x.normalized()};
  }
};

struct Plane {
  Vec3 normal = Vec3::UnitY();
  Intersection intersect(Ray const &r, Object const *obj) const {
    float denom = normal.dot(-r.dir);
    if (denom < EPSILON)
      return {};

    float t = r.origin.dot(normal) / denom;
    if (t < 0)
      return {};
    Vec3 x = r.origin + r.dir * t;
    return {t, obj, x, normal};
  }
};

struct Disc {
  Vec3 normal  = Vec3::UnitY();
  float radius = 1.f;
  Intersection intersect(Ray const &r, Object const *obj) const {
    float denom = normal.dot(-r.dir);
    if (denom < EPSILON)
      return {};

    float t = r.origin.dot(normal) / denom;
    if (t < 0)
      return {};
    Vec3 x = r.origin + r.dir * t;
    if (x.squaredNorm() > radius * radius)
      return {};
    return {t, obj, x, normal};
  }
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
  Object(Object const &o) : mat(o.mat), type(o.type), tr(o.tr), invTr(o.invTr) {
    if (type == SPHERE)
      sphere = o.sphere;
    else if (type == DISC)
      disc = o.disc;
    else
      plane = o.plane;
  }

  void setTransform(Affine const &t) {
    tr    = t;
    invTr = t.inverse();
  }

  Intersection intersect(Ray const &r) const {
    Ray local = invTr * r;
    Intersection isect;
    if (type == SPHERE)
      isect = sphere.intersect(local, this);
    else if (type == DISC)
      isect = disc.intersect(local, this);
    else
      isect = plane.intersect(local, this);
    if (isect) {
      isect.x = tr * isect.x;
      isect.n = tr.linear() * isect.n;
    }
    return isect;
  }
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

struct PathTracer {
  Color4 toSRGB(Radiance r, AppContext &ctx) const {
    r = r.array().min(1.f).max(0.f).pow(1.f / ctx.gamma) * 255.f;
    Color4 c;
    c << r.cast<uint8_t>(), 255;
    return c;
  }

  std::vector<Radiance> radianceBuffer;

  void reset(Camera const &cam) { radianceBuffer.resize(cam.w * cam.h, Radiance::Zero()); }

  void render(Scene const &scene, Camera const &cam, AppContext &ctx, std::vector<Pixel> &image) {
    // 	foreach pixel in imageBuffer:
    size_t idx = 0;
    for (auto &pixel : image) {
      // 		for sample in range(0, N):
      // for (size_t sample = 0; sample < ctx.samples; ++sample)
      // 			pixel.radiance += trace(camera.castRay(pixel.xy)) / N
      radianceBuffer[idx] += trace(scene, cam.castRay(pixel.xy), ctx);

      // 		pixel.color = pixel.toSRGB()
      pixel.color = toSRGB((radianceBuffer[idx] / (ctx.frame + 1)) * ctx.exposure, ctx);
      idx++;
    }
  }

  bool shouldTerminate(Ray const &r, AppContext &ctx) const { return r.depth > ctx.max_depth; }

  Radiance trace(Scene const &scene, Ray const &wo, AppContext &ctx) const {
    // 	object = scene.intersect(x, wo)
    auto hit = scene.intersect(wo);
    // 	if not object	return Radiance(0)
    if (not hit)
      return Radiance::Zero();

    if (ctx.mode == MODE_NORMAL)
      return hit.n;
    if (ctx.mode == MODE_DEPTH) {
      Vec3 invDepthColor = Vec3::Ones() * hit.distance / ctx.far_plane;
      invDepthColor      = invDepthColor.array().min(1.f).max(0.f);
      return Vec3::Ones() - invDepthColor;
    }
    // 	if shouldTerminate() return object.Le(x, wo)
    if (shouldTerminate(wo, ctx))
      return hit.object->mat.Le(hit, wo);
    // 	wi = object.sampleRay(x, wo)
    auto ms = hit.object->mat.sample(hit, wo);
    // 	Li = trace(x, wi)
    Radiance Li = trace(scene, ms.wi, ctx);
    // 	return object.Le(x, wo) + 2*PI * object.fr(x, wo, wi) * Li * wi.dot(object.n)
    return hit.object->mat.Le(hit, wo) + ms.fr.cwiseProduct(Li) * ms.wi.dir.dot(hit.n) / ms.pdf;
  }
};