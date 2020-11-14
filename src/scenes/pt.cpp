#include "pt.h"
#include "sampler_std.h"

#include <omp.h>
#include <random>

/**********************************
 * Rendering
 **********************************/

Radiance trace(Scene const &scene, Ray const &wo, TraceContext &ctx) {
  auto hit = scene.intersect(wo);
  if (not hit)
    return Radiance::Zero();

  if (ctx.app->mode == DisplayMode::Normal)
    return hit.n;
  if (ctx.app->mode == DisplayMode::Depth) {
    Vec3 invDepthColor = Vec3::Ones() * hit.distance / ctx.app->far_plane;
    invDepthColor      = invDepthColor.array().min(1.f).max(0.f);
    return Vec3::Ones() - invDepthColor;
  }
  if (shouldTerminate(wo, ctx))
    return hit.object->mat.Le(hit, wo);
  auto ms = hit.object->mat.sample(hit, wo, ctx);

  Radiance Li = trace(scene, ms.wi, ctx);

  return hit.object->mat.Le(hit, wo) + ms.fr.cwiseProduct(Li) * ms.wi.dir.dot(hit.n) / ms.pdf;
}

void PathTracer::render(Scene const &scene, Camera const &cam, AppContext &ctx,
                        std::vector<Pixel> &image) {

  SamplerStd smp;
  smp.init(ctx.spp + 1, 0);

  TraceContext tctx;
  tctx.app     = &ctx;
  tctx.sampler = &smp;

  float avgChange      = 0;
  size_t changeSamples = 0;

#pragma omp parallel for reduction(+ : avgChange) reduction(+ : changeSamples)
  for (int idx = 0; idx < (int)image.size(); ++idx) { // auto &pixel : image
    auto &pixel      = image.at(idx);
    Ray primary      = cam.castRay(pixel.xy, tctx);
    Radiance rSample = trace(scene, primary, tctx);
    radianceBuffer[idx] += rSample;
    if (not rSample.isZero()) {
      Radiance change = rSample.cwiseQuotient(radianceBuffer[idx]);
      if (not change.hasNaN()) {
        avgChange += change.maxCoeff();
        changeSamples++;
      }
    }
    pixel.color = toSRGB((radianceBuffer[idx] / (ctx.spp + 1)), tctx);
  }
  avgChange /= changeSamples;
  ctx.renderError = avgChange;
}

/**********************************
 * Geometry
 **********************************/

Intersection Object::intersect(Ray const &worldRay) const {
  Ray localRay;
  if (hasScale)
    localRay = invTr * worldRay; //!! about 1.6x slower, but allows for non-uniform scale
  else
    localRay = worldRay.transform(invTr, invRot);
  Intersection isect;
  if (type == SPHERE)
    isect = sphere.intersect(localRay, this);
  else if (type == DISC)
    isect = disc.intersect(localRay, this);
  else
    isect = plane.intersect(localRay, this);
  if (isect) {
    isect.x        = tr * isect.x;
    isect.distance = (isect.x - worldRay.origin).norm();
    if (hasScale)
      isect.n = (tr.linear() * isect.n).normalized();
    else
      isect.n = rot * isect.n;
  }
  return isect;
}

Intersection Sphere::intersect(Ray const &r, Object const *obj) const {
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

  // if (t0 < 0) {
  //   t0 = t1; // if t0 is negative, let's use t1 instead
  //   if (t0 < 0)
  //     return {}; // both t0 and t1 are negative
  // }

  if (t0 < 0 && t1 < 0) {
    return {};
  }

  Vec3 x = r.origin + r.dir * t0;
  return {t0, obj, x, x.normalized()};
}

Intersection Plane::intersect(Ray const &r, Object const *obj) const {
  float denom = normal.dot(-r.dir);
  if (denom < EPSILON)
    return {};

  float t = r.origin.dot(normal) / denom;
  if (t < 0)
    return {};
  Vec3 x = r.origin + r.dir * t;
  return {t, obj, x, normal};
}

Intersection Disc::intersect(Ray const &r, Object const *obj) const {
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

/**********************************
 * BSDF
 **********************************/

MaterialSample Material::sample(Intersection const &i, Ray const &wo, TraceContext &ctx) const {
  MaterialSample ms;
  ms.fr = diffuse;

  if (type == DIFF) {
    Vec3 d = (ctx.sample3D() * 2.f - Vec3::Ones()).normalized();
    if (d.dot(i.n) < 0) {
      d = -d;
    }
    ms.wi = {i.x + i.n * EPSILON, d, wo.depth + 1};

  } else if (type == SPEC) {
    ms.wi = {i.x + i.n * EPSILON, wo.dir - i.n * 2 * i.n.dot(wo.dir), wo.depth + 1};
  } else if (type == REFR) {
    float refractive_idx = 1.52;
    float ro             = (1.0 - refractive_idx) / (1.0 + refractive_idx);
    ro                   = ro * ro;

    Vec3 n           = i.n;
    float normal_dir = 1;
    if (n.dot(wo.dir) > 0) { // Determining if within medium
      normal_dir     = -1;
      n              = n * -1;
      refractive_idx = 1 / refractive_idx;
    }
    refractive_idx    = 1 / refractive_idx;
    double cos_theta1 = (n.dot(wo.dir)) * -1; // Computing CosTheta1
    double cos_theta2 = 1.0 - refractive_idx * refractive_idx *
                                  (1.0 - cos_theta1 * cos_theta1); // Computing CostTheta2
    double rProb = ro + (1.0 - ro) * pow(1.0 - cos_theta1, 5.0);   // Schlick Approximation
    if (cos_theta2 > 0 && ctx.sample1D() > rProb) {                // Refraction
      Vec3 direction =
          ((wo.dir * refractive_idx) + (n * (refractive_idx * cos_theta1 - sqrt(cos_theta2))));
      ms.wi = {i.x - normal_dir * i.n * EPSILON, direction, wo.depth + 1};
    } else { // Else do reflection
      Vec3 direction = wo.dir - i.n * 2 * i.n.dot(wo.dir);
      ms.wi          = {i.x + normal_dir * i.n * EPSILON, direction, wo.depth + 1};
    }
  } else {
    spdlog::error("Not implemented");
    abort();
  }
  ms.pdf = 1.f;
  return ms;
}

/**********************************
 * Camera
 **********************************/

Ray Camera::castRay(Vec2 const &coord, TraceContext &ctx) const {
  Vec3 u   = Vec3::UnitX() * fov;
  Vec3 v   = Vec3::UnitY() * fov * h / w;
  Vec2 rnd = ctx.sample2D();
  Vec3 d =
      u * ((coord.x() + rnd.x()) / w - .5) + v * ((coord.y() + rnd.y()) / h - .5) + Vec3::UnitZ();
  return {tr.translation(), tr.linear() * d.normalized()};
}