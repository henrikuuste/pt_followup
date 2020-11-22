#include "pt.h"
#include "sampler_curand.h"
using namespace std::literals::chrono_literals;

struct CudaRenderContext {
  Camera cam;
  AppContext app;
  DeviceScene scene;
  cuda::raw_ptr<Pixel> image;
  cuda::raw_ptr<Radiance> radiance;
};

__global__ void kRender(CudaRenderContext ctx) {
  auto x = blockIdx.x * blockDim.x + threadIdx.x;
  auto y = blockIdx.y * blockDim.y + threadIdx.y;

  int i          = (ctx.cam.h - y - 1) * ctx.cam.w + x;
  auto &pixel    = ctx.image[i];
  auto &radiance = ctx.radiance[i];

  if (ctx.app.spp == 0)
    radiance = Radiance::Zero();

  TraceContext tctx;
  tctx.app = &ctx.app;
  SamplerCurand smp;
  smp.init(ctx.app.spp + 1, i);
  tctx.sampler = &smp;

  for (int sample = 0; sample < ctx.app.samples; sample++) {
    Ray primary      = ctx.cam.castRay(pixel.xy, tctx);
    Radiance rSample = trace(ctx.scene, primary, tctx);
    radiance += rSample / ctx.app.samples;
  }
  // if (not rSample.isZero()) {
  //   Radiance change = rSample.cwiseQuotient(radianceBuffer[uidx]);
  //   if (not change.hasNaN()) {
  //     avgChange += change.maxCoeff();
  //     changeSamples++;
  //   }
  // }
  pixel.color = toSRGB((radiance / (ctx.app.spp + 1)), tctx);
  pixel.xy << x, y;
}

void PathTracer::renderCuda(Scene const &scene, Camera const &cam, std::mutex &sceneMutex,
                            AppContext &ctx, cuda::raw_ptr<Pixel> image) {

  CudaRenderContext cctx;
  {
    std::unique_lock lk(sceneMutex);
    cctx.cam   = cam;
    cctx.app   = ctx;
    cctx.scene = scene;
  }
  std::unique_lock lk(dataMutex);
  cctx.image    = image;
  cctx.radiance = radianceBuffer;

  dim3 block(16, 16, 1);
  dim3 grid(cam.w / block.x, cam.h / block.y, 1);
  kRender<<<grid, block, 0>>>(cctx);
  CUDA_CALL(cudaDeviceSynchronize());
}

/**********************************
 * Rendering
 **********************************/

CU_D Radiance trace(DeviceScene const &scene, Ray const &primary, TraceContext &ctx) {
  Radiance Lo                         = Radiance::Zero();
  Ray wo                              = primary;
  Radiance attenuation                = Radiance::Ones();
  Material::MaterialType lastMaterial = Material::SPEC; // Init as SPEC for Le at depth==0

  for (size_t depth = 0; depth < ctx.app->max_depth; ++depth) {
    auto hit = scene.intersect(wo);
    if (not hit)
      break;

    if (attenuation.squaredNorm() < EPSILON)
      break;

    if (depth == 0) {
      if (ctx.app->mode == DisplayMode::Normal) {
        Lo = hit.n;
        break;
      }
      if (ctx.app->mode == DisplayMode::Depth) {
        Vec3 invDepthColor = Vec3::Ones() * hit.distance / ctx.app->far_plane;
        invDepthColor      = invDepthColor.array().min(1.f).max(0.f);
        Lo                 = Vec3::Ones() - invDepthColor;
        break;
      }
    }

    if (lastMaterial == Material::SPEC || !(ctx.app->enable_NEE)) {
      Lo += attenuation.cwiseProduct(hit.object->mat.Le(hit, wo));
    }
    auto ms = hit.object->mat.sample(hit, wo, ctx);

    if (ctx.app->enable_NEE) {
      Lo += attenuation.cwiseProduct(sampleLights(hit, scene, ctx, ms));
    }

    attenuation  = attenuation.cwiseProduct(ms.fr * ms.wi.dir.dot(hit.n) / ms.pdf);
    wo           = ms.wi;
    lastMaterial = hit.object->mat.type;
  }

  return Lo;
}

CU_D Radiance sampleLights(Intersection const &hit, DeviceScene const &scene, TraceContext &ctx,
                           MaterialSample &ms) {
  Radiance radiance{0.0f, 0.0f, 0.0f};
  for (auto &o : scene.objects) {

    Radiance Le = o.mat.emittance;
    if (Le.isZero(0) || &o == hit.object) {
      continue;
    }
    if (o.type != SPHERE) {
      continue;
    }

    if (hit.object->mat.type == Material::SPEC) {
      continue;
    }

    Vec3 direction = o.uniformSampling(o.tr.translation() - hit.x, ctx);

    Ray obj2light{hit.x + hit.n * EPSILON, direction, 1};
    Intersection lightIntersect = scene.intersect(obj2light);
    if (lightIntersect.object != &o) {
      continue; // occlusion
    }
    float lightAngle = atan(o.sphere.radius / lightIntersect.distance);
    float solidAngle = R_PI * lightAngle * lightAngle;

    radiance += Le * hit.n.dot(direction) * solidAngle;
  }
  return ms.fr.cwiseProduct(radiance);
}

/**********************************
 * Geometry
 **********************************/

CU_HD Intersection Object::intersect(Ray const &worldRay) const {
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

CU_HD Intersection Sphere::intersect(Ray const &r, Object const *obj) const {
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
  if (t0 > t1) {
    auto tt = t0;
    t0      = t1;
    t1      = tt;
  }

  if (t0 < 0) {
    t0 = t1; // if t0 is negative, let's use t1 instead
    if (t0 < 0)
      return {}; // both t0 and t1 are negative
  }

  Vec3 x = r.origin + r.dir * t0;
  return {t0, obj, x, x.normalized()};
}

CU_HD Intersection Plane::intersect(Ray const &r, Object const *obj) const {
  float denom = normal.dot(-r.dir);
  if (denom < EPSILON)
    return {};

  float t = r.origin.dot(normal) / denom;
  if (t < 0)
    return {};
  Vec3 x = r.origin + r.dir * t;
  return {t, obj, x, normal};
}

CU_HD Intersection Disc::intersect(Ray const &r, Object const *obj) const {
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

CU_D Vec3 Object::uniformSampling(Vec3 const &dir, TraceContext &ctx) const {
  Vec3 sampledPoint;
  if (type == SPHERE) {
    sampledPoint = sphere.uniformSampling(dir, ctx, this);
  } else if (type == PLANE) {
    sampledPoint = plane.uniformSampling(dir, ctx, this);
  } else if (type == DISC) {
    sampledPoint = disc.uniformSampling(dir, ctx, this);
  }
  return sampledPoint;
}

CU_D Vec3 Sphere::uniformSampling(Vec3 const &dir, TraceContext &ctx,
                                  [[maybe_unused]] Object const *obj) const {
  Vec3 p         = uniformHemisphereSampling(ctx);
  p              = onb(p, dir);
  p              = radius * p;
  Vec3 direction = (dir + p).normalized();

  return direction;
}
CU_D Vec3 Plane::uniformSampling(Vec3 const &dir, TraceContext &ctx, Object const *obj) const {
  // TODO
  return Vec3::UnitX();
}
CU_D Vec3 Disc::uniformSampling(Vec3 const &dir, TraceContext &ctx, Object const *obj) const {
  // TODO
  return Vec3::UnitX();
}
/**********************************
 * BSDF
 **********************************/
CU_D Vec3 uniformHemisphereSampling(TraceContext &ctx) {
  float r     = sqrt(ctx.sample1D());
  float theta = ctx.sample1D() * 2 * R_PI;
  Vec3 p      = {r * cos(theta), 0, r * sin(theta)};
  p           = {p.x(), sqrt(fmaxf(0.0f, 1.0 - p.x() * p.x() - p.z() * p.z())), p.z()};
  return p;
}
CU_D Vec3 onb(Vec3 const &o, Vec3 const &dir) {
  Vec3 normal = dir.normalized();
  Vec3 binormal;
  if (fabs(normal.x()) > fabs(normal.z())) {
    binormal.x() = -normal.y();
    binormal.y() = normal.x();
    binormal.z() = 0;
  } else {
    binormal.x() = 0;
    binormal.y() = -normal.z();
    binormal.z() = normal.y();
  }
  binormal     = binormal.normalized();
  Vec3 tangent = binormal.cross(normal).normalized();

  Vec3 p = o.x() * tangent + o.y() * normal + o.z() * binormal;
  return p;
}

CU_D MaterialSample Material::sample(Intersection const &i, Ray const &wo,
                                     TraceContext &ctx) const {
  MaterialSample ms;

  if (type == DIFF) {
    ms.fr  = diffuse / R_PI;
    Vec3 d = uniformHemisphereSampling(ctx);
    d      = onb(d, i.n);
    ms.wi  = {i.x + i.n * EPSILON, d, wo.depth + 1};
    ms.pdf = 1.f / R_2PI;
  } else if (type == SPEC) {
    ms.fr  = diffuse;
    ms.wi  = {i.x, wo.dir - i.n * 2 * i.n.dot(wo.dir), wo.depth + 1};
    ms.pdf = ms.wi.dir.dot(i.n);
  } else {
    asm("exit;");
  }
  return ms;
}

/**********************************
 * Camera
 **********************************/

CU_D Ray Camera::castRay(Vec2 const &coord, TraceContext &ctx) const {
  Vec3 u   = Vec3::UnitX() * fov;
  Vec3 v   = Vec3::UnitY() * fov * h / w;
  Vec2 rnd = ctx.sample2D();
  Vec3 d =
      u * ((coord.x() + rnd.x()) / w - .5f) + v * ((coord.y() + rnd.y()) / h - .5f) + Vec3::UnitZ();
  return {tr.translation(), tr.linear() * d.normalized()};
}

CU_HD void Camera::setCameraTf(Affine const &tf) { tr = tf; }