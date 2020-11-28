#pragma once

#include "../common.h"

struct Ray {
  Vec3 origin;
  Vec3 dir;

  int depth = 1;

  CU_HD Ray &operator*=(Affine const &t) {
    origin = t * origin;
    dir    = (t.linear() * dir).normalized();
    return *this;
  }

  CU_HD Ray transform(Affine const &t, Quat const &rot) const {
    Ray ret;
    ret.origin = t * origin;
    ret.dir    = rot * dir;
    ret.depth  = depth;
    return ret;
  }
};

inline CU_HD Ray operator*(Affine const &t, Ray const &r) {
  return {t * r.origin, (t.linear() * r.dir).normalized(), r.depth};
}

struct OrthonormalBasis {
  Vec3 normal;
  Vec3 binormal;
  Vec3 tangent;
  CU_HD Vec3 changeBasis(Vec3 &v) const {
    return v.x() * tangent + v.y() * binormal + v.z() * normal;
  }
};
