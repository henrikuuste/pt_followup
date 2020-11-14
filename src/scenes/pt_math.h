#pragma once

#include "../common.h"

struct Ray {
  Vec3 origin;
  Vec3 dir;

  int depth = 1;

  Ray &operator*=(Affine const &t) {
    origin = t * origin;
    dir    = (t.linear() * dir).normalized();
    return *this;
  }

  Ray transform(Affine const &t, Quat const &rot) const {
    Ray ret;
    ret.origin = t * origin;
    ret.dir    = rot * dir;
    ret.depth  = depth;
    return ret;
  }
};

inline Ray operator*(Affine const &t, Ray const &r) {
  return {t * r.origin, (t.linear() * r.dir).normalized(), r.depth};
}
