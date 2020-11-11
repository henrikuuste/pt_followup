#pragma once

#include "../common.h"

struct Ray {
  Vec3 origin;
  Vec3 dir;

  int depth = 1;
};

inline Ray operator*(Affine const &t, Ray const &r) {
  return {t * r.origin, t.linear() * r.dir, r.depth};
}
