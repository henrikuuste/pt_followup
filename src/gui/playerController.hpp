#pragma once

#include "../common.h"
#include "../scenes/full_screen_opengl.h"

#include <SFML/Window/Event.hpp>
#include <fmt/ostream.h>

class PlayerController {
  FullScreenOpenGLScene *scene_;

  static constexpr float maxWalkSpeed = 10.f;
  static constexpr float maxLookSpeed = 2.f;

  Vec3 walkSpeed_ = {0.f, 0.f, 0.f};
  Vec3 lookSpeed_ = {0.f, 0.f, 0.f};
  float height_;
  Vec3 pos_;
  Quat look_;
  Quat yaw_;
  Quat pitch_;

public:
  PlayerController(FullScreenOpenGLScene &scene) : scene_(&scene) { reset(); }

  void reset() {
    // Vec3 camDir = scene_->cam_.tr.linear() * Vec3::UnitZ();
    // facingDir_  = (camDir - camDir.dot(Vec3::UnitY()) * Vec3::UnitY()).normalized();
    pos_        = scene_->cam_.tr.translation();
    look_       = scene_->cam_.tr.rotation();
    yaw_        = Quat(look_.w(), 0, look_.y(), 0).normalized();
    pitch_      = yaw_.conjugate() * look_;
    pitch_.z() = 0;
    pitch_.normalize();
    spdlog::info("pos: {} rot: {}", pos_.transpose(), look_.coeffs().transpose());
    height_  = pos_.y();
    pos_.y() = 0.f;
  }

  void handleEvent(sf::Event const &event) {
    // TODO
  }

  void update(AppContext &ctx) {
    walkSpeed_ << 0, 0, 0;
    lookSpeed_ << 0, 0, 0;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
      walkSpeed_.x() = maxWalkSpeed;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
      walkSpeed_.x() = -maxWalkSpeed;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
      walkSpeed_.z() = maxWalkSpeed;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
      walkSpeed_.z() = -maxWalkSpeed;
    }

    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
      lookSpeed_.x() = -maxLookSpeed;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
      lookSpeed_.x() = maxLookSpeed;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
      lookSpeed_.y() = -maxLookSpeed;
    }
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
      lookSpeed_.y() = maxLookSpeed;
    }

    Vec3 dLookEuler{lookSpeed_.x() * ctx.dtime, lookSpeed_.y() * ctx.dtime, 0};
    yaw_   = Quat{AngAx(dLookEuler.y(), Vec3::UnitY())} * yaw_;
    pitch_ = Quat{AngAx(dLookEuler.x(), Vec3::UnitX())} * pitch_;

    look_          = yaw_ * pitch_;
    Vec3 facingDir = yaw_ * Vec3::UnitZ();

    Vec3 sideDir = facingDir.cross(Vec3::UnitY());
    Vec3 dPos    = (facingDir * walkSpeed_.z() + sideDir * walkSpeed_.x()) * ctx.dtime;

    float moved = std::abs(dPos.norm()) > EPSILON or std::abs(dLookEuler.norm()) > EPSILON;
    pos_ += dPos;
    if (moved) {
      Affine tf   = Affine::Identity();
      tf.linear() = look_.toRotationMatrix();
      tf.translation() << pos_ + Vec3::UnitY() * height_;
      scene_->setCameraTf(tf, ctx);
    }
  }
};
