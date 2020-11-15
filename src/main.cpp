#include "common.h"

#include "imgui_helpers.hpp"
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>

#include "cuda_wrapper.h"
#include "options.h"
#include "scenes/full_screen_opengl.h"
#include "gui/stats.hpp"
#include "gui/control.hpp"

static constexpr float LIN_SPEED = 8.0f;
static constexpr float ANG_SPEED = 2.0f;

void handleEvents(sf::RenderWindow &window, FullScreenOpenGLScene &scene, AppContext &ctx);

int main(int argc, const char **argv) {
  Options opt({std::next(argv), std::next(argv, argc)});
  opt.checkOptions();

  spdlog::info("Starting application");

  int gpuId = cuda::findBestDevice();
  cuda::gpuDeviceInit(gpuId);

  sf::RenderWindow window(sf::VideoMode(opt.width, opt.height), "SFML + CUDA",
                          sf::Style::Titlebar | sf::Style::Close);
  ImGui::SFML::Init(window);
  window.setFramerateLimit(60);
  spdlog::info("SFML window created");

  FullScreenOpenGLScene scene(window);
  StatsGUI statsGUI;
  ControlGUI controlGUI(scene);

  AppContext ctx;
  sf::Clock deltaClock;

  scene.run(ctx);
  while (window.isOpen()) {
    ImGui::SFML::Update(window, deltaClock.restart());

    statsGUI.draw(ctx);
    controlGUI.draw(ctx);

    window.clear();
    scene.render(window);
    ImGui::SFML::Render(window);
    window.display();

    handleEvents(window, scene, ctx);
    ctx.dtime = deltaClock.getElapsedTime().asSeconds();
  }

  spdlog::info("Shutting down");
  ImGui::SFML::Shutdown();

  return 0;
}

void handleEvents(sf::RenderWindow &window, FullScreenOpenGLScene &scene, AppContext &ctx) {
  sf::Event event{};
  while (window.pollEvent(event)) {
    ImGui::SFML::ProcessEvent(event);

    if (event.type == sf::Event::Closed) {
      window.close();
    }

    if (event.type == sf::Event::KeyPressed) {
      if (event.key.code == sf::Keyboard::Escape) {
        window.close();
      }
    }
  }
  // Stupid camera controller
  Affine tf  = Affine::Identity();
  bool moved = false;
  // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
  //   tf.rotate(AngAx(-ANG_SPEED, Vec3::UnitX()));
  //   moved = true;
  // }
  // if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
  //   tf.rotate(AngAx(ANG_SPEED, Vec3::UnitX()));
  //   moved = true;
  // }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
    tf.rotate(AngAx(-ANG_SPEED * ctx.dtime, Vec3::UnitY()));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
    tf.rotate(AngAx(ANG_SPEED * ctx.dtime, Vec3::UnitY()));
    moved = true;
  }

  if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
    tf.translate(Vec3(-LIN_SPEED * ctx.dtime, 0, 0));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
    tf.translate(Vec3(LIN_SPEED * ctx.dtime, 0, 0));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
    tf.translate(Vec3(0, 0, LIN_SPEED * ctx.dtime));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
    tf.translate(Vec3(0, 0, -LIN_SPEED * ctx.dtime));
    moved = true;
  }

  if (moved) {
    scene.moveCamera(tf, ctx);
  }
}