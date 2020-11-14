#include "common.h"

#include "imgui_helpers.hpp"
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>

// #include "cuda_wrapper.h"
#include "options.h"
#include "scenes/full_screen_opengl.h"

#define LIN_SPEED (float)0.2
#define ANG_SPEED (float)0.02

void handleEvents(sf::RenderWindow &window, FullScreenOpenGLScene &scene, AppContext &ctx);

int main(int argc, const char **argv) {
  Options opt({std::next(argv), std::next(argv, argc)});
  opt.checkOptions();

  spdlog::info("Starting application");

  // int gpuId = cuda::findBestDevice();
  // cuda::gpuDeviceInit(gpuId);

  sf::RenderWindow window(sf::VideoMode(opt.width, opt.height), "SFML + CUDA",
                          sf::Style::Titlebar | sf::Style::Close);
  ImGui::SFML::Init(window);
  window.setFramerateLimit(144);
  spdlog::info("SFML window created");

  FullScreenOpenGLScene scene(window);

  AppContext ctx;
  sf::Clock deltaClock;
  while (window.isOpen()) {
    ImGui::SFML::Update(window, deltaClock.restart());

    scene.update(ctx);

    ImGui::Begin("Stats");
    ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));
    ImGui::Text("%d SPP", static_cast<int>(ctx.spp));
    ImGui::Text("%.3f s per SPP", static_cast<double>(ctx.elapsed_seconds));
    ImGui::Text("%.2f %% error", static_cast<double>(ctx.renderError * 100.f));
    ImGui::End();

    ImGui::Begin("Control");
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("Display mode", &ctx.mode)) {
      scene.resetBuffer(ctx);
    }
    if (ctx.mode == DisplayMode::Depth) {
      if (ImGui::SliderFloat("Far plane", &ctx.far_plane, 0.0f, 30.0f)) {
        scene.resetBuffer(ctx);
      }
    }
    ImGui::End();

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
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
    tf.translate(Vec3(-LIN_SPEED, 0, 0));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
    tf.translate(Vec3(LIN_SPEED, 0, 0));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
    tf.translate(Vec3(0, 0, LIN_SPEED));
    moved = true;
  }
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
    tf.translate(Vec3(0, 0, -LIN_SPEED));
    moved = true;
  }
  if (moved) {
    scene.moveCamera(tf, ctx);
  }
}