#include "common.h"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include "imgui_helpers.hpp"

// #include "cuda_wrapper.h"
#include "options.h"
#include "scenes/full_screen_opengl.h"

void handleEvents(sf::RenderWindow &window);

int main(int argc, const char **argv) {
  Options opt({std::next(argv), std::next(argv, argc)});
  opt.checkOptions();

  spdlog::info("Starting application");

  // int gpuId = cuda::findBestDevice();
  // cuda::gpuDeviceInit(gpuId);

  sf::RenderWindow window(sf::VideoMode(opt.width, opt.height), "SFML + CUDA",
                          sf::Style::Titlebar | sf::Style::Close);
  ImGui::SFML::Init(window);
  spdlog::info("SFML window created");

  FullScreenOpenGLScene scene(window);

  constexpr auto &modeNames = magic_enum::enum_names<DisplayMode>();

  AppContext ctx;
  sf::Clock deltaClock;
  while (window.isOpen()) {
    ImGui::SFML::Update(window, deltaClock.restart());

    scene.update(ctx);

    ImGui::Begin("Stats");
    ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
    ImGui::Text("%d SPP", ctx.frame);
    ImGui::Text("%.3f s per SPP", ctx.elapsed_seconds);
    ImGui::Text("%.2f %% error", ctx.renderError * 100.f);
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

    handleEvents(window);
    ctx.dtime = deltaClock.getElapsedTime().asSeconds();
    ctx.frame++;
  }

  spdlog::info("Shutting down");
  ImGui::SFML::Shutdown();

  return 0;
}

void handleEvents(sf::RenderWindow &window) {
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
}