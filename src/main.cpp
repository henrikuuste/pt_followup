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
#include "gui/playerController.hpp"

static constexpr float LIN_SPEED = 8.0f;
static constexpr float ANG_SPEED = 2.0f;

void handleEvents(sf::RenderWindow &window, AppContext &ctx, PlayerController &playerControl);

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
  PlayerController playerControl(scene);

  AppContext ctx;
  sf::Clock deltaClock;

  scene.run(ctx);
  while (window.isOpen()) {
    ctx.dtime = deltaClock.getElapsedTime().asSeconds();
    ImGui::SFML::Update(window, deltaClock.restart());

    statsGUI.draw(ctx);
    controlGUI.draw(ctx);

    window.clear();
    scene.render(window);
    ImGui::SFML::Render(window);
    window.display();

    handleEvents(window, ctx, playerControl);
  }

  spdlog::info("Shutting down");
  ImGui::SFML::Shutdown();

  return 0;
}

void handleEvents(sf::RenderWindow &window, AppContext &ctx, PlayerController &playerControl) {
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

    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        // TODO mouse events
        playerControl.handleEvent(event, window);
    }
  }

  playerControl.update(ctx);
}