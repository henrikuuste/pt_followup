#pragma once
#include "../common.h"

#include <imgui-SFML.h>
#include <imgui.h>

class StatsGUI {
public:
  void draw(AppContext &ctx) {
    ImGui::Begin("Stats");
    ImGui::Text("%.1f FPS", static_cast<double>(ImGui::GetIO().Framerate));
    ImGui::Text("%d SPP", static_cast<int>(ctx.spp));
    ImGui::Text("%.1f ms per SPP", static_cast<double>(ctx.elapsed_seconds * 1e3f));
    ImGui::Text("%.2f %% error", static_cast<double>(ctx.renderError * 100.f));
    ImGui::End();
  }
};