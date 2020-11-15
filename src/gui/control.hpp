#include "../common.h"
#include "../scenes/full_screen_opengl.h"

#include <imgui-SFML.h>
#include <imgui.h>

class ControlGUI {
public:
  ControlGUI(FullScreenOpenGLScene &scene) : scene_(&scene) {}
  void draw(AppContext &ctx) {
    ImGui::Begin("Control");
    ImGui::SetNextItemWidth(100);
    if (ImGui::Combo("Display mode", &ctx.mode)) {
      scene_->resetBuffer(ctx);
    }
    if (ctx.mode == DisplayMode::Depth) {
      if (ImGui::SliderFloat("Far plane", &ctx.far_plane, 0.0f, 30.0f)) {
        scene_->resetBuffer(ctx);
      }
    }
    ImGuiTreeNodeFlags leafFlags =
        ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_Bullet;
    {
      bool node_open = ImGui::TreeNodeEx("Per frame options", ImGuiTreeNodeFlags_DefaultOpen);
      if (node_open) {
        ImGui::Columns(2);
        ImGui::PushID(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("Samples", leafFlags);
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputInt("##value", &ctx.samples)) {
          ctx.samples = std::max(1, ctx.samples);
        }
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(1);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("Bounces", leafFlags);
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputInt("##value", &ctx.max_depth)) {
          ctx.max_depth = std::max(1, ctx.max_depth);
          scene_->resetBuffer(ctx);
        }
        ImGui::NextColumn();
        ImGui::Columns(1);
        ImGui::PopID();
      }
      ImGui::TreePop();
    }
    {
      bool node_open = ImGui::TreeNodeEx("Compositing", ImGuiTreeNodeFlags_DefaultOpen);
      if (node_open) {
        ImGui::Columns(2);
        ImGui::PushID(0);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("Exposure", leafFlags);
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##value", &ctx.exposure, 0.1f);
        ImGui::NextColumn();
        ImGui::PopID();

        ImGui::PushID(1);
        ImGui::AlignTextToFramePadding();
        ImGui::TreeNodeEx("Gamma", leafFlags);
        ImGui::NextColumn();
        ImGui::SetNextItemWidth(-1);
        ImGui::DragFloat("##value", &ctx.gamma, 0.1f);
        ImGui::NextColumn();
        ImGui::Columns(1);
        ImGui::PopID();
      }
      ImGui::TreePop();
    }
    ImGui::End();
  }

private:
  FullScreenOpenGLScene *scene_;
};