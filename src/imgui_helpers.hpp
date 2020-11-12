#pragma once

#include <imgui-SFML.h>
#include <imgui.h>
#include <magic_enum.hpp>
#include <type_traits>
#include <vector>
#include <array>
#include <string_view>

template <class EnumT> requires std::is_enum_v<EnumT> auto enum_as_pointer(EnumT &e) {
  return reinterpret_cast<typename std::underlying_type<EnumT>::type *>(&e);
}

namespace ImGui {
static auto vector_getter = [](void *vec, int idx, const char **out_text) {
  auto &vector = *static_cast<std::vector<std::string> *>(vec);
  if (idx < 0 || idx >= static_cast<int>(vector.size())) {
    return false;
  }
  *out_text = vector.at(idx).c_str();
  return true;
};

bool Combo(std::string_view label, int *currIndex, std::vector<std::string> &values) {
  if (values.empty()) {
    return false;
  }
  return Combo(label.data(), currIndex, vector_getter, static_cast<void *>(&values), values.size());
}

template <typename ArrType>
static auto view_array_getter = [](void *vec, int idx, const char **out_text) {
  auto &arr = *static_cast<std::remove_reference_t<ArrType> *>(vec);
  if (idx < 0 || idx >= static_cast<int>(arr.size())) {
    return false;
  }
  *out_text = arr.at(idx).data();
  return true;
};

template <class EnumT> requires std::is_enum_v<EnumT> bool Combo(std::string_view label, EnumT *v) {
  static constexpr auto &names = magic_enum::enum_names<EnumT>();
  return Combo(label.data(), enum_as_pointer(*v), view_array_getter<decltype(names)>,
               static_cast<void *>(const_cast<std::string_view *>(names.data())),
               magic_enum::enum_count<EnumT>());
}
} // namespace ImGui