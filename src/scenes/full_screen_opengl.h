#pragma once

#include "../common.h"
#include "../cuda_memory.hpp"
#include "pt.h"
#include <GL/glew.h>

#include <SFML/Graphics/RenderWindow.hpp>

#include <functional>
#include <future>

class FullScreenOpenGLScene {
public:
  FullScreenOpenGLScene(sf::RenderWindow const &window);
  ~FullScreenOpenGLScene();

  void update(AppContext &ctx);
  void render(sf::RenderWindow &window);
  void resetBuffer(AppContext &ctx);
  void moveCamera(Affine const &tf, AppContext &ctx);

private:
  void initScene();
  std::future<void> runPTHandle;
  unsigned int width, height;

  GLuint glVBO_;
  cudaGraphicsResource_t cudaVBO_;
  cuda::raw_ptr<Pixel> vboPtr_;
  PathTracer pt_;
  Scene scene_;
  Camera cam_;
  std::atomic_bool renderingPT = false;
};