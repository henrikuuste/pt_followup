#pragma once

#include "../common.h"
#include "../cuda_memory.hpp"
#include "pt.h"
#include <GL/glew.h>

#include <SFML/Graphics/RenderWindow.hpp>

#include <functional>
#include <future>

enum class PTState {
  IDLE,
  RUNNING,
  STOP,
};

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
  void mapVBO();
  void unmapVBO();

  std::atomic<PTState> ptState_ = PTState::IDLE;
  std::future<void> runPTHandle_;
  unsigned int width, height;

  std::mutex sceneMutex_;
  std::mutex swapMutex_;
  std::atomic_bool vboMapped_ = false;
  GLuint glVBO_[3];
  cudaGraphicsResource_t cudaVBO_[3];
  cuda::raw_ptr<Pixel> vboPtr_[3];
  int drawVBO_      = 0;
  int renderVBO_    = 1;
  int availableVBO_ = 2;
  PathTracer pt_;
  Scene scene_;
  Camera cam_;
};