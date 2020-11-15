#include "full_screen_opengl.h"

#include <chrono>
#include <cuda_gl_interop.h>

using Time = std::chrono::steady_clock;
using Fsec = std::chrono::duration<float>;

FullScreenOpenGLScene::FullScreenOpenGLScene(sf::RenderWindow const &window) {
  glewInit();
  if (!glewIsSupported("GL_VERSION_2_0 ")) {
    spdlog::error("Support for necessary OpenGL extensions missing.");
    abort();
  }
  spdlog::info("OpenGL initialized");

  glGenBuffers(3, glVBO_);

  // initialize VBO
  width  = static_cast<unsigned int>(static_cast<float>(window.getSize().x) * 0.6f);
  height = window.getSize().y;

  glBindBuffer(GL_ARRAY_BUFFER, glVBO_[0]);
  glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(Pixel), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, glVBO_[1]);
  glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(Pixel), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, glVBO_[2]);
  glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(Pixel), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  cudaGraphicsGLRegisterBuffer(&cudaVBO_[0], glVBO_[0], cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer(&cudaVBO_[1], glVBO_[1], cudaGraphicsMapFlagsNone);
  cudaGraphicsGLRegisterBuffer(&cudaVBO_[2], glVBO_[2], cudaGraphicsMapFlagsNone);

  spdlog::debug("VBO created [{} {}]", width, height);

  initScene();
}

FullScreenOpenGLScene::~FullScreenOpenGLScene() {
  if (runPTHandle_.valid()) {
    ptState_ = PTState::STOP;
    runPTHandle_.wait();
  }
  CUDA_CALL(cudaDeviceSynchronize());
  cudaGraphicsUnregisterResource(cudaVBO_[0]);
  cudaGraphicsUnregisterResource(cudaVBO_[1]);
  cudaGraphicsUnregisterResource(cudaVBO_[2]);
  glDeleteBuffers(3, glVBO_);
}

void FullScreenOpenGLScene::mapVBO() {
  if (vboMapped_.load())
    return;
  CUDA_CALL(cudaGraphicsMapResources(3, cudaVBO_, 0));
  size_t num_bytes;
  CUDA_CALL(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&vboPtr_[0]), &num_bytes,
                                                 cudaVBO_[0]));
  CUDA_CALL(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&vboPtr_[1]), &num_bytes,
                                                 cudaVBO_[1]));
  CUDA_CALL(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&vboPtr_[2]), &num_bytes,
                                                 cudaVBO_[2]));
  vboMapped_ = true;
}

void FullScreenOpenGLScene::unmapVBO() {
  if (!vboMapped_.load())
    return;
  CUDA_CALL(cudaGraphicsUnmapResources(3, cudaVBO_, 0));
  vboMapped_ = false;
}

void FullScreenOpenGLScene::update([[maybe_unused]] AppContext &ctx) {
  mapVBO();
  if (ptState_.load() == PTState::IDLE) {
    ptState_     = PTState::RUNNING;
    runPTHandle_ = std::async(std::launch::async, [&]() {
      while (ptState_.load() == PTState::RUNNING) {
        {
          std::unique_lock lk(swapMutex_);
          std::swap(renderVBO_, availableVBO_);
        }
        if (ctx.request_reset) {
          ctx.spp           = 0;
          ctx.request_reset = false;
        }
        auto start = Time::now();
        pt_.renderCuda(scene_, cam_, sceneMutex_, ctx, vboPtr_[renderVBO_]);
        auto finish         = Time::now();
        ctx.elapsed_seconds = Fsec{finish - start}.count();
        ctx.spp++;
      }
    });
  }
}

void FullScreenOpenGLScene::render(sf::RenderWindow &window) {
  window.pushGLStates();

  glClearColor(0.1f, 0.1f, 0.1f, 0.0f);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, static_cast<GLdouble>(window.getSize().x), 0.0,
          static_cast<GLdouble>(window.getSize().y), -1, 1);

  glClear(GL_COLOR_BUFFER_BIT);
  glDisable(GL_DEPTH_TEST);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  {
    std::unique_lock lk(swapMutex_);
    std::swap(drawVBO_, availableVBO_);
  }

  glBindBuffer(GL_ARRAY_BUFFER, glVBO_[drawVBO_]);
  glVertexPointer(2, GL_FLOAT, 12, 0);
  glColorPointer(4, GL_UNSIGNED_BYTE, 12, reinterpret_cast<GLvoid *>(8));

  glDrawArrays(GL_POINTS, 0, static_cast<int>(width * height));
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  window.popGLStates();
}

void FullScreenOpenGLScene::initScene() {
  const float roomR = 6.f;

  cam_.w = static_cast<float>(width);
  cam_.h = static_cast<float>(height);
  cam_.tr.translation() << 0.f, -roomR + 4.f, 14.f;
  cam_.tr.rotate(AngAx(R_PI, Vec3::UnitY()));
  // cam_.tr.rotate(AngAx(R_PI * .05f, -Vec3::UnitZ()));
  cam_.fov = R_PI * 0.4f;

  Material whiteLight{Vec3::Zero(), {1.f, 1.f, 1.f}};
  Material yellowLight{Vec3::Zero(), {2.f, 1.5f, 1.f}};

  Material white{{1.f, 1.f, 1.f}};
  // Material red{{1.f, .2f, .2f}};
  Material blue{{.5f, .5f, 1.f}};
  Material green{{.2f, 1.f, .2f}};

  Material greenRefl{{.2f, 1.f, .2f}, Vec3::Zero(), Material::SPEC};
  Material redRefl{{1.f, .2f, .2f}, Vec3::Zero(), Material::SPEC};
  // Material whiteRefl{{1.f, 1.f, 1.f}, Vec3::Zero(), Material::SPEC};

  scene_.objects.reserve(16);

  Affine tr = Affine::Identity();

  tr.translation() << 0.f, -roomR + 1.f, -0.5f;
  tr.scale(1.f);
  scene_.objects.push_back({"light sphere", Sphere{1.f}, whiteLight, tr});
  tr.setIdentity();
  tr.translation() << 2.f, -roomR + 2.f, -2.f;
  scene_.objects.push_back({"green reflective sphere", Sphere{2.f}, greenRefl, tr});
  tr.setIdentity();
  tr.translation() << -2.f, -roomR + 2.f, -2.f;
  scene_.objects.push_back({"white sphere", Sphere{2.f}, white, tr});

  tr.translation() << -Vec3::UnitY() * roomR;
  scene_.objects.push_back({"floor", Plane{Vec3::UnitY()}, white, tr});
  tr.translation() << Vec3::UnitY() * roomR;
  scene_.objects.push_back({"ceiling", Plane{-Vec3::UnitY()}, white, tr});

  tr.rotate(AngAx(R_PI * .1f, Vec3::UnitZ()));
  tr.translation() << Vec3::UnitX() * roomR;
  scene_.objects.push_back({"left wall", Plane{-Vec3::UnitX()}, redRefl, tr});
  tr.setIdentity();
  tr.rotate(AngAx(-R_PI * .1f, Vec3::UnitZ()));
  tr.translation() << -Vec3::UnitX() * roomR;
  scene_.objects.push_back({"right wall", Plane{Vec3::UnitX()}, green, tr});
  tr.setIdentity();

  tr.translation() << -Vec3::UnitZ() * roomR;
  scene_.objects.push_back({"back wall", Plane{Vec3::UnitZ()}, blue, tr});

  tr.translation() << Vec3::UnitZ() * roomR * 4;
  scene_.objects.push_back({"Z+ wall", Plane{-Vec3::UnitZ()}, blue, tr});

  tr.translation() << 0, roomR * 0.99f, 2.f;
  tr.scale(Vec3{1.f, 1.f, roomR * 0.8f});
  scene_.objects.push_back({"ceiling light", Disc{-Vec3::UnitY(), 2.f}, yellowLight, tr});

  pt_.reset(cam_);
}

void FullScreenOpenGLScene::resetBuffer(AppContext &ctx) {
  pt_.reset(cam_);
  ctx.request_reset = true;
}

void FullScreenOpenGLScene::setCameraTf(Affine const &tf, AppContext &ctx) {
  std::unique_lock lk(sceneMutex_);
  cam_.setCameraTf(tf);
  resetBuffer(ctx);
}