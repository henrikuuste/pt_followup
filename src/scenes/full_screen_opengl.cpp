#include "full_screen_opengl.h"

#include <chrono>
// #include <cuda_gl_interop.h>

using Time = std::chrono::steady_clock;
using Fsec = std::chrono::duration<float>;

FullScreenOpenGLScene::FullScreenOpenGLScene(sf::RenderWindow const &window) {
  glewInit();
  if (!glewIsSupported("GL_VERSION_2_0 ")) {
    spdlog::error("Support for necessary OpenGL extensions missing.");
    abort();
  }
  spdlog::info("OpenGL initialized");

  glGenBuffers(1, &glVBO_);
  glBindBuffer(GL_ARRAY_BUFFER, glVBO_);

  // initialize VBO
  width  = static_cast<unsigned int>(static_cast<float>(window.getSize().x) * 0.6f);
  height = window.getSize().y;
  glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(Pixel), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // cudaGraphicsGLRegisterBuffer(&cudaVBO_, glVBO_, cudaGraphicsMapFlagsNone);

  spdlog::debug("VBO created [{} {}]", width, height);
  screenBuffer_.resize(width * height);
  for (unsigned int row = 0; row < height; ++row) {
    for (unsigned int col = 0; col < width; ++col) {
      auto idx = row * width + col;
      screenBuffer_[idx].xy << static_cast<float>(col), static_cast<float>(row);
      screenBuffer_[idx].color << static_cast<uint8_t>(col * 255 / width),
          static_cast<uint8_t>(row * 255 / height), 0, 255;
    }
  }

  initScene();
}

FullScreenOpenGLScene::~FullScreenOpenGLScene() {
  runPTHandle.wait();
  glDeleteBuffers(1, &glVBO_);
}

void FullScreenOpenGLScene::update([[maybe_unused]] AppContext &ctx) {
  // CUDA_CALL(cudaGraphicsMapResources(1, &cudaVBO_, 0));
  // size_t num_bytes;
  // CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void **)&vboPtr_, &num_bytes, cudaVBO_));
  // renderCuda();
  // CUDA_CALL(cudaGraphicsUnmapResources(1, &cudaVBO_, 0));
  if (not renderingPT.load()) {
    glBindBuffer(GL_ARRAY_BUFFER, glVBO_);
    glBufferData(GL_ARRAY_BUFFER, screenBuffer_.size() * sizeof(Pixel), screenBuffer_.data(),
                 GL_DYNAMIC_DRAW);
    renderingPT = true;
    runPTHandle = std::async(std::launch::async, [&]() {
      auto start = Time::now();
      pt_.render(scene_, cam_, ctx, screenBuffer_);
      auto finish         = Time::now();
      ctx.elapsed_seconds = Fsec{finish - start}.count();
      ctx.spp++;
      renderingPT = false;
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
  glBindBuffer(GL_ARRAY_BUFFER, glVBO_);
  glVertexPointer(2, GL_FLOAT, 12, 0);
  glColorPointer(4, GL_UNSIGNED_BYTE, 12, reinterpret_cast<GLvoid *>(8));

  glDrawArrays(GL_POINTS, 0, static_cast<int>(width * height));
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  window.popGLStates();
}

void FullScreenOpenGLScene::initScene() {
  cam_.w = static_cast<float>(width);
  cam_.h = static_cast<float>(height);
  cam_.tr.translation() << 0.f, 0.f, 14.f;
  cam_.tr.rotate(AngAx(R_PI, Vec3::UnitY()));
  cam_.tr.rotate(AngAx(R_PI * .05f, -Vec3::UnitZ()));
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

  const float roomR = 4.f;

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
  ctx.spp = 0;
}

void FullScreenOpenGLScene::moveCamera(Affine const &tf, AppContext &ctx) {
  cam_.moveCamera(tf);
  resetBuffer(ctx);
}