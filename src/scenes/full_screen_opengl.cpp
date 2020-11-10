#include "full_screen_opengl.h"

// #include <cuda_gl_interop.h>

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
  width  = window.getSize().x;
  height = window.getSize().y;
  glBufferData(GL_ARRAY_BUFFER, width * height * sizeof(Pixel), 0, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // cudaGraphicsGLRegisterBuffer(&cudaVBO_, glVBO_, cudaGraphicsMapFlagsNone);

  spdlog::debug("VBO created [{} {}]", width, height);
  screenBuffer_.resize(width * height);
  for (unsigned int row = 0; row < height; ++row) {
    for (unsigned int col = 0; col < width; ++col) {
      auto idx = row * width + col;
      screenBuffer_[idx].xy << col, row;
      screenBuffer_[idx].color << col * 255 / width, row * 255 / height, 0, 255;
    }
  }

  initScene();
}

FullScreenOpenGLScene::~FullScreenOpenGLScene() { glDeleteBuffers(1, &glVBO_); }

void FullScreenOpenGLScene::update([[maybe_unused]] AppContext &ctx) {
  // CUDA_CALL(cudaGraphicsMapResources(1, &cudaVBO_, 0));
  // size_t num_bytes;
  // CUDA_CALL(cudaGraphicsResourceGetMappedPointer((void **)&vboPtr_, &num_bytes, cudaVBO_));
  // renderCuda();
  // CUDA_CALL(cudaGraphicsUnmapResources(1, &cudaVBO_, 0));
  pt_.render(scene_, cam_, ctx, screenBuffer_);

  glBindBuffer(GL_ARRAY_BUFFER, glVBO_);
  glBufferData(GL_ARRAY_BUFFER, screenBuffer_.size() * sizeof(Pixel), screenBuffer_.data(),
               GL_DYNAMIC_DRAW);
}

void FullScreenOpenGLScene::render(sf::RenderWindow &window) {
  window.pushGLStates();

  glClearColor(0.2f, 0.0f, 0.0f, 0.0f);
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
  glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid *)8);

  glDrawArrays(GL_POINTS, 0, width * height);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  window.popGLStates();
}

void FullScreenOpenGLScene::initScene() {
  cam_.w = width;
  cam_.h = height;
  cam_.pos << 0, 0, 14;
  cam_.fov = R_PI * 0.45;
  cam_.dir = Vec3(0.f, 0.f, -1.f).normalized();

  Material whiteLight{{0, 0, 0}, {1, 1, 1}};
  Material yellowLight{{0, 0, 0}, {2, 1.5, 1}};

  Material white{{1, 1, 1}};
  Material red{{1, .2, .2}};
  Material blue{{.5, .5, 1}};
  Material green{{.2, 1., .2}};

  const float wallR = 1e4f;
  const float roomR = 4.f;
  const float wallD = wallR + roomR;

  scene_.objects.push_back({Sphere{1.f}, whiteLight, {-1, -roomR + 1.f, -1}});
  scene_.objects.push_back({Sphere{2.f}, white, {2, -roomR + 2.f, -2}});

  scene_.objects.push_back({Disc{-Vec3::UnitY(), 2.f}, yellowLight, {0, roomR * 0.99f, 2.f}});

  scene_.objects.push_back({Plane{Vec3::UnitY()}, white, -Vec3::UnitY() * roomR});
  scene_.objects.push_back({Plane{-Vec3::UnitY()}, white, Vec3::UnitY() * roomR});

  scene_.objects.push_back({Plane{-Vec3::UnitX()}, red, Vec3::UnitX() * roomR});
  scene_.objects.push_back({Plane{Vec3::UnitX()}, green, -Vec3::UnitX() * roomR});

  scene_.objects.push_back({Plane{Vec3::UnitZ()}, blue, -Vec3::UnitZ() * roomR});

  pt_.reset(cam_);
}

void FullScreenOpenGLScene::resetBuffer(AppContext &ctx) {
  std::fill(pt_.radianceBuffer.begin(), pt_.radianceBuffer.end(), Radiance::Zero());
  ctx.frame = 0;
}