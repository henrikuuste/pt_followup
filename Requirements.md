# MVP for hackathon
* Given a scene, and a camera, simulate light transport (realistically) to an error of < 1%
* How do we get the scene?
  * [x] Hardcode
* What is in the scene?
  * [x] Sphere at first
  * [ ] Support for others
    * [x] plane
    * [ ] cube
    * [ ] cylinder
* Materials?
  * [x] Light emitter
  * [x] Lambertian diffuse
  * [ ] Support for
    * [x] mirror
    * [ ] refractive
  * [ ] Support for textures
* What kind of camera?
  * [x] Pinhole
* Where and how to output?
  * [x] Screen
  * [x] RGB -> sRGB
  * Resolution
    * [ ] max 1920x1080
* GUI
  * [x] SPP
  * [x] Time per frame
  * [ ] Exposure control
  * [ ] Enable-disable features
* Interactive?
  * No - for MVP
  * Support for future
  * [ ] Camera and object properties
    * [ ] Live movement through GUI
  * No rigid-body physics

# Non-functional
* Performance
  * [x] Let's use less than 1 GB of RAM and VRAM
  * [ ] CPU usage < 100%
  * [ ] Possibility use accelerators
  * [x] Don't kill GUI thread
* Timing
  * [x] We want to see something in at least 1 second
  * [ ] Final result (< 1% error) in less than 5 minutes
* Environment
  * [x] Linux, portable
  * [ ] My laptop
* Testability
  * [x] Debug display modes
* Scalability
  * Distributed?
    * No, not planned
  * Not scalable to new features

# System and component requirements
* Features
  * [x] Pure monte-carlo path tracing
    * [ ] Support for next event estimation
  * [x] Multi-threading
  * [x] GUI
  * [ ] Scene manager
  * [ ] Camera controller
  * [ ] Material library
  * [ ] Modular pipeline
* Data models
  * [x] Polymorphic Object
  * [x] Linear algebra data/objects
  * Access patterns
    * Access one pixel depth first
    * Access the whole scene for each pixel for each interaction
    * Calculation in material
  * [x] Camera
  * [x] Global configuration block
  * [ ] Psedorandom generator
    * Thread local
* Technologies/frameworks
  * C++
  * CUDA
  * SFML
  * ImGUI
  * Eigen