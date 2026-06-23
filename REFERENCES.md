# References & prior art

Sensing is the hard problem: RGB background subtraction fails under a bright dynamic projection. Go IR or depth (both projection-immune). Calibration, pose-stabilization and the water render sit outside the tracking algorithm.

## Closest end-to-end prior art (AR Sandbox family)

- thomwolf/Magic-Sand — https://github.com/thomwolf/Magic-Sand (Windows, depth + projector + reactive water; auto procam calibration)

- KeckCAVES/SARndbox — https://github.com/KeckCAVES/SARndbox (GPU shallow-water; GPL → reimplement)

- cgre-aachen/open_AR_Sandbox — https://github.com/cgre-aachen/open_AR_Sandbox (Python/OpenCV)

## Sensing under projection

- IR survey — https://arxiv.org/pdf/2512.05071

- Community Core Vision (IR blob tracker) — https://github.com/nuigroup/ccv15

- Depth: Open3D plane+cluster — https://learngeodata.eu/learn-3d-point-cloud-segmentation-with-python/

- RealSense IR projector note — https://dev.realsenseai.com/docs/projectors/ (active-stereo tolerates IR; structured-light breaks)

## Projector↔camera calibration

- kamino410/procam-calibration (MIT, Python) — https://github.com/kamino410/procam-calibration

- mehrab2603/scan3d-capture (BSD-3) — https://github.com/mehrab2603/scan3d-capture

## Pose / rotation (downstream hardening)

- Kabsch-Cookbook (MIT) — https://github.com/hunter-heidenreich/Kabsch-Cookbook

- OpenCV PCA orientation — https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html

## Transport

- TUIO 2.0 schema (template) — https://www.tuio.org/?specification

- python-tuio (MIT) — https://github.com/tweigel-dev/python-tuio

- TUIO simulator (test render without CV) — https://github.com/mkalten/TUIO11_Simulator

## Water reacting to obstacles

- TouchDesigner: touchFluid (MIT) — https://github.com/kamindustries/touchFluid · WaveTOP — https://derivative.ca/community-post/asset/wavetop/67796

- Unreal: Niagara Fluids — https://80.lv/articles/working-with-niagara-fluids-to-create-water-simulations · UDP-Unreal (MIT) — https://github.com/getnamo/UDP-Unreal

- Web: WebFlood — https://github.com/aeplay/WebFlood · osc-js UDP↔WebSocket — https://github.com/adzialocha/osc-js

- SDF flow-around: Jump Flooding — https://blog.demofox.org/2016/02/29/fast-voronoi-diagrams-and-distance-dield-textures-on-the-gpu-with-the-jump-flooding-algorithm/

## Art-direction references

- teamLab — Universe of Water Particles on a Rock — https://www.teamlab.art/ew/iwa-waterparticles/

- Tellart — Terraform Table — https://www.dezeen.com/2018/06/01/tellart-terraform-table-topographic-sandpit-move-mountains-technology/

Note: AR-Sandbox family is GPL — reuse techniques, don't embed source. Ship MIT/BSD/public-domain pieces.
