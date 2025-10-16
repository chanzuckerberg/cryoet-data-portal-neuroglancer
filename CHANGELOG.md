# Changelog

## [1.7.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.6.1...v1.7.0) (2025-10-06)


### ✨ Features

* allow colors to be dict of seg label to color str in seg mask gen ([1a21dc4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1a21dc41e64f379ecf3d65cce96f54d327eff0f4))
* allow colors to be dict of seg label to color str in seg mask gen ([#71](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/71)) ([bf64fdb](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/bf64fdbbc31a067739950ebf2093f52ab340ea4e))
* replace floating point error checking with option to round ([457f42c](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/457f42c5a387ab9b65a6ed7918bbb9365f7d725d))


### 🐞 Bug Fixes

* ensure segmentation data is always int ([ee6c732](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ee6c732c9dea04b6a675721678f4115c8765c089))
* ensure segmentation data is always int ([#70](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/70)) ([3fe087e](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3fe087e96ae0c4afd53932456254fdf8886affb5))


### 🧹 Miscellaneous Chores

* CCIE-4984 conform to open sourcing guidelines ([8e957a7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/8e957a7c112748499479bd0deb5521c90b360e1a))
* CCIE-4984 conform to open sourcing guidelines ([#68](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/68)) ([399d97a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/399d97aa81f5ef4ed23788e19fb6b45d4310d223))
* fix docstring ([b654d5a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b654d5a2b2f06be01f9644024849010000c27517))
* fix linting issue ([181af5f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/181af5f1da601f469e74d19e220c333a5dc56f73))


### ♻️ Code Refactoring

* clearer check for float data ([60302b4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/60302b4a06095d857fcc4785860c86cc192ca39d))
* use da equivalent of np fns for efficiency ([3ec4fa8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3ec4fa86d1bdd42cc3759cc005155939745a6f13))

## [1.6.1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.6.0...v1.6.1) (2025-06-27)


### 🐞 Bug Fixes

* match name to portal name of tool palette ([61747a6](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/61747a6c77fdda2d8357a090d9877b643e0da634))
* match name to portal name of tool palette ([#66](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/66)) ([35e1d14](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/35e1d1465b1e704d3c92594c0659ad81a85069cf))
* **test:** also update name in test ([8f327f7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/8f327f7fb888f6af66cb0cc1090d6fdf6288e4df))

## [1.6.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.5.0...v1.6.0) (2025-06-26)


### ✨ Features

* Add dimension selector tool palette at bottom of screen ([#35](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/35)) ([92347f0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/92347f052b565b629f30d72e366afa0bf70b19c1))
* update igneous to not rely on custom patch ([#65](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/65)) ([cee5aa7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/cee5aa70782877a4cd069e876b7b459d1fefaee3))
* update state gen for dim widget to match ts ([dd205ce](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/dd205ce33aa3547c96d687f3f64b1b33fba6ebce))


### 📝 Documentation

* update examples with http-server for meshes ([60590e8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/60590e80d3f6936f9e449caf054b667d8cfdd7bd))


### 🧹 Miscellaneous Chores

* don't depend on custom igneous patch anymore ([742048b](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/742048be07cc1de55b671b3b431eae90156553bb))

## [1.5.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.4.2...v1.5.0) (2025-06-11)


### ✨ Features

* support different output dim to input dim ([b0a35f8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b0a35f8224b43bedd75bff5ad874a5c5f0b4e50e))
* support different output dim to input dim ([#62](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/62)) ([306707a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/306707a7cda75cf0904cade50809962191b840bd))
* support scaling output neuroglancer starting state ([1fce87f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1fce87f2970296b1598d81de208a3ad0cec1cb5d))


### 🐞 Bug Fixes

* remove accidental pritn ([3d01b4f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3d01b4f8b39a5a70c3e498e242c986a879faf08c))

## [1.4.2](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.4.1...v1.4.2) (2025-05-26)


### 🐞 Bug Fixes

* compare resolutions with numpy to avoid warnings with 1 != 1.0 ([d586407](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d58640760e83bf1dc9af119c596e58c064021219))
* compare resolutions with numpy to avoid warnings with 1 != 1.0 ([#60](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/60)) ([8e1eee4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/8e1eee45190684081f738638ce0a2a6e27a5d8df))

## [1.4.1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.4.0...v1.4.1) (2025-05-15)


### 🧹 Miscellaneous Chores

* Update LICENSE ([83a850d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/83a850d2e475a2ef820b5ced6d77dd9fc377532c))
* Update LICENSE ([#58](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/58)) ([ab7ff22](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ab7ff22885cc99c4e8edf07ce53f531610af82b7))


### ♻️ Code Refactoring

* clean up resolution comparison ([2fe890f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2fe890f5e3bef8343d22782194c936b09a062764))

## [1.4.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.3.0...v1.4.0) (2025-05-01)


### ✨ Features

* allow control over empty seg blocks ([6105dc4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/6105dc4c871f3ba67ee8b5499bd5f163ae169a3c))
* allow control over empty seg blocks ([#54](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/54)) ([7dfb5f7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7dfb5f7e75b89a4d9ed04a1189a3e1f9a355ea96))


### 🐞 Bug Fixes

* use scene.to_geometry instead of scene.dump(concatenate=True) ([#51](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/51)) ([1b97260](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1b97260c75bba080611eb2fd8cf044f65e4b58e2))


### 📝 Documentation

* add docstrings to public functions that are common ([2effc3f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2effc3f0c6d02d2e9a7996482ab9fdbe637db544))
* add docstrings to public functions that are common ([#55](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/55)) ([a86bd80](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/a86bd807fad7bf553918c5865b78e80c07532ccd))

## [1.3.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.2.3...v1.3.0) (2025-04-01)


### ✨ Features

* set default GPU mem limit to 1.5GB with setting to change ([#50](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/50)) ([5e238d8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/5e238d8cd534dcc105854f1f7214d758b804f6ba))
* set default mem limit to 1.5GB with setting to change ([12e368b](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/12e368b6562897ec8d145982d930ba91f96fca44))

## [1.2.3](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.2.2...v1.2.3) (2025-03-26)


### 🐞 Bug Fixes

* set pyproject package version more strictly ([14912fd](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/14912fdabaee8a91ccdd85ca35e9ac4588cd6280))
* set pyproject package version more strictly ([#47](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/47)) ([9e55750](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/9e55750f30794adbfad4ee0af77458679ab956fa))


### 🧹 Miscellaneous Chores

* fix the kimimaro package version ([58daef2](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/58daef231102cb6b7e8c86458f43d81abf7c8668))

## [1.2.2](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.2.1...v1.2.2) (2025-03-26)


### 🐞 Bug Fixes

* correct bad copy paste of config path from docs ([c60c91f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/c60c91fefead8530003dc881008074293dbc5524))
* manually mark locations for release please to update ([e57d1f1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/e57d1f14797eb40ac76c54cdf9242cb45fba9232))
* manually mark locations for release please to update ([#44](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/44)) ([cfc1e0a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/cfc1e0a0bd2f3c93896fc5399d3a972b388177f9))
* remove {} around release please version instruction ([fe57987](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/fe5798712871896ecb44b2a1ac5b17014e6b5bfb))


### 🧹 Miscellaneous Chores

* change release please to use generic updater ([01b1bf1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/01b1bf12de414048dda010096f2ef162bbb2b26f))
* change release please to use generic updater ([#46](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/46)) ([ed601ea](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ed601ea57f61edc8e8b8c9b4f02adac2fd6f8cdd))

## [1.2.1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.2.0...v1.2.1) (2025-03-21)


### 🐞 Bug Fixes

* rename hide cross section to correct name ([9658a89](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/9658a895aeecaae83a43d6a9d0550307b832aed1))
* rename hide cross section to correct name ([#41](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/41)) ([9f0f423](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/9f0f4239e3308b0dd0dd407d57fa1153cf945e72))

## [1.2.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.1.0...v1.2.0) (2025-03-21)


### ✨ Features

* Add default value to false for code editor display ([#29](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/29)) ([b058255](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b058255e495365012e7de3733feff8f83d971f6a))
* Add layer color sync activation option ([#34](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/34)) ([8d5277d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/8d5277dd88e689852e880ab9225c3eb49dc2f2e5))
* allow new panel type as optional param ([#39](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/39)) ([be2e1c7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/be2e1c7e9a542314a22a85b0ef4aa27b06bf62ef))
* bool flag to switch between 4panel (old - preferred) and 4panel-alt (new) ([1ee9b80](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1ee9b80ba07a09e8731250c821aae439ff7369e7))
* Change the default value to activate auto-mode for the volume ([3b92523](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3b92523ff7c6cf07e69147b01ac3de2baf22b7d8))
* Change the way the has volume shadder rendering condition is computed for Image state generation ([#40](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/40)) ([70fd159](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/70fd159b3dd2e571535c59d1759f66317e827185))
* Enable color legend by default ([41b2f0b](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/41b2f0bee99be344d2f55c1c5121e8f0ea81d8e8))
* hide 3D cross section background setting ([0673c57](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/0673c57bed9229f3abcf2a2730463b861a7abb30))
* hide 3D cross section background setting ([#36](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/36)) ([71c8aff](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/71c8aff8725210583304c51dee90a9f360a21142))
* layout hidden panels properly. This is necessary because otherwise they pop up in odd places ([cd0d183](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/cd0d183d8eac070fb4c008d6819ff5aabf273654))
* move annotation color to defaultColor ([f9b883f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/f9b883f2478da3233e6c6ef693bab5f54c571581))
* top left layer controls, bottom left layer list panel ([cf96b09](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/cf96b09afe3cda4fb79c9443ba123edfd0abd282))
* update default layout of the main neuroglancer landing page ([#32](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/32)) ([8605849](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/86058495bf050fa7e6d83ad24008942e1705fa7b))


### 🐞 Bug Fixes

* Change the way the has volume shadder rendering condition is ([9bff4eb](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/9bff4eb6149dbb82619b41cde6e4c7da16cb4d4f))
* remove accidental print ([3a34dab](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3a34dab2b53e4e06a9aabc7800d3a1a17b9dccd7))
* Update runs-on labels in GitHub Actions workflows ([#37](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/37)) ([7c91df1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7c91df13c39bb9816c6000b46826c9891aa26500))
* Update runs-on to use ARM64 or X64 ([3df2d70](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3df2d70c7b432e594de7c90e2bb80a1635917e00))


### 🧹 Miscellaneous Chores

* Updating poetry to 2.0.0 ([#31](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/31)) ([7dcc2b0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7dcc2b0971e79b4764a808b061d5b8d09cd59849))


### ♻️ Code Refactoring

* clarify test name ([0638c1f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/0638c1fb1ab1c097a771aea0f1392a0b2d47c370))
* remove accidental comment ([a699867](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/a699867acfb42136b89011aa1a486f9a53516cc8))


### 🧪 Tests

* add test for variable parts of the top level state generator ([463a289](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/463a2896af9d2d1161d06cb5799f8cfdd9b72a75))

## [1.1.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.0.0...v1.1.0) (2024-12-06)


### ✨ Features

* add all hyperopts to contrast limits ([1fa9d6a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1fa9d6a1347183ea3b0071f70d5ac93c53b0a776))
* add decimation algorithm for contrast limit computation ([9499967](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/9499967fcb26d58091dbd75c1a1433dff8c4afb7))
* Add default opacity to 1 and blend to additive ([cb32980](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/cb32980268d3947a0dbc54a61bdf9e7a81f06fda))
* Add default opacity to 1 and blend to additive (CC-165) ([#27](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/27)) ([3316c78](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3316c78bd432a217c1ac7069b0dc1d430f685207))
* add hide high noise option ([bf97c93](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/bf97c935ce55f8d850e1ae8c5949be174349790b))
* add initial hyperopt ([05ab24d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/05ab24db76374dae256965c79d5a3de86c7cee8e))
* add optimisation for hyperparams to the contrast limit class w test ([17a311b](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/17a311bac870b1eef81afe904d03d85d8762beee))
* add segmentation property objects ([d7664bc](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d7664bc8fd487fbdab46e727665209df2fba95c6))
* add v1 of GMM and k-means contrast limits ([eda99a9](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/eda99a9cb8de211a596fe9a89ec8e77c0a617848))
* Add volume rendering contrast limits ([#25](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/25)) ([10669d5](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/10669d5a695fc6e8ec11698cd3dde60879020935))
* Adding release please ([#19](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/19)) ([4ec9ca7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/4ec9ca7c0558203bd363d49599c074ade951d905))
* allow seg gen to control the segments shown, have random colours, and the mesh render scale ([759def1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/759def1658cf248fa5b71743e8db11cd466bcc68))
* allow to not persist big zarrs ([62166af](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/62166aff2290f87e898756014884132cedfa9a37))
* better window limits from contrast ([96ece94](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/96ece94e0e9eb4a8813bed0093872510579534cc))
* change print to logging of chunk comparison size ([878a7a7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/878a7a7e0b041fdbe942a11bdbdb8f55b8369691))
* clean up contrast limits ([87e25e5](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/87e25e57ceb26239588716fd629d84287f734894))
* contrast limit improvements ([ff5ef94](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ff5ef9426d193efd09aa5b84fcf0f1ac9ad45cdb))
* datasets for contrast limit testing ([25b7861](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/25b7861f643b253f9b60f4d89f41962ac6738402))
* Disable "pick" by default for segmentation layer. ([#28](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/28)) ([8287be7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/8287be7775de5af2fb0b13bb5fce20e31b1a5b96))
* Disable "pick" for default value in segmentation layer ([3af57f4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/3af57f41dfefe0bce95a58f262515415c644fcd2))
* improve CDF based contrast limits ([822be9f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/822be9fbfa41143d02d7c03032dc2c8f147516cc))
* improve high value hide control ([5d3a8ae](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/5d3a8ae768572d5f5d2c96928ddcf3c82e46029c))
* improve manual limits testing and add auto screenshots ([a24ad8c](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/a24ad8cca1bf580e5583ea91a85632656628bde6))
* include manual test for contrast ([1882a91](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1882a911ee55074fd9d64b23fc4a966842d59317))
* include new contrast limit methods ([75b1a2e](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/75b1a2e2ba1c2d1974fa93535a964484d7637f68))
* incorportate hyperopt in api test ([0003de9](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/0003de9553949348e1fc2735412399243aab6453))
* link seg properties into mesh and seg creation ([dfbd34a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/dfbd34a24e4413cc62baf37251c24456b822e5b8))
* main public API to limits tuning ([71112c9](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/71112c973ab88305b3ba43eabdf8c0f6acadf6ab))
* make public API to contrast limits ([30eaa75](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/30eaa757cb820268fde5061d0b42477291825fea))
* mesh precompute functions and state generation ([#23](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/23)) ([02439d8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/02439d81f96f2f3582c1bec2102bad1c4540efb3))
* more control over image JSON generator ([e15bace](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/e15bacecccb0bac66a0fdad39583129faf1ff581))
* more control over mesh from segmentation ([e1af340](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/e1af340a2844639dadf5f84209082ffeb219662d))
* per comp GMM standard deviation ([94240a7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/94240a7521d44b4466ed044f85c441bd453c2a93))
* progress on contrast limits ([5f9e7e0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/5f9e7e0f6adb7e0f8d514da4cd2caee18d24173c))
* tuned two best methods, need removal of tuning params ([f4c076f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/f4c076f921e5afad518e1e772a949e52688f665c))
* tuning contrast limits ([356ca03](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/356ca03a0099be2976a76ba9083b6ece98e12d78))
* update gmm sample size ([ff01ae5](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ff01ae5c9024edad824f364f8895a6ace2e299f9))
* update shader names ([2dfe36f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2dfe36fb2d7399c2c941beeb748d38eca4bcd044))
* update tuning ([65c3f93](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/65c3f930814ffee3ccd3e654cba195e18a9b80e3))


### 🐞 Bug Fixes

* allow segmentation to overwrite existing segment properties ([#26](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/26)) ([dc38b7d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/dc38b7d42e777e7d3db7223b84589876c9a221c4))
* allow segmentation to overwrite existing segment properties on new run ([54c1b14](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/54c1b1454c9dfe779ea9da656dcf777296998efe))
* calculate grid size over all LODs, not LOD 0 only ([2753486](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/27534863e48ab586c436b7ecbc66ac6fec86f8d5))
* correct flag for projection_quaternion default ([2692a6d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2692a6dde4fe3e7d410ceca288bbf0cb23e27733))
* correct numpy array in type hint after using | None ([945fbc8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/945fbc81372604fa409db844fcc33520bf134be1))
* correct octree processing at borders ([02cd34e](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/02cd34e00afaeceb4accadab2b12b7bd3c2af1ec))
* correct val typo for value in shader builder ([16b89ca](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/16b89ca3bfc38d51c95979a886fadea696aa66af))
* don't crash on projection quaternion, allow axis line hiding ([683babf](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/683babf8acc81acfa81ac75aff5ad35b720ea820))
* first round of octree improvements ([0f3d863](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/0f3d8630dd0c0cf405d6df4483173c573271e4b1))
* gmm limits don't go out of data bounds ([ae99495](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ae99495a5b36f3d1e2b786a489b464c82117bb9e))
* have VR limits inverted using our control not builtin ([ebc8911](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ebc8911e62e3e2296f6adc3aa207f2a30afcc59a))
* improve contrast limits ([452a7bf](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/452a7bfc7a5e8606abe3056429bbed1190a9096b))
* include patched octree in multires code ([1df1024](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1df102427fd6e977508287130b023da7c232338c))
* issue with empty screenshots ([c4ff38a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/c4ff38aa53a4218d0e01ad022565f6827f522d89))
* json scale typing and also processing for meshes ([4c5123c](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/4c5123c80a652d48c60e1d9fb75af56a8643d6b4))
* process data in chunks for bounding box to not run out of memory ([7e09ce4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7e09ce497ac55af6576a54742ffe1e69703e43ff))
* remove accidental debug print ([b52ea24](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b52ea24602721fa4b30ed1a828cdb48d53d5c376))
* remove top level matplotlib import ([62dddaa](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/62dddaadcf170feb402780bf23985d5f2f21c796))
* **test:** test fixes after shader and limit changes ([7923a2a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7923a2a827cf8cab38246f816b2f316b8bef263f))
* update docstring for decimation aggressiveness ([2516420](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2516420a69ed2469b6f739b019540e62dd12193b))


### 📝 Documentation

* add docstring ([bcc2c14](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/bcc2c149fb742cc88068207160a135a001f155f5))
* improve docs on window from limits ([653cea1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/653cea1473d0a6a75a980fb4b859d6e5282aa023))
* update notes about meshes ([29d7010](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/29d701011170d713536ce51c49397004fb35e764))


### 💅 Styles

* rename oriented point JSON generator ([#21](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/21)) ([d1fd921](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d1fd921c229b42bac59dc57904cdf20bb4b45817))


### 🧹 Miscellaneous Chores

* formatting ([b19dd7d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b19dd7d4b8ff3f200b1e19fc157dfa4882de5e0c))
* linting fix ([a066566](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/a0665660a4cb9eab4ddf79478ca4592382652044))
* update lock file ([5fdc75c](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/5fdc75c43698ea200b0f0aa05bf7b5053c663dc5))


### ♻️ Code Refactoring

* make calculator not intended to be reusable, but tied to a volume. Also make the base calc abstract ([b108b2d](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b108b2dead8c20206abcdc082dc08150461563d0))
* remove Optional and more generic dirs, fix redundant code line ([f1a179a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/f1a179a833ed1c19bd805d540a6d3877898543a8))
* remove timing function for experiment ([aeb13db](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/aeb13db25af9c99212857f856644bbfefa5a132a))
* rename oriented point JSON generator ([2f2c8e9](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2f2c8e935e90f09fe154a73140eae6f0d2bd2c46))


### 🧪 Tests

* add contrast limit test ([d70c05f](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d70c05f0c2ec6128adb5cde43d6a2ae46ac6fc2a))
* add manual test for mask from segmentation ([ea3ffd7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ea3ffd7c09f74eb95e51064af1e6c5e2808b143f))
* add other URLs for contrast limits ([4cef902](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/4cef902ecc4f15ce582d5fe7c6f313ca1b0d00b1))
* better contrast limit manual test ([9a60925](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/9a60925c917514ab2e92e9ccb129a1910679f257))
* better mesh from seg test ([7d29fd6](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7d29fd60c1074996774105f2bea179846026137c))
* disable long running test ([d8af9d4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d8af9d49464ebe1a7caaef767596dea26d9b1e34))
* update manual tests with seg properties ([b482f6e](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b482f6e2ca33a43c5665d809f2c6c84ee01db749))
