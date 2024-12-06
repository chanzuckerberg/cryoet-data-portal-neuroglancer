# Changelog

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
