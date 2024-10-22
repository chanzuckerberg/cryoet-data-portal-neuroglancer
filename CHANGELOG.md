# Changelog

## [1.1.0](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/compare/v1.0.0...v1.1.0) (2024-10-22)


### ✨ Features

* add segmentation property objects ([d7664bc](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d7664bc8fd487fbdab46e727665209df2fba95c6))
* Adding release please ([#19](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/19)) ([4ec9ca7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/4ec9ca7c0558203bd363d49599c074ade951d905))
* allow seg gen to control the segments shown, have random colours, and the mesh render scale ([759def1](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/759def1658cf248fa5b71743e8db11cd466bcc68))
* change print to logging of chunk comparison size ([878a7a7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/878a7a7e0b041fdbe942a11bdbdb8f55b8369691))
* link seg properties into mesh and seg creation ([dfbd34a](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/dfbd34a24e4413cc62baf37251c24456b822e5b8))
* mesh precompute functions and state generation ([#23](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/23)) ([02439d8](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/02439d81f96f2f3582c1bec2102bad1c4540efb3))
* more control over mesh from segmentation ([e1af340](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/e1af340a2844639dadf5f84209082ffeb219662d))


### 🐞 Bug Fixes

* calculate grid size over all LODs, not LOD 0 only ([2753486](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/27534863e48ab586c436b7ecbc66ac6fec86f8d5))
* correct octree processing at borders ([02cd34e](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/02cd34e00afaeceb4accadab2b12b7bd3c2af1ec))
* first round of octree improvements ([0f3d863](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/0f3d8630dd0c0cf405d6df4483173c573271e4b1))
* include patched octree in multires code ([1df1024](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/1df102427fd6e977508287130b023da7c232338c))
* json scale typing and also processing for meshes ([4c5123c](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/4c5123c80a652d48c60e1d9fb75af56a8643d6b4))
* process data in chunks for bounding box to not run out of memory ([7e09ce4](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7e09ce497ac55af6576a54742ffe1e69703e43ff))
* remove accidental debug print ([b52ea24](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b52ea24602721fa4b30ed1a828cdb48d53d5c376))
* update docstring for decimation aggressiveness ([2516420](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2516420a69ed2469b6f739b019540e62dd12193b))


### 📝 Documentation

* update notes about meshes ([29d7010](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/29d701011170d713536ce51c49397004fb35e764))


### 💅 Styles

* rename oriented point JSON generator ([#21](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/issues/21)) ([d1fd921](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/d1fd921c229b42bac59dc57904cdf20bb4b45817))


### ♻️ Code Refactoring

* rename oriented point JSON generator ([2f2c8e9](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/2f2c8e935e90f09fe154a73140eae6f0d2bd2c46))


### 🧪 Tests

* add manual test for mask from segmentation ([ea3ffd7](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/ea3ffd7c09f74eb95e51064af1e6c5e2808b143f))
* better mesh from seg test ([7d29fd6](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/7d29fd60c1074996774105f2bea179846026137c))
* update manual tests with seg properties ([b482f6e](https://github.com/chanzuckerberg/cryoet-data-portal-neuroglancer/commit/b482f6e2ca33a43c5665d809f2c6c84ee01db749))
