// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
// must include pybind11/stl.h if using containers in STL in arguments.
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
// #include <vector>
#include <iostream>
#include <math.h>

namespace py = pybind11;
using namespace pybind11::literals;

template <typename DType, typename DTypeInt, int NDim>
int points_to_voxel_3d_np(py::array_t<DType> points, py::array_t<DType> voxels,
                          py::array_t<int> coors,
                          py::array_t<int> num_points_per_voxel,
                          py::array_t<int> coor_to_voxelidx,
                          std::vector<DType> voxel_size,
                          std::vector<DType> coors_range, int max_points,
                          int max_voxels) {
  auto points_rw = points.template mutable_unchecked<2>(); // (-1, 4/5)
  auto voxels_rw = voxels.template mutable_unchecked<3>(); // (max_voxels, max_point, 4/5)
  auto coors_rw = coors.mutable_unchecked<2>(); // (max_voxels, 3)
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>(); // (max_voxels)
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>(); // (500, 500, 3)
  auto N = points_rw.shape(0);
  auto num_features = points_rw.shape(1);
  // auto ndim = points_rw.shape(1) - 1;
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim]; // NDim: 3
  int c;
  int grid_size[NDim]; // 3
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < N; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels)
        break;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  for (int i = 0; i < voxel_num; ++i) {
    coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
  }
  return voxel_num;
}


template <typename DType, int NDim>
int nusc_points_to_voxel_3d_np(
                          py::array_t<DType> cur_sweep_points, 
                          py::array_t<DType> other_sweep_points, 
                          py::array_t<DType> voxels,
                          py::array_t<int> coors,
                          py::array_t<int> num_points_per_voxel,
                          py::array_t<int> coor_to_voxelidx,
                          std::vector<DType> voxel_size,
                          std::vector<DType> coors_range, int max_points,
                          int max_voxels, int max_cur_sample_num) {
  auto cur_sweep_points_rw = cur_sweep_points.template mutable_unchecked<2>(); // (-1, 4/5)
  auto other_sweep_points_rw = other_sweep_points.template mutable_unchecked<2>();

  auto voxels_rw = voxels.template mutable_unchecked<3>(); // (max_voxels, max_point, 4/5)

  auto coors_rw = coors.mutable_unchecked<2>(); // (max_voxels, 3)
  auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>(); // (max_voxels)
  auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>(); // (500, 500, 3)
  auto cur_sample_num = cur_sweep_points_rw.shape(0);
  auto other_sample_num = other_sweep_points_rw.shape(0);
  auto num_features = cur_sweep_points_rw.shape(1);
  // auto ndim = points_rw.shape(1) - 1;
  constexpr int ndim_minus_1 = NDim - 1;
  int voxel_num = 0;
  bool failed = false;
  int coor[NDim]; // NDim: 3
  int c;
  int grid_size[NDim]; // 3
  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }
  int voxelidx, num;
  for (int i = 0; i < cur_sample_num; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((cur_sweep_points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_cur_sample_num)
        break;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = cur_sweep_points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  // after random select all positive points, then padding to max_cur_sample_num
  while (voxel_num < max_cur_sample_num){
    for (int i = 0; i < num_features; i++){
      voxels_rw(voxel_num, 0, i) = cur_sweep_points_rw(0, i);
    }
    num_points_per_voxel_rw(voxel_num) += 1;
    voxel_num += 1;
  }
  // then for other_sample_points
  for (int i = 0; i < other_sample_num; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((other_sweep_points_rw(i, j) - coors_range[j]) / voxel_size[j]);
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }
    if (failed)
      continue;
    voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
    if (voxelidx == -1) {
      voxelidx = voxel_num;
      if (voxel_num >= max_voxels)
        break;
      voxel_num += 1;
      coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
      for (int k = 0; k < NDim; ++k) {
        coors_rw(voxelidx, k) = coor[k];
      }
    }
    num = num_points_per_voxel_rw(voxelidx);
    if (num < max_points) {
      for (int k = 0; k < num_features; ++k) {
        voxels_rw(voxelidx, num, k) = other_sweep_points_rw(i, k);
      }
      num_points_per_voxel_rw(voxelidx) += 1;
    }
  }
  // after random select all positive points, then padding to max_cur_sample_num
  while (voxel_num < max_voxels){
    for (int i = 0; i < num_features; i++){
      voxels_rw(voxel_num, 0, i) = other_sweep_points_rw(0, i);
    }
    num_points_per_voxel_rw(voxel_num) += 1;
    voxel_num += 1;
  }
  return voxel_num;
}

PYBIND11_MODULE(points2voxel, m){
  m.doc() = R"pbdoc(
          Voxel Generation Module on Pybind11
          -----------------------
          .. currentmodule:: points2voxel 
          .. autosummary::
          points_to_voxel_3d_np
      )pbdoc";
  m.def("points_to_voxel_3d_np", &points_to_voxel_3d_np<float, int, 3>,
        "Generate voxels based on the whole point cloud"); 
  m.def("nusc_points_to_voxel_3d_np", &nusc_points_to_voxel_3d_np<float, 3>,
        "Generate voxels based on the whole point cloud"); 
}
