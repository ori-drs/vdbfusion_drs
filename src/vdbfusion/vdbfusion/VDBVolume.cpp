// MIT License
//
// # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "VDBVolume.h"

// OpenVDB
#include <openvdb/Types.h>
#include <openvdb/math/DDA.h>
#include <openvdb/math/Ray.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/GridTransformer.h>
#include <openvdb/tools/Composite.h>

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

namespace {

float ComputeSDF(const Eigen::Vector3d& origin,
                 const Eigen::Vector3d& point,
                 const Eigen::Vector3d& voxel_center) {
    const Eigen::Vector3d v_voxel_origin = voxel_center - origin;
    const Eigen::Vector3d v_point_voxel = point - voxel_center;
    const double dist = v_point_voxel.norm();
    const double proj = v_voxel_origin.dot(v_point_voxel);
    const double sign = proj / std::abs(proj);
    return static_cast<float>(sign * dist);
}

Eigen::Vector3d GetVoxelCenter(const openvdb::Coord& voxel, const openvdb::math::Transform& xform) {
    const float voxel_size = xform.voxelSize()[0];
    openvdb::math::Vec3d v_wf = xform.indexToWorld(voxel) + voxel_size / 2.0;
    return Eigen::Vector3d(v_wf.x(), v_wf.y(), v_wf.z());
}



}  // namespace

namespace vdbfusion {

VDBVolume::VDBVolume(float voxel_size, float sdf_trunc, bool space_carving /* = false*/)
    : voxel_size_(voxel_size), sdf_trunc_(sdf_trunc), space_carving_(space_carving) {
    tsdf_ = openvdb::FloatGrid::create(sdf_trunc_);
    tsdf_->setName("D(x): signed distance grid");
    tsdf_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    tsdf_->setGridClass(openvdb::GRID_LEVEL_SET);

    weights_ = openvdb::FloatGrid::create(0.0f);
    weights_->setName("W(x): weights grid");
    weights_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    weights_->setGridClass(openvdb::GRID_UNKNOWN);
}

void VDBVolume::UpdateTSDF(const float& sdf,
                           const openvdb::Coord& voxel,
                           const std::function<float(float)>& weighting_function) {
    using AccessorRW = openvdb::tree::ValueAccessorRW<openvdb::FloatTree>;
    if (sdf > -sdf_trunc_) {
        AccessorRW tsdf_acc = AccessorRW(tsdf_->tree());
        AccessorRW weights_acc = AccessorRW(weights_->tree());
        const float tsdf = std::min(sdf_trunc_, sdf);
        const float weight = weighting_function(sdf);
        const float last_weight = weights_acc.getValue(voxel);
        const float last_tsdf = tsdf_acc.getValue(voxel);
        const float new_weight = weight + last_weight;
        const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
        tsdf_acc.setValue(voxel, new_tsdf);
        weights_acc.setValue(voxel, new_weight);
    }
}

void VDBVolume::Integrate(openvdb::FloatGrid::Ptr grid,
                          const std::function<float(float)>& weighting_function) {
    for (auto iter = grid->cbeginValueOn(); iter.test(); ++iter) {
        const auto& sdf = iter.getValue();
        const auto& voxel = iter.getCoord();
        this->UpdateTSDF(sdf, voxel, weighting_function);
    }
}

void VDBVolume::Integrate(const std::vector<Eigen::Vector3d>& points,
                          const Eigen::Vector3d& origin,
                          const std::function<float(float)>& weighting_function) {
    if (points.empty()) {
        std::cerr << "PointCloud provided is empty\n";
        return;
    }

    // Get some variables that are common to all rays
    const openvdb::math::Transform& xform = tsdf_->transform();
    const openvdb::Vec3R eye(origin.x(), origin.y(), origin.z());

    // Get the "unsafe" version of the grid acessors
    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();

    // Launch an for_each execution, use std::execution::par to parallelize this region
    std::for_each(points.cbegin(), points.cend(), [&](const auto& point) {
        // Get the direction from the sensor origin to the point and normalize it
        const Eigen::Vector3d direction = point - origin;
        openvdb::Vec3R dir(direction.x(), direction.y(), direction.z());
        dir.normalize();

        // Truncate the Ray before and after the source unless space_carving_ is specified.
        const auto depth = static_cast<float>(direction.norm());
        const float t0 = space_carving_ ? 0.0f : depth - sdf_trunc_;
        const float t1 = depth + sdf_trunc_;

        // Create one DDA per ray(per thread), the ray must operate on voxel grid coordinates.
        const auto ray = openvdb::math::Ray<float>(eye, dir, t0, t1).worldToIndex(*tsdf_);
        openvdb::math::DDA<decltype(ray)> dda(ray);
        do {
            const auto voxel = dda.voxel();
            const auto voxel_center = GetVoxelCenter(voxel, xform);
            const auto sdf = ComputeSDF(origin, point, voxel_center);
            if (sdf > -sdf_trunc_) {
                const float tsdf = std::min(sdf_trunc_, sdf);
                const float weight = weighting_function(sdf);
                const float last_weight = weights_acc.getValue(voxel);
                const float last_tsdf = tsdf_acc.getValue(voxel);
                const float new_weight = weight + last_weight;
                const float new_tsdf = (last_tsdf * last_weight + tsdf * weight) / (new_weight);
                tsdf_acc.setValue(voxel, new_tsdf);
                weights_acc.setValue(voxel, new_weight);
            }
        } while (dda.step());
    });
}

openvdb::FloatGrid::Ptr VDBVolume::Prune(float min_weight) const {
    const auto weights = weights_->tree();
    const auto tsdf = tsdf_->tree();
    const auto background = sdf_trunc_;
    openvdb::FloatGrid::Ptr clean_tsdf = openvdb::FloatGrid::create(sdf_trunc_);
    clean_tsdf->setName("D(x): Pruned signed distance grid");
    clean_tsdf->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
    clean_tsdf->setGridClass(openvdb::GRID_LEVEL_SET);
    clean_tsdf->tree().combine2Extended(tsdf, weights, [=](openvdb::CombineArgs<float>& args) {
        if (args.aIsActive() && args.b() > min_weight) {
            args.setResult(args.a());
            args.setResultIsActive(true);
        } else {
            args.setResult(background);
            args.setResultIsActive(false);
        }
    });
    return clean_tsdf;
}


void VDBVolume::DualGridSampler(openvdb::FloatGrid::Ptr sourceGrid,
                     openvdb::FloatGrid::Ptr targetGrid,
                     openvdb::Coord coord){

//   // Instantiate the DualGridSampler template on the grid type and on
//   // a box sampler for thread-safe but uncached trilinear interpolation.
//   openvdb::tools::DualGridSampler<openvdb::FloatGrid::Ptr, openvdb::tools::BoxSampler>
//       sampler(sourceGrid, targetGrid.constTransform());
//   // Compute the value of the source grid at a location in the
//   // target grid's index space.
//   openvdb::FloatGrid::Ptr::ValueType value = sampler(openvdb::Coord(-23, -50, 202));
//   // Request a value accessor for accelerated access to the source grid.
//   // (Because value accessors employ a cache, it is important to declare
//   // one accessor per thread.)
//   openvdb::FloatGrid::Ptr::ConstAccessor accessor = sourceGrid.getConstAccessor();
//   // Instantiate the DualGridSampler template on the accessor type and on
//   // a box sampler for accelerated trilinear interpolation.
//   openvdb::tools::DualGridSampler<GridType::ConstAccessor, openvdb::tools::BoxSampler>
//       fastSampler(accessor, sourceGrid.constTransform(), targetGrid.constTransform());
//   // Compute the value of the source grid at a location in the
//   // target grid's index space.
//   value = fastSampler(openvdb::Coord(-23, -50, 202));



}


std::string VDBVolume::CompareTSDFGrids(openvdb::FloatGrid::Ptr grid_1,
                                 openvdb::FloatGrid::Ptr grid_2,
                                 openvdb::FloatGrid::Ptr change_grid) {
                                    // changegrid.Compare(etc...)
    // Transform grids 
    //this->TransformGrids(grid_1, grid_2);

    // Get the "unsafe" version of the grid acessors
    auto tsdf_acc = tsdf_->getUnsafeAccessor();
    auto weights_acc = weights_->getUnsafeAccessor();
    
    // Get an accessor for coordinate-based access to voxels.
    openvdb::FloatGrid::Accessor accessor = grid_2->getAccessor();
    openvdb::FloatGrid::Accessor change_grid_accessor = change_grid->getAccessor();
    std::string message;
    for (auto iter = grid_1->cbeginValueOn(); iter.test(); ++iter) {
      const auto& voxel1 = iter.getCoord();
      const auto& sdf2 = accessor.getValue(voxel1);
      const auto& sdf1 = iter.getValue();
      const float last_tsdf = tsdf_acc.getValue(voxel1);
      const auto& sdfchange = sdf1 - sdf2 - last_tsdf;
      change_grid_accessor.setValue(voxel1, sdfchange);
      this->UpdateTSDF(sdfchange, voxel1, 0); //weighting function set as zero for now
      message += (" \nVoxel values: " + std::to_string(sdf1) + std::to_string(sdf1) ); 
       
    }
    return message; 
}

void VDBVolume::TransformGrids(openvdb::FloatGrid::Ptr sourceGrid,
                               openvdb::FloatGrid::Ptr targetGrid,
                               bool prune = false) {

    // Get the source and target grids' index space to world space transforms.
    const openvdb::math::Transform
        &sourceXform = sourceGrid->transform(),
        &targetXform = targetGrid->transform();
    // Compute a source grid to target grid transform.
    // (For this example, we assume that both grids' transforms are linear,
    // so that they can be represented as 4 x 4 matrices.)
    openvdb::Mat4R xform =
        sourceXform.baseMap()->getAffineMap()->getMat4() *
        targetXform.baseMap()->getAffineMap()->getMat4().inverse();
    // Create the transformer.
    openvdb::tools::GridTransformer transformer(xform);
    // Resample using nearest-neighbor interpolation.
    transformer.transformGrid<openvdb::tools::PointSampler, openvdb::FloatGrid>(
          *sourceGrid, *targetGrid);
    // Resample using trilinear interpolation.
    transformer.transformGrid<openvdb::tools::BoxSampler, openvdb::FloatGrid>(
          *sourceGrid, *targetGrid);
    // Resample using triquadratic interpolation.
    transformer.transformGrid<openvdb::tools::QuadraticSampler, openvdb::FloatGrid>(
          *sourceGrid, *targetGrid);
    if (prune) {
        // Prune the target tree for optimal sparsity.
        targetGrid->tree().prune();
    }
}

void VDBVolume::GridValueTransformation(openvdb::FloatGrid::Ptr grid,
                            float value){
    // Define a local function that doubles the value to which the given
    // value iterator points.
    struct Local {
        static inline void op(const openvdb::FloatGrid::ValueAllIter& iter) {
            iter.setValue(*iter * 2); //value
        }
    };
    // Apply the function to all values.
    openvdb::tools::foreach(grid->beginValueAll(), Local::op);
}

void VDBVolume::CombineGrids(openvdb::FloatGrid::Ptr gridA,
                  openvdb::FloatGrid::Ptr gridB){
    // Two grids of the same type containing level set volumes

    // Save copies of the two grids; CSG operations always modify
    // the A grid and leave the B grid empty.
    openvdb::FloatGrid::ConstPtr
        copyOfGridA = gridA->deepCopy(),
        copyOfGridB = gridB->deepCopy();
    // Compute the union (A u B) of the two level sets.
    openvdb::tools::csgUnion(*gridA, *gridB);
    // Restore the original level sets.
    gridA = copyOfGridA->deepCopy();
    gridB = copyOfGridB->deepCopy();
    // Compute the intersection (A n B) of the two level sets.
    openvdb::tools::csgIntersection(*gridA, *gridB);
    // Restore the original level sets.
    gridA = copyOfGridA->deepCopy();
    gridB = copyOfGridB->deepCopy();
    // Compute the difference (A / B) of the two level sets.
    openvdb::tools::csgDifference(*gridA, *gridB);

}

void VDBVolume::CompositingGrids(openvdb::FloatGrid::Ptr gridA,
                                                    openvdb::FloatGrid::Ptr gridB){
    // Save copies of the two grids; compositing operations always
    // modify the A grid and leave the B grid empty.
    openvdb::FloatGrid::ConstPtr
        copyOfGridA = gridA->deepCopy(),
        copyOfGridB = gridB->deepCopy();
    // At each voxel, compute a = max(a, b).
    // openvdb::tools::compMax(*gridA, *gridB);
    // // Restore the original grids.
    // gridA = copyOfGridA->deepCopy();
    // gridB = copyOfGridB->deepCopy();
    // // At each voxel, compute a = min(a, b).
    // openvdb::tools::compMin(*gridA, *gridB);
    // // Restore the original grids.
    // gridA = copyOfGridA->deepCopy();
    // gridB = copyOfGridB->deepCopy();
    // At each voxel, compute a = a + b.
    // openvdb::tools::compSum(*gridA, *gridB);
    openvdb::tools::csgDifference(*gridA, *gridB);
    // Restore the original grids.
    // gridA = copyOfGridA->deepCopy();
    // gridB = copyOfGridB->deepCopy();
    // // At each voxel, compute a = a * b.
    // openvdb::tools::compMul(*gridA, *gridB);
    //    return gridA;
}

// const float VDBVolume::Accessor(const GridType sourceGrid,
//                          const GridType targetGrid,
//                          openvdb::Coord ijk) {

//   // Instantiate the DualGridSampler template on the grid type and on
//   // a box sampler for thread-safe but uncached trilinear interpolation.
//   openvdb::tools::DualGridSampler<GridType, openvdb::tools::BoxSampler>
//       sampler(sourceGrid, targetGrid.constTransform());
//   // Compute the value of the source grid at a location in the
//   // target grid's index space.
//   GridType::ValueType value = sampler(ijk);
//   // Request a value accessor for accelerated access to the source grid.
//   // (Because value accessors employ a cache, it is important to declare
//   // one accessor per thread.)
//   GridType::ConstAccessor accessor = sourceGrid.getConstAccessor();
//   // Instantiate the DualGridSampler template on the accessor type and on
//   // a box sampler for accelerated trilinear interpolation.
//   openvdb::tools::DualGridSampler<GridType::ConstAccessor, openvdb::tools::BoxSampler>
//       fastSampler(accessor, sourceGrid.constTransform(), targetGrid.constTransform());
//   // Compute the value of the source grid at a location in the
//   // target grid's index space.
//   tsdf = fastSampler(ijk);
//   return tsdf;
// }

}  // namespace vdbfusion
