#pragma once

#include <voxel.hpp>

namespace dynamicMap {

    struct VizuDataFrame {

        ArrayVector3d pointcloud = {};
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();                // Transformation matrix

        // Clear all data
        void clear() {
            pointcloud.clear();
            T = Eigen::Matrix4d::Identity();   
        }
    };

} // namespace dynamicMap