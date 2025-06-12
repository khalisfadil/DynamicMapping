#pragma once

#include <Eigen/Dense>

namespace dynamicMap {

    struct PoseData {
        double timestamp = 0;
        Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    };

} // namespace dynamicMap
