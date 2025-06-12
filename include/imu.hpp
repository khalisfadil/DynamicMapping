#pragma once

#include <cstdint>
#include <Eigen/Core>

namespace dynamicMap {

    struct IMUData {
        double timestamp = 0;
        Eigen::Vector3d ang_vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d lin_acc = Eigen::Vector3d::Zero();

        IMUData(double timestamp_, Eigen::Vector3d ang_vel_, Eigen::Vector3d lin_acc_)
            : timestamp(timestamp_), ang_vel(ang_vel_), lin_acc(lin_acc_) {}

        IMUData() {}
    };

} // namespace dynamicMap