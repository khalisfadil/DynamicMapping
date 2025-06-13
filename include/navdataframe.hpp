#pragma once

#include <LidarDataframe.hpp>
#include <DataFrame_NavMsg.hpp>
#include <navMath.hpp>

namespace dynamicMap {

    struct NavDataFrame {
        lidarDecode::LidarDataFrame lidar_data; // LiDAR data frame
        decodeNav::DataFrameNavMsg nav_data;    // Navigation data message
        double oriLat = 0.0;                    // Origin latitude (radians)
        double oriLon = 0.0;                    // Origin longitude (radians)
        double oriAlt = 0.0;                    // Origin altitude (meters)
        double N = 0.0;                         // North coordinate (meters)
        double E = 0.0;                         // East coordinate (meters)
        double D = 0.0;                         // Down coordinate (meters)

        // Default constructor
        NavDataFrame() = default;

        // Constructor with initialization
        NavDataFrame(const lidarDecode::LidarDataFrame& lidar, 
                     const decodeNav::DataFrameNavMsg& nav, 
                     double orilat, double orilon, double orialt)
            : lidar_data(lidar), nav_data(nav), oriLat(orilat), oriLon(orilon), oriAlt(orialt) {
            // Create NavMath instance to perform LLA to NED conversion
            navMath::NavMath nav_math;
            Eigen::Vector3d NED = nav_math.LLA2NED(nav_data.latitude, nav_data.longitude, 
                                                  static_cast<double>(nav_data.altitude), 
                                                  oriLat, oriLon, oriAlt);
            N = NED(0); // North
            E = NED(1); // East
            D = NED(2); // Down

            Eigen::Vector4d q = nav_math.getQuat(nav_data.roll, nav_data.pitch, nav_data.yaw);


        }


        // Clear all data
        void clear() {
            lidar_data.clear();
            nav_data.clear();
            oriLat = 0.0;
            oriLon = 0.0;
            oriAlt = 0.0;
            N = 0.0;
            E = 0.0;
            D = 0.0;
        }

        // Reserve space for LiDAR data
        void reserve(size_t size) {
            lidar_data.reserve(size);
        }

        // Validate data consistency
        bool is_valid() const {
            // Check LiDAR and nav data validity
            bool lidar_valid = lidar_data.numberpoints > 0 && lidar_data.timestamp > 0.0;
            bool nav_valid = nav_data.timestamp > 0.0 && 
                            std::isfinite(nav_data.latitude) && 
                            std::isfinite(nav_data.longitude) && 
                            std::isfinite(nav_data.altitude);
            // Check origin and NED coordinates
            bool coords_valid = std::isfinite(oriLat) && std::isfinite(oriLon) && 
                               std::isfinite(oriAlt) && 
                               std::isfinite(N) && std::isfinite(E) && std::isfinite(D);
            return lidar_valid && nav_valid && coords_valid;
        }
    };

} // namespace dynamicMap