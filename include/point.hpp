#pragma once

#include <cstdint>

namespace dynamicMap {

    struct Point3D {
        double x = 0.0;                // X coordinate in sensor frame
        double y = 0.0;                // Y coordinate in sensor frame
        double z = 0.0;                // Z coordinate in sensor frame
        uint16_t c_id = -1;            // Channel index (0 to 127, corresponding to row)
        uint16_t m_id = -1;            // Measurement index (0 to 2047, corresponding to column)
        double timestamp = 0.0;        // Absolute timestamp (seconds)
        double relative_timestamp = 0.0; // Relative timestamp from the initial frame (seconds)
        uint16_t reflectivity = 0;     // Reflectivity value
        uint16_t signal = 0;           // Signal strength
        uint16_t nir = 0;              // Near-infrared value
    };

    struct Points3D {
        Eigen::Vector3d raw_pt;  // Raw point read from the sensor
        Eigen::Vector3d pt;      // Corrected point taking into account the motion of the sensor during frame acquisition
        Eigen::Vector3i att;
        double relative_timestamp = 0.0;  // Relative timestamp in the frame in [0.0, 1.0]
        double timestamp = 0.0;        // The absolute timestamp (if applicable)
        int m_id = -1;              // The beam id of the point
    };

} // namespace dynamicMap