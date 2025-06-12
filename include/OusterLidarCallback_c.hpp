#pragma once

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <mutex> // Though not used for threading in this version, kept if future needs arise

#include <Point.hpp>    // Assuming Point.hpp defines necessary point structures if used by DataFrame
#include <LidarDataframe.hpp> // Assuming Dataframe.hpp defines the DataFrame structure
#include <LidarIMUDataFrame.hpp> // Assuming Dataframe.hpp defines the DataFrame structure

namespace dynamicMap {

    class OusterLidarCallback {
    public:
        explicit OusterLidarCallback(const std::string& json_path);
        explicit OusterLidarCallback(const nlohmann::json& json_data);

        void decode_packet_single_return(const std::vector<uint8_t>& packet, lidarDecode::LidarDataFrame& frame);
        void decode_packet_legacy(const std::vector<uint8_t>& packet, lidarDecode::LidarDataFrame& frame); 
        void decode_packet_LidarIMU(const std::vector<uint8_t>& packet, lidarDecode::LidarIMUDataFrame& frame); 
        const lidarDecode::LidarDataFrame& get_latest_lidar_frame() const { return buffer_toggle_ ? data_buffer1_ : data_buffer2_; }
        const nlohmann::json& get_metadata() const { return metadata_; }

    private:
        nlohmann::json metadata_;

        // Lookup tables: [m_id][c_id] for contiguous access per measurement
        std::vector<std::vector<double, Eigen::aligned_allocator<double>>> x_1_; // Per-measurement, per-channel direction component
        std::vector<std::vector<double, Eigen::aligned_allocator<double>>> y_1_;
        std::vector<std::vector<double, Eigen::aligned_allocator<double>>> z_1_;
        std::vector<double> x_2_; // Per-measurement offset
        std::vector<double> y_2_;
        std::vector<double> z_2_;

        std::vector<double> r_min_; // Minimum range per channel
        std::vector<double> sin_beam_azimuths_;
        std::vector<double> cos_beam_azimuths_;
        std::vector<double> sin_beam_altitudes_;
        std::vector<double> cos_beam_altitudes_;
        std::vector<int> pixel_shifts_; // Column cycle phase per row/channel

        Eigen::Matrix4d lidar_to_sensor_transform_;
        // Eigen::Matrix4d vehicle_to_lidar_transform_; // Declared but not used in provided .cpp
        double lidar_origin_to_beam_origin_mm_;

        // Packet structure parameters (some will be calculated in initialize)
        size_t block_size_;           // Size of a single measurement block, calculated from pixels_per_column_
        size_t expected_size_;        // Expected total packet size, calculated from columns_per_packet_ & block_size_
        
        // Default values, can be overridden by JSON or remain if not in JSON (though code expects them)
        size_t PACKET_HEADER_BYTES = 32;     // Assumed size of data before the first measurement block in the packet (bytes)
        size_t PACKET_FOOTER_BYTES = 32;     // Assumed size of data after the last measurement block
        size_t COLUMN_HEADER_BYTES = 12;
        size_t CHANNEL_STRIDE_BYTES = 12;
        size_t MEASUREMENT_BLOCK_STATUS_BYTES = 0;
        
        // Metadata-derived parameters
        std::string udp_profile_lidar_ = "UNKNOWN";
        int columns_per_frame_ = 2048;
        int pixels_per_column_ = 128;
        int columns_per_packet_ = 16; // Default, will be read from JSON
        // double signal_multiplier_ = 1.0; // Loaded from JSON but not used in provided .cpp

        // Frame state
        uint16_t frame_id_ = 0;           // ID of the frame currently being built
        uint32_t number_points_ = 0;      // Number of points accumulated for the current frame
        double latest_timestamp_s = 0.0; // Timestamp of the latest processed column block

        // Double buffers for point cloud data
        lidarDecode::LidarDataFrame data_buffer1_;
        lidarDecode::LidarDataFrame data_buffer2_;
        bool buffer_toggle_ = true; // true: data_buffer1_ is read_buffer, data_buffer2_ is write_buffer
                                // false: data_buffer2_ is read_buffer, data_buffer1_ is write_buffer

        void initialize();
        void parse_metadata(const nlohmann::json& json_data);
        void swap_buffers() { buffer_toggle_ = !buffer_toggle_; }
    };
} // namespace dynamicMap