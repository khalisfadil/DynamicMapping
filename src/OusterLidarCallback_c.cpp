#include <fstream>
#include <stdexcept>
#include <cmath>
#include <cstring> // For std::memcpy
#include <iostream> // For std::cerr
#include <algorithm> // For std::max

#include "OusterLidarCallback_c.hpp" // Use quotes for local headers

#ifdef __AVX2__
#include <immintrin.h>
#endif


// Endian conversion (ensure these are available, e.g. via <endian.h> on Linux/glibc, or provide fallback)
// For cross-platform, you might need custom implementations or a library.
// Assuming htobe16, htole16, etc. are available or defined elsewhere.
// For this example, we'll assume standard Linux/glibc style. If not, this part needs attention.
#include <endian.h> // For leXXtoh functions if on a big-endian system with glibc
// Or define manually:
// static inline uint16_t le16toh_manual(uint16_t le_val) {
//     return (le_val >> 8) | (le_val << 8);
// } // etc. for 32 and 64


namespace dynamicMap {

    using json = nlohmann::json;

    // -----------------------------------------------------------------------------

    OusterLidarCallback::OusterLidarCallback(const std::string& json_path) {
        std::ifstream file(json_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open JSON file: " + json_path);
        }
        json json_data;
        try {
            file >> json_data;
        } catch (const json::parse_error& e) {
            throw std::runtime_error("JSON parse error in " + json_path + ": " + e.what());
        }
        parse_metadata(json_data);
        initialize();
    }

    // -----------------------------------------------------------------------------

    OusterLidarCallback::OusterLidarCallback(const json& json_data) {
        parse_metadata(json_data);
        initialize();
    }

    // -----------------------------------------------------------------------------

    void OusterLidarCallback::parse_metadata(const json& json_data_param) {
        if (!json_data_param.is_object()) {
            throw std::runtime_error("JSON data must be an object");
        }
        // Store the provided JSON data; subsequent accesses use the metadata_ member.
        metadata_ = json_data_param;

        try {
            if (!metadata_.contains("lidar_data_format") || !metadata_["lidar_data_format"].is_object()) {
                throw std::runtime_error("Missing or invalid 'lidar_data_format' object");
            }
            if (!metadata_.contains("config_params") || !metadata_["config_params"].is_object()) {
                throw std::runtime_error("Missing or invalid 'config_params' object");
            }
            if (!metadata_.contains("beam_intrinsics") || !metadata_["beam_intrinsics"].is_object()) {
                throw std::runtime_error("Missing or invalid 'beam_intrinsics' object");
            }
            if (!metadata_.contains("lidar_intrinsics") || !metadata_["lidar_intrinsics"].is_object() ||
                !metadata_["lidar_intrinsics"].contains("lidar_to_sensor_transform")) {
                throw std::runtime_error("Missing or invalid 'lidar_intrinsics.lidar_to_sensor_transform'");
            }

            columns_per_frame_ = metadata_["lidar_data_format"]["columns_per_frame"].get<int>();
            pixels_per_column_ = metadata_["lidar_data_format"]["pixels_per_column"].get<int>();
            columns_per_packet_ = metadata_["config_params"]["columns_per_packet"].get<int>();
            udp_profile_lidar_ = metadata_["config_params"]["udp_profile_lidar"].get<std::string>();

            const auto& beam_intrinsics = metadata_["beam_intrinsics"];
            lidar_origin_to_beam_origin_mm_ = beam_intrinsics["lidar_origin_to_beam_origin_mm"].get<double>();

            const auto& pixel_shift_by_row = metadata_["lidar_data_format"]["pixel_shift_by_row"];
            if (!pixel_shift_by_row.is_array() || pixel_shift_by_row.size() != static_cast<size_t>(pixels_per_column_)) {
                throw std::runtime_error("'pixel_shift_by_row' must be an array of " + std::to_string(pixels_per_column_) + " elements");
            }
            pixel_shifts_.resize(pixels_per_column_);
            for (int i = 0; i < pixels_per_column_; ++i) {
                pixel_shifts_[i] = pixel_shift_by_row[i].get<int>();
            }

            const auto& lidar_transform_json = metadata_["lidar_intrinsics"]["lidar_to_sensor_transform"];
            if (!lidar_transform_json.is_array() || lidar_transform_json.size() != 16) {
                throw std::runtime_error("'lidar_to_sensor_transform' must be an array of 16 elements");
            }
            
            Eigen::Matrix4d raw_transform = Eigen::Matrix4d::Identity(); // Not strictly needed if lidar_to_sensor_transform_ is filled directly
            lidar_to_sensor_transform_ = Eigen::Matrix4d::Identity();
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    double value = lidar_transform_json[i * 4 + j].get<double>();
                    // raw_transform(i, j) = value; // if needed for other purposes
                    // Scale translation components (last column, rows 0-2) from mm to m
                    if (j == 3 && i < 3) {
                        lidar_to_sensor_transform_(i, j) = value * 0.001;
                    } else {
                        lidar_to_sensor_transform_(i, j) = value;
                    }
                }
            }
        } catch (const json::exception& e) {
            throw std::runtime_error("JSON parsing error in metadata: " + std::string(e.what()));
        }
    }

    // -----------------------------------------------------------------------------

    void OusterLidarCallback::initialize() {

        if (udp_profile_lidar_ == "RNG19_RFL8_SIG16_NIR16") {
            PACKET_HEADER_BYTES = 32;     // Assumed size of data before the first measurement block in the packet (bytes)
            PACKET_FOOTER_BYTES = 32;     // Assumed size of data after the last measurement block
            COLUMN_HEADER_BYTES = 12;
            CHANNEL_STRIDE_BYTES = 12;
            MEASUREMENT_BLOCK_STATUS_BYTES = 0;
        } else if (udp_profile_lidar_ == "LEGACY") {
            PACKET_HEADER_BYTES = 0;     // Assumed size of data before the first measurement block in the packet (bytes)
            PACKET_FOOTER_BYTES = 0;     // Assumed size of data after the last measurement block
            COLUMN_HEADER_BYTES = 16;
            CHANNEL_STRIDE_BYTES = 12;
            MEASUREMENT_BLOCK_STATUS_BYTES = 4;
        } else {
            throw std::runtime_error("Unsupported udp_profile_lidar: " + udp_profile_lidar_);
        }

        // Calculate block_size and expected_size
        //single return:
        // block size = 12 + (128 * 12) + 0 = 1548
        // expected size = 32 + (16 * 1548) + 23 = 24823
        //legacy:
        // block size = 16 + (128 * 12) + 4 = 1556
        // expected size = 0 + (16 * 1556) + 0 = 24896
        block_size_ = COLUMN_HEADER_BYTES + (pixels_per_column_ * CHANNEL_STRIDE_BYTES) + MEASUREMENT_BLOCK_STATUS_BYTES;
        expected_size_ = PACKET_HEADER_BYTES + (columns_per_packet_ * block_size_) + PACKET_FOOTER_BYTES;

        const auto& beam_intrinsics = metadata_["beam_intrinsics"];
        const auto& azimuth_angles_json = beam_intrinsics["beam_azimuth_angles"];
        const auto& altitude_angles_json = beam_intrinsics["beam_altitude_angles"];

        if (!azimuth_angles_json.is_array() || azimuth_angles_json.size() != static_cast<size_t>(pixels_per_column_) ||
            !altitude_angles_json.is_array() || altitude_angles_json.size() != static_cast<size_t>(pixels_per_column_)) {
            throw std::runtime_error("Beam azimuth/altitude angles missing or wrong size in JSON.");
        }

        // Reserve buffer space
        data_buffer1_.reserve(columns_per_frame_ * pixels_per_column_);
        data_buffer2_.reserve(columns_per_frame_ * pixels_per_column_);

        sin_beam_azimuths_.resize(pixels_per_column_);
        cos_beam_azimuths_.resize(pixels_per_column_);
        sin_beam_altitudes_.resize(pixels_per_column_);
        cos_beam_altitudes_.resize(pixels_per_column_);
        r_min_.resize(pixels_per_column_, 0.1);

        // Initialize transposed lookup tables: [m_id][c_id]
        x_1_.assign(columns_per_frame_, std::vector<double, Eigen::aligned_allocator<double>>(pixels_per_column_));
        y_1_.assign(columns_per_frame_, std::vector<double, Eigen::aligned_allocator<double>>(pixels_per_column_));
        z_1_.assign(columns_per_frame_, std::vector<double, Eigen::aligned_allocator<double>>(pixels_per_column_));
        
        x_2_.resize(columns_per_frame_);
        y_2_.resize(columns_per_frame_);
        z_2_.resize(columns_per_frame_);

        // Transformation matrix: sensor (Ouster default: x=forward, y=left, z=up) to desired ROS-like (x=front, y=right, z=down)
        Eigen::Matrix4d sensor_to_desired_transform = Eigen::Matrix4d::Identity();
        sensor_to_desired_transform(0, 0) = -1.0;
        sensor_to_desired_transform(1, 1) = 1.0;
        sensor_to_desired_transform(2, 2) = -1.0;

        for (int i = 0; i < pixels_per_column_; ++i) {
            double az_deg = azimuth_angles_json[i].get<double>();
            double alt_deg = altitude_angles_json[i].get<double>();
            double az_rad = az_deg * M_PI / 180.0;
            double alt_rad = alt_deg * M_PI / 180.0;
            sin_beam_azimuths_[i] = std::sin(az_rad);
            cos_beam_azimuths_[i] = std::cos(az_rad);
            sin_beam_altitudes_[i] = std::sin(alt_rad);
            cos_beam_altitudes_[i] = std::cos(alt_rad);
        }

        for (int m_id = 0; m_id < columns_per_frame_; ++m_id) {
            double measurement_azimuth_rad = m_id * 2.0 * M_PI / columns_per_frame_;
            double cos_meas_az = std::cos(measurement_azimuth_rad);
            double sin_meas_az = std::sin(measurement_azimuth_rad);

            Eigen::Vector4d offset_lidar_frame(
                lidar_origin_to_beam_origin_mm_ * 0.001 * cos_meas_az,
                lidar_origin_to_beam_origin_mm_ * 0.001 * sin_meas_az,
                0.0,
                1.0);

            Eigen::Vector4d offset_transformed = sensor_to_desired_transform * lidar_to_sensor_transform_ * offset_lidar_frame;
            x_2_[m_id] = offset_transformed.x();
            y_2_[m_id] = offset_transformed.y();
            z_2_[m_id] = offset_transformed.z();

            for (int ch = 0; ch < pixels_per_column_; ++ch) {
                double beam_az_rad = azimuth_angles_json[ch].get<double>() * M_PI / 180.0;
                double total_az_rad = measurement_azimuth_rad + beam_az_rad;
                
                double cos_total_az = std::cos(total_az_rad);
                double sin_total_az = std::sin(total_az_rad);
                
                double cos_alt = cos_beam_altitudes_[ch];
                double sin_alt = sin_beam_altitudes_[ch];

                Eigen::Vector4d dir_lidar_frame(
                    cos_alt * cos_total_az,
                    cos_alt * sin_total_az,
                    sin_alt,
                    0.0);

                Eigen::Vector4d dir_transformed = sensor_to_desired_transform * lidar_to_sensor_transform_ * dir_lidar_frame;
                x_1_[m_id][ch] = dir_transformed.x();
                y_1_[m_id][ch] = dir_transformed.y();
                z_1_[m_id][ch] = dir_transformed.z();
            }
        }

        // Sanity checks for lookup table sizes
        if (x_1_.size() != static_cast<size_t>(columns_per_frame_) || (!x_1_.empty() && x_1_[0].size() != static_cast<size_t>(pixels_per_column_))) {
            throw std::runtime_error("x_1_ lookup table size mismatch after initialization");
        }
        if (x_2_.size() != static_cast<size_t>(columns_per_frame_)) {
            throw std::runtime_error("x_2_ lookup table size mismatch after initialization");
        }
        if (pixel_shifts_.size() != static_cast<size_t>(pixels_per_column_)) {
            throw std::runtime_error("pixel_shifts_ size mismatch after initialization");
        }

    #ifdef DEBUG
        if (!x_1_.empty() && !x_1_[0].empty()) {
            assert(reinterpret_cast<uintptr_t>(x_1_[0].data()) % 32 == 0 && "x_1_[0].data() not 32-byte aligned!");
        }
    #endif
    }

    // -----------------------------------------------------------------------------

    void OusterLidarCallback::decode_packet_single_return(const std::vector<uint8_t>& packet, lidarDecode::LidarDataFrame& frame) {
        if (packet.size() != expected_size_) {
            std::cerr << "Invalid packet size: " << packet.size() << ", expected: " << expected_size_ << std::endl;
            return;
        }

        uint16_t packet_type_raw;
        std::memcpy(&packet_type_raw, packet.data(), sizeof(uint16_t));
        uint16_t packet_type = le16toh(packet_type_raw);
        if (packet_type != 0x0001) { // Check against Ouster's Lidar Packet type (0x0001 for OS-series)
            std::cerr << "Invalid packet type: 0x" << std::hex << packet_type << std::dec << " (expected 0x1)" << std::endl;
            return;
        }

        uint16_t current_packet_frame_id_raw;
        std::memcpy(&current_packet_frame_id_raw, packet.data() + 2, sizeof(uint16_t)); // frame_id is at offset 2
        uint16_t current_packet_frame_id = le16toh(current_packet_frame_id_raw);

        lidarDecode::LidarDataFrame* p_current_write_buffer;
        if (buffer_toggle_) { // true: data_buffer1_ is read, data_buffer2_ is write
            p_current_write_buffer = &data_buffer2_;
        } else { // false: data_buffer2_ is read, data_buffer1_ is write
            p_current_write_buffer = &data_buffer1_;
        }

        double prev_frame_completed_latest_ts = 0.0;

        if (current_packet_frame_id != this->frame_id_) {
            // Frame transition detected. The frame in p_current_write_buffer is now complete.
            if (this->frame_id_ != 0 || this->number_points_ > 0) { // Avoid operations for uninitialized/empty first "previous" frame
                p_current_write_buffer->numberpoints = this->number_points_; // p_current_write_buffer->timestamp and ->frame_id were set by the first point of that frame.
            }
            
            prev_frame_completed_latest_ts = this->latest_timestamp_s; // Timestamp of the last point of the frame that just completed.

            swap_buffers(); // Flips buffer_toggle_. Roles of data_buffer1_ and data_buffer2_ are swapped.

            // Update p_current_write_buffer to point to the NEW write buffer.
            if (buffer_toggle_) {
                p_current_write_buffer = &data_buffer2_;
            } else {
                p_current_write_buffer = &data_buffer1_;
            }

            this->number_points_ = 0; // Reset point count for the new frame.
            this->frame_id_ = current_packet_frame_id; // Update the class's current frame_id tracker.

            p_current_write_buffer->clear();
            p_current_write_buffer->reserve(columns_per_frame_ * pixels_per_column_);
        }

        bool is_first_point_of_current_frame = (this->number_points_ == 0);

        for (int col = 0; col < columns_per_packet_; ++col) {
            size_t block_offset = PACKET_HEADER_BYTES + col * block_size_; // header_size_ is offset to first block from packet start

            uint64_t timestamp_ns_raw;
            std::memcpy(&timestamp_ns_raw, packet.data() + block_offset, sizeof(uint64_t));
            uint64_t timestamp_ns = le64toh(timestamp_ns_raw);
            double current_col_timestamp_s = static_cast<double>(timestamp_ns) * 1e-9;
            
            if (current_col_timestamp_s < 0) {
                std::cerr << "Negative column timestamp: " << current_col_timestamp_s << std::endl;
                continue;
            }
            this->latest_timestamp_s = current_col_timestamp_s; // Update class member with latest processed timestamp

            uint16_t m_id_raw;
            std::memcpy(&m_id_raw, packet.data() + block_offset + 8, sizeof(uint16_t)); // Measurement ID is at offset 8 in column block
            uint16_t m_id = le16toh(m_id_raw);

            if (m_id >= static_cast<uint16_t>(columns_per_frame_)) {
                std::cerr << "Invalid measurement ID: " << m_id << " (>= " << columns_per_frame_ << ")" << std::endl;
                continue;
            }

            // Column status is at offset 10 in column block. Bit 0 indicates valid data.
            uint8_t column_status;
            std::memcpy(&column_status, packet.data() + block_offset + 10, sizeof(uint8_t));
            if (!(column_status & 0x01)) { // Check if measurement block is valid
                continue;
            }

    #ifdef __AVX2__
            for (uint16_t c_id_base = 0; c_id_base < static_cast<uint16_t>(pixels_per_column_); c_id_base += 4) {
                // Offset to first pixel data in column block is 12 bytes (ts 8B, m_id 2B, status 1B, encoder 2B -> total 12B before pixels)
                size_t first_pixel_in_block_offset = block_offset + 12; 
                size_t pixel_group_offset = first_pixel_in_block_offset + c_id_base * 12; // 12 bytes per pixel data

                alignas(32) double range_m[4];
                alignas(32) double r_min_vals[4];
                uint16_t c_ids[4];
                uint8_t reflectivity[4]; // Reflectivity is uint8_t
                uint16_t signal[4], nir[4];

                for (int i = 0; i < 4; ++i) {
                    uint16_t current_c_id = c_id_base + i;
                    if (current_c_id >= static_cast<uint16_t>(pixels_per_column_)) {
                        range_m[i] = 0.0; // Mark as invalid if past actual pixels_per_column
                        r_min_vals[i] = 1.0; // Ensure it fails range_m < r_min_vals if range_m is 0
                        continue;
                    }
                    
                    size_t pixel_data_offset = first_pixel_in_block_offset + current_c_id * 12;
                    if (pixel_data_offset + 11 >= packet.size()) { // Check bounds for reading up to 12 bytes
                        range_m[i] = 0.0; r_min_vals[i] = 1.0; continue;
                    }

                    uint32_t range_mm_raw;
                    // Ouster range is 3 bytes, then 1 byte reflectivity. Read 3 bytes into lower part of uint32_t.
                    uint8_t range_bytes[4] = {packet[pixel_data_offset], packet[pixel_data_offset + 1], packet[pixel_data_offset + 2], 0};
                    std::memcpy(&range_mm_raw, range_bytes, sizeof(uint32_t));
                    uint32_t range_mm = le32toh(range_mm_raw) & 0x0007FFFF; // Ouster new format range is 19 bits for REV06+
                                                                        // User mentioned "not legacy format", older non-legacy could be 0x0007FFFF (19 bits)
                    range_m[i] = static_cast<double>(range_mm) * 0.001; // mm to m
                    r_min_vals[i] = r_min_[current_c_id];
                    c_ids[i] = current_c_id;

                    std::memcpy(&reflectivity[i], packet.data() + pixel_data_offset + 4, sizeof(uint8_t)); // Reflectivity (1B)
                    uint16_t signal_raw, nir_raw;
                    std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t)); // Signal (2B)
                    std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));   // NIR (2B)
                    signal[i] = le16toh(signal_raw);
                    nir[i] = le16toh(nir_raw);
                }

                __m256d m256d_range = _mm256_load_pd(range_m);
                __m256d m256d_r_min_vec = _mm256_load_pd(r_min_vals);
                // Valid points are those where range_m >= r_min_val
                __m256d valid_mask = _mm256_cmp_pd(m256d_range, m256d_r_min_vec, _CMP_GE_OQ);

                __m256d x1_vec = _mm256_load_pd(x_1_[m_id].data() + c_id_base);
                __m256d y1_vec = _mm256_load_pd(y_1_[m_id].data() + c_id_base);
                __m256d z1_vec = _mm256_load_pd(z_1_[m_id].data() + c_id_base);
                __m256d x2_val = _mm256_set1_pd(x_2_[m_id]);
                __m256d y2_val = _mm256_set1_pd(y_2_[m_id]);
                __m256d z2_val = _mm256_set1_pd(z_2_[m_id]);

                __m256d pt_x = _mm256_fmadd_pd(m256d_range, x1_vec, x2_val);
                __m256d pt_y = _mm256_fmadd_pd(m256d_range, y1_vec, y2_val);
                __m256d pt_z = _mm256_fmadd_pd(m256d_range, z1_vec, z2_val);

                alignas(32) double pt_x_arr[4], pt_y_arr[4], pt_z_arr[4];
                _mm256_store_pd(pt_x_arr, pt_x);
                _mm256_store_pd(pt_y_arr, pt_y);
                _mm256_store_pd(pt_z_arr, pt_z);
                
                alignas(32) double valid_mask_arr[4]; // Store mask to check elements
                _mm256_store_pd(valid_mask_arr, valid_mask);

                double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                    ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                    : 0.0;

                for (int i = 0; i < 4; ++i) {
                    uint16_t current_c_id = c_id_base + i;
                    if (current_c_id >= static_cast<uint16_t>(pixels_per_column_)) break; // Processed all valid pixels for this group

                    // Check if the point is valid based on the AVX comparison (mask will be all 1s (NaN) for true, 0 for false)
                    // and also the original range check (in case AVX was skipped due to boundary conditions)
                    if (range_m[i] >= r_min_vals[i] && range_m[i] > 0) { // Double check scalar condition
                        p_current_write_buffer->x.push_back(pt_x_arr[i]);
                        p_current_write_buffer->y.push_back(pt_y_arr[i]);
                        p_current_write_buffer->z.push_back(pt_z_arr[i]);
                        p_current_write_buffer->c_id.push_back(c_ids[i]);
                        p_current_write_buffer->m_id.push_back(m_id);
                        p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                        p_current_write_buffer->relative_timestamp.push_back(relative_timestamp_s);
                        p_current_write_buffer->reflectivity.push_back(reflectivity[i]);
                        p_current_write_buffer->signal.push_back(signal[i]);
                        p_current_write_buffer->nir.push_back(nir[i]);
                        
                        this->number_points_++;
                        if (is_first_point_of_current_frame) {
                            p_current_write_buffer->timestamp = current_col_timestamp_s;
                            p_current_write_buffer->frame_id = this->frame_id_;
                            p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                                ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                            is_first_point_of_current_frame = false;
                        }
                    }
                }
            }
    #else // Scalar fallback
            size_t first_pixel_in_block_offset = block_offset + 12;
            for (uint16_t c_id = 0; c_id < static_cast<uint16_t>(pixels_per_column_); ++c_id) {
                size_t pixel_data_offset = first_pixel_in_block_offset + c_id * 12;
                if (pixel_data_offset + 11 >= packet.size()) { // Check bounds for reading up to 12 bytes
                    // std::cerr << "Pixel offset out of bounds\n"; // Can be noisy
                    continue;
                }

                uint32_t range_mm_raw;
                uint8_t range_bytes[4] = {packet[pixel_data_offset], packet[pixel_data_offset + 1], packet[pixel_data_offset + 2], 0};
                std::memcpy(&range_mm_raw, range_bytes, sizeof(uint32_t));
                uint32_t range_mm = le32toh(range_mm_raw) & 0x0007FFFF; // 19-bit range
                double range_m = static_cast<double>(range_mm) * 0.001;

                if (range_m < r_min_[c_id] || range_m == 0) { // Skip if below min range or zero
                    continue;
                }

                uint8_t current_reflectivity;
                std::memcpy(&current_reflectivity, packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
                uint16_t signal_raw, nir_raw;
                std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
                std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
                uint16_t current_signal = le16toh(signal_raw);
                uint16_t current_nir = le16toh(nir_raw);

                double pt_x = range_m * x_1_[m_id][c_id] + x_2_[m_id];
                double pt_y = range_m * y_1_[m_id][c_id] + y_2_[m_id];
                double pt_z = range_m * z_1_[m_id][c_id] + z_2_[m_id];

                double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                    ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                    : 0.0;

                p_current_write_buffer->x.push_back(pt_x);
                p_current_write_buffer->y.push_back(pt_y);
                p_current_write_buffer->z.push_back(pt_z);
                p_current_write_buffer->c_id.push_back(c_id);
                p_current_write_buffer->m_id.push_back(m_id);
                p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                p_current_write_buffer->relative_timestamp.push_back(relative_timestamp_s);
                p_current_write_buffer->reflectivity.push_back(current_reflectivity);
                p_current_write_buffer->signal.push_back(current_signal);
                p_current_write_buffer->nir.push_back(current_nir);
                
                this->number_points_++;
                if (is_first_point_of_current_frame) {
                    p_current_write_buffer->timestamp = current_col_timestamp_s;
                    p_current_write_buffer->frame_id = this->frame_id_;
                    p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                        ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                    is_first_point_of_current_frame = false;
                }
            }
    #endif
        } // End of column loop

        // Update number of points for the buffer currently being written to,
        // reflecting points added from this packet.
        if (p_current_write_buffer) { // Should always be true here
            p_current_write_buffer->numberpoints = this->number_points_;
        }
        
        // Assign the current "read" buffer (which holds the last completed frame) to the output parameter.
        frame = get_latest_lidar_frame();
    }

    // -----------------------------------------------------------------------------

    void OusterLidarCallback::decode_packet_legacy(const std::vector<uint8_t>& packet, lidarDecode::LidarDataFrame& frame) {
        if (packet.size() != expected_size_) {
            std::cerr << "Invalid packet size: " << packet.size() << ", expected: " << expected_size_ << std::endl;
            return;
        }

        lidarDecode::LidarDataFrame* p_current_write_buffer;
        if (buffer_toggle_) {
            p_current_write_buffer = &data_buffer2_;
        } else {
            p_current_write_buffer = &data_buffer1_;
        }

        double prev_frame_completed_latest_ts = 0.0;

        for (int col = 0; col < columns_per_packet_; ++col) {
            size_t block_offset = col * block_size_; // PACKET_HEADER_BYTES in legacy format is set to 0

            uint64_t timestamp_ns_raw;
            std::memcpy(&timestamp_ns_raw, packet.data() + block_offset, sizeof(uint64_t));
            uint64_t timestamp_ns = le64toh(timestamp_ns_raw);
            double current_col_timestamp_s = static_cast<double>(timestamp_ns) * 1e-9;

            if (current_col_timestamp_s < 0) {
                std::cerr << "Negative column timestamp: " << current_col_timestamp_s << std::endl;
                continue;
            }
            this->latest_timestamp_s = current_col_timestamp_s;

            uint16_t m_id_raw;
            std::memcpy(&m_id_raw, packet.data() + block_offset + 8, sizeof(uint16_t));
            uint16_t m_id = le16toh(m_id_raw);

            if (m_id >= static_cast<uint16_t>(columns_per_frame_)) {
                std::cerr << "Invalid measurement ID: " << m_id << " (>= " << columns_per_frame_ << ")" << std::endl;
                continue;
            }

            uint16_t current_packet_frame_id_raw;
            std::memcpy(&current_packet_frame_id_raw, packet.data() + block_offset + 10, sizeof(uint16_t));
            uint16_t current_packet_frame_id = le16toh(current_packet_frame_id_raw);

            if (current_packet_frame_id != this->frame_id_) {
                if (this->frame_id_ != 0 || this->number_points_ > 0) {
                    p_current_write_buffer->numberpoints = this->number_points_;
                }

                prev_frame_completed_latest_ts = this->latest_timestamp_s;
                swap_buffers();

                if (buffer_toggle_) {
                    p_current_write_buffer = &data_buffer2_;
                } else {
                    p_current_write_buffer = &data_buffer1_;
                }

                this->number_points_ = 0;
                this->frame_id_ = current_packet_frame_id;

                p_current_write_buffer->clear();
                p_current_write_buffer->reserve(columns_per_frame_ * pixels_per_column_);
            }

            // Measurement block status is after header (16 bytes) and channel data (pixels_per_column_ * 12 bytes)
            uint32_t block_status;
            size_t status_offset = block_offset + 16 + (pixels_per_column_ * 12);
            std::memcpy(&block_status, packet.data() + status_offset, sizeof(uint32_t));
            block_status = le32toh(block_status);
            if (block_status != 0xFFFFFFFF) {
                continue;
            }

            bool is_first_point_of_current_frame = (this->number_points_ == 0);

    #ifdef __AVX2__
            for (uint16_t c_id_base = 0; c_id_base < static_cast<uint16_t>(pixels_per_column_); c_id_base += 4) {
                size_t first_pixel_in_block_offset = block_offset + 16;
                size_t pixel_group_offset = first_pixel_in_block_offset + c_id_base * 12;

                alignas(32) double range_m[4];
                alignas(32) double r_min_vals[4];
                uint16_t c_ids[4];
                uint8_t reflectivity[4];
                uint16_t signal[4], nir[4];

                for (int i = 0; i < 4; ++i) {
                    uint16_t current_c_id = c_id_base + i;
                    if (current_c_id >= static_cast<uint16_t>(pixels_per_column_)) {
                        range_m[i] = 0.0;
                        r_min_vals[i] = 1.0;
                        continue;
                    }

                    size_t pixel_data_offset = first_pixel_in_block_offset + current_c_id * 12;
                    if (pixel_data_offset + 11 >= packet.size()) {
                        range_m[i] = 0.0;
                        r_min_vals[i] = 1.0;
                        continue;
                    }

                    uint32_t range_mm_raw;
                    std::memcpy(&range_mm_raw, packet.data() + pixel_data_offset, sizeof(uint32_t));
                    uint32_t range_mm = le32toh(range_mm_raw) & 0x000FFFFF; // 20-bit range
                    range_m[i] = static_cast<double>(range_mm) * 0.001;
                    r_min_vals[i] = r_min_[current_c_id];
                    c_ids[i] = current_c_id;

                    std::memcpy(&reflectivity[i], packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
                    uint16_t signal_raw, nir_raw;
                    std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
                    std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
                    signal[i] = le16toh(signal_raw);
                    nir[i] = le16toh(nir_raw);
                }

                __m256d m256d_range = _mm256_load_pd(range_m);
                __m256d m256d_r_min_vec = _mm256_load_pd(r_min_vals);
                __m256d valid_mask = _mm256_cmp_pd(m256d_range, m256d_r_min_vec, _CMP_GE_OQ);

                __m256d x1_vec = _mm256_load_pd(x_1_[m_id].data() + c_id_base);
                __m256d y1_vec = _mm256_load_pd(y_1_[m_id].data() + c_id_base);
                __m256d z1_vec = _mm256_load_pd(z_1_[m_id].data() + c_id_base);
                __m256d x2_val = _mm256_set1_pd(x_2_[m_id]);
                __m256d y2_val = _mm256_set1_pd(y_2_[m_id]);
                __m256d z2_val = _mm256_set1_pd(z_2_[m_id]);

                __m256d pt_x = _mm256_fmadd_pd(m256d_range, x1_vec, x2_val);
                __m256d pt_y = _mm256_fmadd_pd(m256d_range, y1_vec, y2_val);
                __m256d pt_z = _mm256_fmadd_pd(m256d_range, z1_vec, z2_val);

                alignas(32) double pt_x_arr[4], pt_y_arr[4], pt_z_arr[4];
                _mm256_store_pd(pt_x_arr, pt_x);
                _mm256_store_pd(pt_y_arr, pt_y);
                _mm256_store_pd(pt_z_arr, pt_z);

                alignas(32) double valid_mask_arr[4];
                _mm256_store_pd(valid_mask_arr, valid_mask);

                double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                    ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                    : 0.0;

                for (int i = 0; i < 4; ++i) {
                    uint16_t current_c_id = c_id_base + i;
                    if (current_c_id >= static_cast<uint16_t>(pixels_per_column_)) break;

                    if (range_m[i] >= r_min_vals[i] && range_m[i] > 0) {
                        p_current_write_buffer->x.push_back(pt_x_arr[i]);
                        p_current_write_buffer->y.push_back(pt_y_arr[i]);
                        p_current_write_buffer->z.push_back(pt_z_arr[i]);
                        p_current_write_buffer->c_id.push_back(c_ids[i]);
                        p_current_write_buffer->m_id.push_back(m_id);
                        p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                        p_current_write_buffer->relative_timestamp.push_back(relative_timestamp_s);
                        p_current_write_buffer->reflectivity.push_back(reflectivity[i]);
                        p_current_write_buffer->signal.push_back(signal[i]);
                        p_current_write_buffer->nir.push_back(nir[i]);

                        this->number_points_++;
                        if (is_first_point_of_current_frame) {
                            p_current_write_buffer->timestamp = current_col_timestamp_s;
                            p_current_write_buffer->frame_id = this->frame_id_;
                            p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                                ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                            is_first_point_of_current_frame = false;
                        }
                    }
                }
            }
    #else
            size_t first_pixel_in_block_offset = block_offset + 16;
            for (uint16_t c_id = 0; c_id < static_cast<uint16_t>(pixels_per_column_); ++c_id) {
                size_t pixel_data_offset = first_pixel_in_block_offset + c_id * 12;
                if (pixel_data_offset + 11 >= packet.size()) {
                    continue;
                }

                uint32_t range_mm_raw;
                std::memcpy(&range_mm_raw, packet.data() + pixel_data_offset, sizeof(uint32_t));
                uint32_t range_mm = le32toh(range_mm_raw) & 0x000FFFFF; // 20-bit range
                double range_m = static_cast<double>(range_mm) * 0.001;

                if (range_m < r_min_[c_id] || range_m == 0) {
                    continue;
                }

                uint8_t current_reflectivity;
                std::memcpy(&current_reflectivity, packet.data() + pixel_data_offset + 4, sizeof(uint8_t));
                uint16_t signal_raw, nir_raw;
                std::memcpy(&signal_raw, packet.data() + pixel_data_offset + 6, sizeof(uint16_t));
                std::memcpy(&nir_raw, packet.data() + pixel_data_offset + 8, sizeof(uint16_t));
                uint16_t current_signal = le16toh(signal_raw);
                uint16_t current_nir = le16toh(nir_raw);

                double pt_x = range_m * x_1_[m_id][c_id] + x_2_[m_id];
                double pt_y = range_m * y_1_[m_id][c_id] + y_2_[m_id];
                double pt_z = range_m * z_1_[m_id][c_id] + z_2_[m_id];

                double relative_timestamp_s = (p_current_write_buffer->numberpoints > 0 || this->number_points_ > 0) && p_current_write_buffer->timestamp > 0
                    ? std::max(0.0, current_col_timestamp_s - p_current_write_buffer->timestamp)
                    : 0.0;

                p_current_write_buffer->x.push_back(pt_x);
                p_current_write_buffer->y.push_back(pt_y);
                p_current_write_buffer->z.push_back(pt_z);
                p_current_write_buffer->c_id.push_back(c_id);
                p_current_write_buffer->m_id.push_back(m_id);
                p_current_write_buffer->timestamp_points.push_back(current_col_timestamp_s);
                p_current_write_buffer->relative_timestamp.push_back(relative_timestamp_s);
                p_current_write_buffer->reflectivity.push_back(current_reflectivity);
                p_current_write_buffer->signal.push_back(current_signal);
                p_current_write_buffer->nir.push_back(current_nir);

                this->number_points_++;
                if (is_first_point_of_current_frame) {
                    p_current_write_buffer->timestamp = current_col_timestamp_s;
                    p_current_write_buffer->frame_id = this->frame_id_;
                    p_current_write_buffer->interframe_timedelta = (prev_frame_completed_latest_ts > 0.0)
                        ? std::max(0.0, current_col_timestamp_s - prev_frame_completed_latest_ts) : 0.0;
                    is_first_point_of_current_frame = false;
                }
            }
    #endif
        }

        if (p_current_write_buffer) {
            p_current_write_buffer->numberpoints = this->number_points_;
        }

        frame = get_latest_lidar_frame();
    }

    // -----------------------------------------------------------------------------

    void OusterLidarCallback::decode_packet_LidarIMU(const std::vector<uint8_t>& packet, lidarDecode::LidarIMUDataFrame& frame) {
            // Expected IMU packet size: 48 bytes (8+8+8+4+4+4+4+4+4)
            const size_t EXPECTED_IMU_PACKET_SIZE = 48;
            if (packet.size() != EXPECTED_IMU_PACKET_SIZE) {
                std::cerr << "Invalid IMU packet size: " << packet.size() 
                        << ", expected: " << EXPECTED_IMU_PACKET_SIZE << std::endl;
                frame.clear();
                return;
            }

            size_t offset = 0;

            // IMU Diagnostic Time (uint64_t, 8 bytes)
            uint64_t diag_time_raw;
            std::memcpy(&diag_time_raw, packet.data() + offset, sizeof(uint64_t));
            frame.IMU_Diagnostic_Time_s = static_cast<double>(le64toh(diag_time_raw)) / 1e9;
            offset += 8;

            // Accelerometer Read Time (uint64_t, 8 bytes)
            uint64_t accel_time_raw;
            std::memcpy(&accel_time_raw, packet.data() + offset, sizeof(uint64_t));
            frame.Accelerometer_Read_Time_s = static_cast<double>(le64toh(accel_time_raw)) / 1e9;
            offset += 8;

            // Gyroscope Read Time (uint64_t, 8 bytes)
            uint64_t gyro_time_raw;
            std::memcpy(&gyro_time_raw, packet.data() + offset, sizeof(uint64_t));
            frame.Gyroscope_Read_Time_s = static_cast<double>(le64toh(gyro_time_raw)) / 1e9;
            offset += 8;

            // Acceleration X (float, 4 bytes)
            std::memcpy(&frame.Acceleration_X, packet.data() + offset, sizeof(float));
            offset += 4;

            // Acceleration Y (float, 4 bytes)
            std::memcpy(&frame.Acceleration_Y, packet.data() + offset, sizeof(float));
            offset += 4;

            // Acceleration Z (float, 4 bytes)
            std::memcpy(&frame.Acceleration_Z, packet.data() + offset, sizeof(float));
            offset += 4;

            // Angular Velocity X (float, 4 bytes)
            std::memcpy(&frame.AngularVelocity_X, packet.data() + offset, sizeof(float));
            offset += 4;

            // Angular Velocity Y (float, 4 bytes)
            std::memcpy(&frame.AngularVelocity_Y, packet.data() + offset, sizeof(float));
            offset += 4;

            // Angular Velocity Z (float, 4 bytes)
            std::memcpy(&frame.AngularVelocity_Z, packet.data() + offset, sizeof(float));
            // offset += 4; // Total offset = 48, end of packet

            // Validate timestamps
            if (frame.Accelerometer_Read_Time_s == 0.0 || frame.Gyroscope_Read_Time_s == 0.0) {
                std::cerr << "Invalid IMU timestamps: Accel=" << frame.Accelerometer_Read_Time_s 
                        << ", Gyro=" << frame.Gyroscope_Read_Time_s << std::endl;
                frame.clear();
                return;
            }

            // Compute normalized timestamp (average of accel and gyro times)
            const uint64_t max_diff_ns = 1000000; // 1ms threshold
            uint64_t accel_time_ns = static_cast<uint64_t>(frame.Accelerometer_Read_Time_s * 1e9);
            uint64_t gyro_time_ns = static_cast<uint64_t>(frame.Gyroscope_Read_Time_s * 1e9);
            uint64_t diff = (accel_time_ns > gyro_time_ns) 
                            ? (accel_time_ns - gyro_time_ns) 
                            : (gyro_time_ns - accel_time_ns);
            if (diff > max_diff_ns) {
                std::cerr << "[packet LidarIMU] Timestamp difference accel and gyro too large: " << diff << " ns" << std::endl;
                frame.clear();
                return;
            }

            frame.Normalized_Timestamp_s = (frame.Accelerometer_Read_Time_s + frame.Gyroscope_Read_Time_s) / 2.0;
        }
        
    // -----------------------------------------------------------------------------

} // namespace dynamicMap

