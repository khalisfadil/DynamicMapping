#include <pipeline.hpp>

namespace dynamicMap {

    std::atomic<bool> Pipeline::running_{true};
    std::condition_variable Pipeline::globalCV_;
    boost::lockfree::spsc_queue<lidarDecode::LidarDataFrame, boost::lockfree::capacity<128>> Pipeline::decodedPoint_buffer_;
    boost::lockfree::spsc_queue<lidarDecode::LidarIMUDataFrame, boost::lockfree::capacity<128>> Pipeline::decodedLidarIMU_buffer_;
    boost::lockfree::spsc_queue<std::vector<decodeNav::DataFrameNavMsg>, boost::lockfree::capacity<128>> Pipeline::decodedNav_buffer_;
    boost::lockfree::spsc_queue<dynamicMap::NavDataFrame, boost::lockfree::capacity<128>> Pipeline::interpolatedNav_buffer_;
    boost::lockfree::spsc_queue<VizuDataFrame, boost::lockfree::capacity<128>> Pipeline::vizu_buffer_;
    // -----------------------------------------------------------------------------

    Pipeline::Pipeline(const std::string& json_path) : lidarCallback(json_path) {}
    // Pipeline::~Pipeline() {}

    // -----------------------------------------------------------------------------

    void Pipeline::signalHandler(int signal) {
        if (signal == SIGINT || signal == SIGTERM) {
            running_.store(false, std::memory_order_release);
            globalCV_.notify_all();

            constexpr const char* message = "[signalHandler] Shutting down...\n";
            constexpr size_t messageLen = sizeof(message) - 1;
            ssize_t result = write(STDOUT_FILENO, message, messageLen);
        }
    }

    // -----------------------------------------------------------------------------

    void Pipeline::setThreadAffinity(const std::vector<int>& coreIDs) {
        if (coreIDs.empty()) {return;}
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        const unsigned int maxCores = std::thread::hardware_concurrency();
        uint32_t validCores = 0;

        for (int coreID : coreIDs) {
            if (coreID >= 0 && static_cast<unsigned>(coreID) < maxCores) {
                CPU_SET(coreID, &cpuset);
                validCores |= (1 << coreID);
            }
        }
        if (!validCores) {
                return;
            }

        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
            running_.store(false); // Optionally terminate
        }
    }

    // -----------------------------------------------------------------------------

    void Pipeline::runOusterLidarListenerSingleReturn(boost::asio::io_context& ioContext,
                                        const std::string& host,
                                        uint16_t port,
                                        uint32_t bufferSize,
                                        const std::vector<int>& allowedCores) {
        
        setThreadAffinity(allowedCores); // Sets affinity for this listener thread

        if (host.empty() || port == 0) {
            std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
            std::cerr << "[Pipeline] Listener: Invalid host or port. Host: " << host << ", Port: " << port << std::endl;
            return;
        }

        try {
        lidarDecode::UdpSocket listener(ioContext, host, port,
            // Lambda callback:
            [&](const std::vector<uint8_t>& packet_data) {
                // LidarDataFrame frame_data_copy; // 1. Local DataFrame created.
                                        //    It will hold a deep copy of the lidar data.

                // 2. lidarCallback processes the packet.
                //    Inside decode_packet_single_return, frame_data_copy is assigned
                //    (via DataFrame::operator=) the contents of lidarCallback's completed buffer.
                //    This results in a deep copy into frame_data_copy.
                lidarCallback.decode_packet_single_return(packet_data, frame_data_copy_);

                // 3. Now frame_data_copy is an independent, deep copy of the relevant frame.
                //    We can safely use it and then move it into the queue.
                if (frame_data_copy_.numberpoints > 0 && frame_data_copy_.frame_id != this->frame_id_) {
                    this->frame_id_ = frame_data_copy_.frame_id;
                    
                    // 4. Move frame_data_copy into the SPSC queue.
                    //    This transfers ownership of frame_data_copy's internal resources (vector data)
                    //    to the element constructed in the queue, avoiding another full copy.
                    //    frame_data_copy is left in a valid but unspecified (likely empty) state.
                    if (!decodedPoint_buffer_.push(std::move(frame_data_copy_))) {
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame " 
                                << this->frame_id_ // Use this->frame_id_ as frame_data_copy might be moved-from
                                << ". Buffer Lidar Points might be full." << std::endl;
                    }
                }
                // frame_data_copy goes out of scope here. If it was moved, its destruction is trivial.
                // If it was not pushed (e.g., due to condition not met), it's destructed normally (releasing its copied data).
            }, // End of lambda
            bufferSize);

            // Main loop to run Asio's I/O event processing.
            while (running_.load(std::memory_order_acquire)) {
                try {
                    ioContext.run(); // This will block until work is done or ioContext is stopped.
                                    // If it returns without an exception, it implies all work is done.
                    if (!running_.load(std::memory_order_acquire)) { // Check running_ again if run() returned cleanly
                        break;
                    }
                    // If run() returns and there's still potentially work (or to handle stop signals),
                    // you might need to reset and run again, or break if shutting down.
                    // For a continuous listener, run() might not return unless stopped or an error occurs.
                    // If ioContext.run() returns because it ran out of work, and we are still 'running_',
                    // we should probably restart it if the intent is to keep listening.
                    // However, typically for a server/listener, io_context.run() is expected to block until stop() is called.
                    // If it returns prematurely, ensure io_context is reset if needed before next run() call.
                    // For this pattern, if run() returns, we break, assuming stop() was called elsewhere or an error occurred.
                    break; 
                } catch (const std::exception& e) {
                    // Handle exceptions from ioContext.run()
                    std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
                    std::cerr << "[Pipeline] Listener: Exception in ioContext.run(): " << e.what() << std::endl;
                    if (running_.load(std::memory_order_acquire)) {
                        ioContext.restart(); // Restart Asio io_context to attempt recovery.
                        std::cerr << "[Pipeline] Listener: ioContext restarted." << std::endl;
                    } else {
                        break; // Exit loop if shutting down.
                    }
                }
            }
        }
        catch(const std::exception& e){
            // Handle exceptions from UdpSocket creation or other setup.
            std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
            std::cerr << "[Pipeline] Listener: Setup exception: " << e.what() << std::endl;
        }

        // Ensure ioContext is stopped when the listener is done or an error occurs.
        if (!ioContext.stopped()) {
            ioContext.stop();
        }
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cerr << "[Pipeline] Ouster LiDAR listener stopped." << std::endl;
    }

    // -----------------------------------------------------------------------------

    void Pipeline::runOusterLidarListenerLegacy(boost::asio::io_context& ioContext,
                                        const std::string& host,
                                        uint16_t port,
                                        uint32_t bufferSize,
                                        const std::vector<int>& allowedCores) {
        
        setThreadAffinity(allowedCores); // Sets affinity for this listener thread

        if (host.empty() || port == 0) {
            std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
            std::cerr << "[Pipeline] Listener: Invalid host or port. Host: " << host << ", Port: " << port << std::endl;
            return;
        }

        try {
        lidarDecode::UdpSocket listener(ioContext, host, port,
            // Lambda callback:
            [&](const std::vector<uint8_t>& packet_data) {
                // LidarDataFrame frame_data_copy; // 1. Local DataFrame created.
                                        //    It will hold a deep copy of the lidar data.

                // 2. lidarCallback processes the packet.
                //    Inside decode_packet_single_return, frame_data_copy is assigned
                //    (via DataFrame::operator=) the contents of lidarCallback's completed buffer.
                //    This results in a deep copy into frame_data_copy.
                lidarCallback.decode_packet_legacy(packet_data, frame_data_copy_);

                // 3. Now frame_data_copy is an independent, deep copy of the relevant frame.
                //    We can safely use it and then move it into the queue.
                if (frame_data_copy_.numberpoints > 0 && frame_data_copy_.frame_id != this->frame_id_) {
                    this->frame_id_ = frame_data_copy_.frame_id;
                    
                    // 4. Move frame_data_copy into the SPSC queue.
                    //    This transfers ownership of frame_data_copy's internal resources (vector data)
                    //    to the element constructed in the queue, avoiding another full copy.
                    //    frame_data_copy is left in a valid but unspecified (likely empty) state.
                    if (!decodedPoint_buffer_.push(std::move(frame_data_copy_))) {
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame " 
                                << this->frame_id_ // Use this->frame_id_ as frame_data_copy might be moved-from
                                << ". Buffer Lidar Points might be full." << std::endl;
                    }
                }
                // frame_data_copy goes out of scope here. If it was moved, its destruction is trivial.
                // If it was not pushed (e.g., due to condition not met), it's destructed normally (releasing its copied data).
            }, // End of lambda
            bufferSize);

            // Main loop to run Asio's I/O event processing.
            while (running_.load(std::memory_order_acquire)) {
                try {
                    ioContext.run(); // This will block until work is done or ioContext is stopped.
                                    // If it returns without an exception, it implies all work is done.
                    if (!running_.load(std::memory_order_acquire)) { // Check running_ again if run() returned cleanly
                        break;
                    }
                    // If run() returns and there's still potentially work (or to handle stop signals),
                    // you might need to reset and run again, or break if shutting down.
                    // For a continuous listener, run() might not return unless stopped or an error occurs.
                    // If ioContext.run() returns because it ran out of work, and we are still 'running_',
                    // we should probably restart it if the intent is to keep listening.
                    // However, typically for a server/listener, io_context.run() is expected to block until stop() is called.
                    // If it returns prematurely, ensure io_context is reset if needed before next run() call.
                    // For this pattern, if run() returns, we break, assuming stop() was called elsewhere or an error occurred.
                    break; 
                } catch (const std::exception& e) {
                    // Handle exceptions from ioContext.run()
                    std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
                    std::cerr << "[Pipeline] Listener: Exception in ioContext.run(): " << e.what() << std::endl;
                    if (running_.load(std::memory_order_acquire)) {
                        ioContext.restart(); // Restart Asio io_context to attempt recovery.
                        std::cerr << "[Pipeline] Listener: ioContext restarted." << std::endl;
                    } else {
                        break; // Exit loop if shutting down.
                    }
                }
            }
        }
        catch(const std::exception& e){
            // Handle exceptions from UdpSocket creation or other setup.
            std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
            std::cerr << "[Pipeline] Listener: Setup exception: " << e.what() << std::endl;
        }

        // Ensure ioContext is stopped when the listener is done or an error occurs.
        if (!ioContext.stopped()) {
            ioContext.stop();
        }
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cerr << "[Pipeline] Ouster LiDAR listener stopped." << std::endl;
    }

    // -----------------------------------------------------------------------------

    void Pipeline::runNavMsgListener(boost::asio::io_context& ioContext,
                                        const std::string& host,
                                        uint16_t port,
                                        uint32_t bufferSize,
                                        const std::vector<int>& allowedCores) {
        
        setThreadAffinity(allowedCores); // Sets affinity for this listener thread

        if (host.empty() || port == 0) {
            std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
            std::cerr << "[Pipeline] Listener: Invalid host or port. Host: " << host << ", Port: " << port << std::endl;
            return;
        }

        try {
        lidarDecode::UdpSocket listener(ioContext, host, port,
            // Lambda callback:
            [&](const std::vector<uint8_t>& packet_data) {
                
                // Decode the packet into frame_data_IMU_copy
                navMsgCallback.decode_NavMsg(packet_data, frame_data_Nav_copy);

                // Check if the frame is valid
                if (frame_data_Nav_copy.timestamp > 0 && 
                    frame_data_Nav_copy.timestamp != this->timestampNav_ ) {

                    this->timestampNav_ = frame_data_Nav_copy.timestamp;

                    // If the vector is full, remove the oldest frame
                    if (frame_buffer_Nav_vec.size() >= VECTOR_SIZE_NAV) {
                        frame_buffer_Nav_vec.erase(frame_buffer_Nav_vec.begin()); // Remove the oldest element
                    }

                    // Push the new frame into the vector
                    frame_buffer_Nav_vec.push_back(frame_data_Nav_copy); // Deep copy into vector
                    
                    // Push the copy into the SPSC queue
                    if (!decodedNav_buffer_.push(frame_buffer_Nav_vec)) {
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame. Buffer Nav might be full." << std::endl;
                    }
                }
                // frame_data_copy goes out of scope here. If it was moved, its destruction is trivial.
                // If it was not pushed (e.g., due to condition not met), it's destructed normally (releasing its copied data).
            }, // End of lambda
            bufferSize);

            // Main loop to run Asio's I/O event processing.
            while (running_.load(std::memory_order_acquire)) {
                try {
                    ioContext.run(); // This will block until work is done or ioContext is stopped.
                                    // If it returns without an exception, it implies all work is done.
                    if (!running_.load(std::memory_order_acquire)) { // Check running_ again if run() returned cleanly
                        break;
                    }
                    // If run() returns and there's still potentially work (or to handle stop signals),
                    // you might need to reset and run again, or break if shutting down.
                    // For a continuous listener, run() might not return unless stopped or an error occurs.
                    // If ioContext.run() returns because it ran out of work, and we are still 'running_',
                    // we should probably restart it if the intent is to keep listening.
                    // However, typically for a server/listener, io_context.run() is expected to block until stop() is called.
                    // If it returns prematurely, ensure io_context is reset if needed before next run() call.
                    // For this pattern, if run() returns, we break, assuming stop() was called elsewhere or an error occurred.
                    break; 
                } catch (const std::exception& e) {
                    // Handle exceptions from ioContext.run()
                    std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
                    std::cerr << "[Pipeline] Listener: Exception in ioContext.run(): " << e.what() << std::endl;
                    if (running_.load(std::memory_order_acquire)) {
                        ioContext.restart(); // Restart Asio io_context to attempt recovery.
                        std::cerr << "[Pipeline] Listener: ioContext restarted." << std::endl;
                    } else {
                        break; // Exit loop if shutting down.
                    }
                }
            }
        }
        catch(const std::exception& e){
            // Handle exceptions from UdpSocket creation or other setup.
            std::lock_guard<std::mutex> lock(consoleMutex); // Protect std::cerr
            std::cerr << "[Pipeline] Listener: Setup exception: " << e.what() << std::endl;
        }

        // Ensure ioContext is stopped when the listener is done or an error occurs.
        if (!ioContext.stopped()) {
            ioContext.stop();
        }
        std::lock_guard<std::mutex> lock(consoleMutex);
        std::cerr << "[Pipeline] Ouster LiDAR(IMU) listener stopped." << std::endl;
    }
    
    // -----------------------------------------------------------------------------

    void Pipeline::runDataAlignment(const std::vector<int>& allowedCores){

        setThreadAffinity(allowedCores);

        while (running_.load(std::memory_order_acquire)) {
            try {
                // If no lidar data is available, wait briefly and retry
                if (decodedPoint_buffer_.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                lidarDecode::LidarDataFrame temp_LidarData;
                if (!decodedPoint_buffer_.pop(temp_LidarData)) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] DataAlignment: Failed to pop from decodedPoint_buffer_." << std::endl;
                    continue;
                }

                // Skip if lidar data is empty
                if (temp_LidarData.timestamp_points.empty()) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] DataAlignment: Empty lidar points for frame ID " << temp_LidarData.frame_id << "." << std::endl;
                    continue;
                }

                // Get min and max lidar timestamps (sorted, so use front and back)
                double lidar_time = temp_LidarData.timestamp_points.front();
                // double max_lidar_time = temp_LidarData.timestamp_points.back();

                // Loop to find an IMU vector that aligns with the current lidar frame
                bool aligned = false;
                while (!aligned){
                    std::vector<decodeNav::DataFrameNavMsg> temp_NavVec;
                    if (!decodedNav_buffer_.pop(temp_NavVec)) {
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] DataAlignment: Failed to pop from decodedNav_buffer_." << std::endl;
                        break;
                    }

                    // Skip if IMU data is empty
                    if (temp_NavVec.empty()) {
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] DataAlignment: Empty NAV data vector for lidar frame ID " 
                                << temp_LidarData.frame_id << "." << std::endl;
                        continue;
                    }

                    // find min/max
                    double min_nav_time = temp_NavVec.front().timestamp;
                    double max_nav_time = temp_NavVec.back().timestamp;

                    // Verify IMU timestamps are valid and ordered
                    if (min_nav_time > max_nav_time) {
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] DataAlignment: Invalid NAV timestamp range: min "
                                << min_nav_time << " > max " << max_nav_time 
                                << " for lidar frame ID " << temp_LidarData.frame_id << "." << std::endl;
                        continue;
                    }

                    // Check if lidar timestamps are within IMU range
                    if (lidar_time >= min_nav_time && lidar_time <= max_nav_time) {
                        // Timestamps are aligned; process the data
                        aligned = true;
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cout << "[Pipeline] DataAlignment: Timestamps aligned for frame ID " 
                                << temp_LidarData.frame_id << ". Lidar range: [" 
                                << lidar_time << "], NAV range: ["
                                << min_nav_time << ", " << min_nav_time << "]" << std::endl;
                        // Add processing logic here (e.g., store aligned data, interpolate IMU, pass to lioOdometry)
                        // todo>>
                        
                        decodeNav::DataFrameNavMsg interpolatedNavMsg;
                        int idx1 = -1, idx2 = -1; // idx1 for closest, idx2 for second closest
                        double min_diff1 = std::numeric_limits<double>::max();
                        double min_diff2 = std::numeric_limits<double>::max();

                        for (int i = 0; i < temp_NavVec.size(); i++) {
                            double current_diff = std::abs(temp_NavVec[i].timestamp - lidar_time);
                            if (current_diff < min_diff1) {
                                min_diff2 = min_diff1;
                                idx2 = idx1;
                                min_diff1 = current_diff;
                                idx1 = i;
                            } else if (current_diff < min_diff2) {
                                min_diff2 = current_diff;
                                idx2 = i;
                            }
                        }

                        // Get the two closest navigation messages
                        const auto& msg1 = temp_NavVec[idx1];
                        const auto& msg2 = temp_NavVec[idx2];

                        // Order messages by timestamp (t1 <= t2)
                        const auto& earlier_msg = (msg1.timestamp <= msg2.timestamp) ? msg1 : msg2;
                        const auto& later_msg = (msg1.timestamp <= msg2.timestamp) ? msg2 : msg1;
                        double t1 = earlier_msg.timestamp;
                        double t2 = later_msg.timestamp;

                        // Verify that lidar_time is within the interpolation range
                        if (lidar_time < t1 || lidar_time > t2) {
                            std::lock_guard<std::mutex> lock(consoleMutex);
                            std::cerr << "[Pipeline] DataAlignment: Lidar time " << lidar_time
                                    << " outside NAV range [" << t1 << ", " << t2 << "] for frame ID "
                                    << temp_LidarData.frame_id << "." << std::endl;
                            continue;
                        }

                        // Handle case where timestamps are equal
                        if (t1 == t2) {
                            interpolatedNavMsg = earlier_msg; // Use either message
                            interpolatedNavMsg.timestamp = lidar_time; // Set to desired timestamp
                        } else {
                            // Compute interpolation factor
                            double alpha = (lidar_time - t1) / (t2 - t1);

                            // Linear interpolation for each field
                            interpolatedNavMsg.timestamp = lidar_time; // Set directly to lidar_time
                            interpolatedNavMsg.latitude = earlier_msg.latitude + alpha * (later_msg.latitude - earlier_msg.latitude);
                            interpolatedNavMsg.longitude = earlier_msg.longitude + alpha * (later_msg.longitude - earlier_msg.longitude);
                            interpolatedNavMsg.altitude = earlier_msg.altitude + alpha * (later_msg.altitude - earlier_msg.altitude);
                            interpolatedNavMsg.roll = earlier_msg.roll + alpha * (later_msg.roll - earlier_msg.roll);
                            interpolatedNavMsg.pitch = earlier_msg.pitch + alpha * (later_msg.pitch - earlier_msg.pitch);
                            interpolatedNavMsg.yaw = earlier_msg.yaw + alpha * (later_msg.yaw - earlier_msg.yaw);
                            interpolatedNavMsg.velU = earlier_msg.velU + alpha * (later_msg.velU - earlier_msg.velU);
                            interpolatedNavMsg.velV = earlier_msg.velV + alpha * (later_msg.velV - earlier_msg.velV);
                            interpolatedNavMsg.velW = earlier_msg.velW + alpha * (later_msg.velW - earlier_msg.velW);
                            interpolatedNavMsg.velP = earlier_msg.velP + alpha * (later_msg.velP - earlier_msg.velP);
                            interpolatedNavMsg.velQ = earlier_msg.velQ + alpha * (later_msg.velQ - earlier_msg.velQ);
                            interpolatedNavMsg.velR = earlier_msg.velR + alpha * (later_msg.velR - earlier_msg.velR);
                            interpolatedNavMsg.accU = earlier_msg.accU + alpha * (later_msg.accU - earlier_msg.accU);
                            interpolatedNavMsg.accV = earlier_msg.accV + alpha * (later_msg.accV - earlier_msg.accV);
                            interpolatedNavMsg.accW = earlier_msg.accW + alpha * (later_msg.accW - earlier_msg.accW);
                            interpolatedNavMsg.accP = earlier_msg.accP + alpha * (later_msg.accP - earlier_msg.accP);
                            interpolatedNavMsg.accQ = earlier_msg.accQ + alpha * (later_msg.accQ - earlier_msg.accQ);
                            interpolatedNavMsg.accR = earlier_msg.accR + alpha * (later_msg.accR - earlier_msg.accR);
                            interpolatedNavMsg.velN = earlier_msg.velN + alpha * (later_msg.velN - earlier_msg.velN);
                            interpolatedNavMsg.velE = earlier_msg.velE + alpha * (later_msg.velE - earlier_msg.velE);
                            interpolatedNavMsg.velD = earlier_msg.velD + alpha * (later_msg.velD - earlier_msg.velD);
                        }
                        if(first_pose && std::isfinite(interpolatedNavMsg.latitude) && std::isfinite(interpolatedNavMsg.longitude) && std::isfinite(interpolatedNavMsg.altitude) 
                        && interpolatedNavMsg.latitude > 0.0, interpolatedNavMsg.longitude > 0.0, interpolatedNavMsg.altitude > 0.0){
                            first_pose = false;
                            oriLat_ = interpolatedNavMsg.latitude;
                            oriLon_ = interpolatedNavMsg.longitude;
                            oriAlt_ = static_cast<double>(interpolatedNavMsg.altitude);
                        } else {continue;}

                        NavDataFrame interpolatedData(temp_LidarData, interpolatedNavMsg,oriLat_,oriLon_,oriAlt_);

                        // Push the copy into the SPSC queue
                        if (!interpolatedNav_buffer_.push(interpolatedData)) {
                            std::lock_guard<std::mutex> lock(consoleMutex);
                            std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame. Buffer interpolatedData might be full." << std::endl;
                        }

                    } else if (lidar_time > max_nav_time){
                        // Lidar is too new or partially overlaps; pop another newer IMU vector >> skip while
                        continue;
                    } else {
                        // Lidar impossible to catch up with the IMU timestamp, need to discard this Lidar frame.
                        // Potential Solution, increase the size buffer frame.
                        std::lock_guard<std::mutex> lock(consoleMutex);
                        std::cerr << "[Pipeline] DataAlignment: Lidar cannot catch up with Nav data, please increase Buffer size" << std::endl;
                        break;
                    }
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(consoleMutex);
                std::cerr << "[Pipeline] DataAlignment: Exception occurred: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }
    }

    // -----------------------------------------------------------------------------

    void Pipeline::updateOccMap(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        // Map parameters (adjust as needed)
        constexpr double voxel_size = 0.1; // Voxel size in meters
        constexpr int max_num_points_in_voxel = 20; // Max points per voxel
        constexpr double min_distance_points = 0.01; // Min distance between points in a voxel
        constexpr int min_num_points = 1; // Min points required in a voxel before adding more
        while (running_.load(std::memory_order_acquire)) {
            try {
                if (interpolatedNav_buffer_.empty()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                NavDataFrame temp_NavData;
                if (!interpolatedNav_buffer_.pop(temp_NavData)) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] updateOccMap: Failed to pop from interpolatedNav_buffer_." << std::endl;
                    continue;
                }

                // Get point cloud as std::vector<lidarDecode::Point3D>
                std::vector<lidarDecode::Points3D> point_cloud = temp_NavData.lidar_data.toPoints3D();

                if (point_cloud.empty()) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] updateOccMap: Empty point cloud for frame ID "
                            << temp_NavData.lidar_data.frame_id << "." << std::endl;
                    continue;
                }
                Eigen::Vector3d local_pose;
                local_pose << temp_NavData.N,temp_NavData.E,temp_NavData.D;
                
                // Add points to the map
                map_.add(point_cloud, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
                
                // raycast points to the map
                map_.raycast(local_pose, point_cloud, voxel_size);

                //remove isolated voxel
                map_.removeIsolatedVoxels();

                // Update voxel lifetimes
                map_.update_and_filter_lifetimes();

                ArrayVector3d currmapArray = map_.pointcloud();
                Eigen::Matrix4d currT = temp_NavData.T;

                vizuFrame_.pointcloud = currmapArray;
                vizuFrame_.T = currT;
                
                // Push the copy into the SPSC queue
                if (!vizu_buffer_.push(vizuFrame_)) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame. Buffer Nav might be full." << std::endl;
                }

                // Log success
                std::lock_guard<std::mutex> lock(consoleMutex);
                std::cout << "[Pipeline] updateOccMap: Added " << point_cloud.size()
                        << " points to map for frame ID " << temp_NavData.lidar_data.frame_id
                        << ". Total map size: " << map_.size() << " points." << std::endl;
            
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(consoleMutex);
                std::cerr << "[Pipeline] updateOccMap: Exception occurred: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
        }
    }

    // -----------------------------------------------------------------------------

    void Pipeline::runVisualizer(const std::vector<int>& allowedCores) {
        setThreadAffinity(allowedCores);

        try {
            if (!vis.CreateVisualizerWindow("3D Point Cloud Visualization", 2560, 1440, 50, 50, true)) { // Added visible=true
                { // Scope for lock
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] Visualizer: Failed to create window." << std::endl;
                }
                return;
            }
            // Access render option after window creation
            vis.GetRenderOption().background_color_ = Eigen::Vector3d(1, 1, 1); // Dark grey background
            vis.GetRenderOption().point_size_ = 1.0; // Slightly larger points

            // Setup camera
            auto& view_control = vis.GetViewControl();
            view_control.SetLookat({0.0, 0.0, 0.0});    // Look at origin
            view_control.SetFront({0.0, 0.0, -1.0}); // Camera slightly tilted down, looking from +Y
            view_control.SetUp({0.0, 1.0, 0.0});     // Z is up
            // view_control.Scale(0.1);               // Zoom level (smaller value = more zoomed in)

            // Initialize point_cloud_ptr_ if it hasn't been already
            if (!point_cloud_ptr_) {
                point_cloud_ptr_ = std::make_shared<open3d::geometry::PointCloud>();
                // Optionally add a placeholder point if you want to see something before data arrives,
                // or leave it empty. updatePtCloudStream will handle empty frames.
                point_cloud_ptr_->points_.push_back(Eigen::Vector3d(0, 0, 0));
                point_cloud_ptr_->colors_.push_back(Eigen::Vector3d(1, 0, 0)); 
            }
            vis.AddGeometry(point_cloud_ptr_);

            if (!vehiclemesh_ptr_) {
                    vehiclemesh_ptr_ = createVehicleMesh(Eigen::Matrix4d::Identity()); // Default vehicle at origin
                }
            vis.AddGeometry(vehiclemesh_ptr_);


            // Add a coordinate frame for reference
            auto coord_frame = open3d::geometry::TriangleMesh::CreateCoordinateFrame(5.0); // Size 1.0 meter
            vis.AddGeometry(coord_frame);

            // Register the animation callback
            // The lambda captures 'this' to call the member function updateVisualizer.
            vis.RegisterAnimationCallback([this](open3d::visualization::Visualizer* callback_vis_ptr) {
                // 'this->' is optional for member function calls but can improve clarity
                return this->updateVisualizer(callback_vis_ptr);
            });
            
            { // Scope for lock
                std::lock_guard<std::mutex> lock(consoleMutex);
                std::cerr << "[Pipeline] Visualizer: Starting Open3D event loop." << std::endl;
            }

            vis.Run(); // This blocks until the window is closed or animation callback returns false

            // Clean up
            vis.DestroyVisualizerWindow();
            { // Scope for lock
                std::lock_guard<std::mutex> lock(consoleMutex);
                std::cerr << "[Pipeline] Visualizer: Open3D event loop finished." << std::endl;
            }

        } catch (const std::exception& e) {
            { // Scope for lock
                std::lock_guard<std::mutex> lock(consoleMutex);
                std::cerr << "[Pipeline] Visualizer: Exception caught: " << e.what() << std::endl;
            }
            // Ensure window is destroyed even if an exception occurs mid-setup (if vis is valid)
            if (vis.GetWindowName() != "") { // A simple check if window might have been created
                vis.DestroyVisualizerWindow();
            }
        }
    }

    // -----------------------------------------------------------------------------

    bool Pipeline::updateVisualizer(open3d::visualization::Visualizer* vis_ptr) {
        constexpr double smoothingFactor = 0.1;
        // Record start time for frame rate limiting (measures processing time of this function)
        // If you want to limit to an absolute FPS regardless of processing time,
        // you'd need a static `last_render_timepoint`.
        auto call_start_time = std::chrono::steady_clock::now();

        VizuDataFrame frame_to_display;
        bool new_frame_available = false;
        auto& view_control = vis_ptr->GetViewControl();

        // Consume all frames currently in the buffer, but only process the latest one for display.
        // This helps the visualizer "catch up" if the producer is faster.
        VizuDataFrame temp_frame;
        while (vizu_buffer_.pop(temp_frame)) {
            frame_to_display = std::move(temp_frame); // Keep moving the latest popped frame
            new_frame_available = true;
        }

        bool geometry_needs_update = false;
        if (new_frame_available) {
            // Process the latest available frame
            // The condition `frame_to_display.numberpoints > 0` is implicitly handled
            // by updatePtCloudStream, which will clear the cloud if numberpoints is 0.
            // std::cerr << "numpoint in decoded: " << frame_to_display.numberpoints << std::endl; //both show same value
            updateStream(point_cloud_ptr_, vehiclemesh_ptr_, frame_to_display);
            geometry_needs_update = true; // Assume geometry changed if we processed a new frame
        }

        if (geometry_needs_update) {
            vis_ptr->UpdateGeometry(point_cloud_ptr_); // Tell Open3D to refresh this geometry
            if(vis_ptr->UpdateGeometry(vehiclemesh_ptr_)){
                Eigen::Vector3d targetLookat;
                targetLookat << frame_to_display.T(0,3),frame_to_display.T(1,3),frame_to_display.T(2,3);
                currentLookat_ = currentLookat_ + smoothingFactor * (targetLookat - currentLookat_);
            }
            view_control.SetLookat(currentLookat_);
            view_control.SetFront({0.0, 0.0, -1.0}); // Camera slightly tilted down, looking from +Y
            view_control.SetUp({0.0, 1.0, 0.0});
            // std::cerr << "numpoint in visualizer: " << point_cloud_ptr_->points_.size() << std::endl; // both show same value
        }

        // Frame rate limiter: ensure this function call (including processing and sleep)
        // takes at least targetFrameDuration.
        auto processing_done_time = std::chrono::steady_clock::now();
        auto processing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(processing_done_time - call_start_time);

        if (processing_duration < targetFrameDuration) {
            std::this_thread::sleep_for(targetFrameDuration - processing_duration);
        }
                
        // Return true to continue animation if the application is still running.
        return running_.load(std::memory_order_acquire);
    }

    // -----------------------------------------------------------------------------

    void Pipeline::updateStream(std::shared_ptr<open3d::geometry::PointCloud>& ptCloud_ptr,
                    std::shared_ptr<open3d::geometry::TriangleMesh>& vehiclemesh_ptr,
                    const dynamicMap::VizuDataFrame& frame) {
        // Validate point cloud pointer
        if (!ptCloud_ptr) {
            return; // Exit if point cloud is null
        }

        if (!vehiclemesh_ptr) {
            return; // Exit if vehiclemesh is null
        }

        // Handle empty frame
        if (frame.pointcloud.size() == 0) {
            if (!ptCloud_ptr->IsEmpty()) {
                ptCloud_ptr->Clear();
            }
            return;
        }

        // Resize point cloud vectors
        ptCloud_ptr->points_.resize(frame.pointcloud.size());
        ptCloud_ptr->colors_.resize(frame.pointcloud.size());

        // Assign points and black colors
        for (size_t i = 0; i < frame.pointcloud.size(); ++i) {
            // Transform coordinates: OusterLidar (x=front, y=right, z=down) to Viz (x=front, y=-right, z=-down)
            ptCloud_ptr->points_[i] = Eigen::Vector3d(
                frame.pointcloud[i].x(),
                -frame.pointcloud[i].y(),
                -frame.pointcloud[i].z()
            );
            // Set color to black (RGB: 0, 0, 0)
            ptCloud_ptr->colors_[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
        }

        // Update vehicle mesh if pointer is valid
        if (vehiclemesh_ptr) {
            vehiclemesh_ptr = createVehicleMesh(frame.T);
        }
    }

    // -----------------------------------------------------------------------------

    std::shared_ptr<open3d::geometry::TriangleMesh> Pipeline::createVehicleMesh(const Eigen::Matrix4d& T) {
        // Validate transformation matrix
        if (T.rows() != 4 || T.cols() != 4 || !T.allFinite()) {
            return std::make_shared<open3d::geometry::TriangleMesh>(); // Return empty mesh for invalid input
        }

        auto vehicle_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

        // Define local vertices in NED coordinates (vehicle shape: 10m long, 10m wide at rear)
        std::vector<Eigen::Vector3d> local_vertices = {
            {10.0, 0.0, 0.0},    // Front tip
            {-10.0, -5.0, 0.0},  // Rear left
            {-10.0, 5.0, 0.0}    // Rear right
        };

        // Define NED-to-visualization transformation matrix
        Eigen::Matrix4d ned_to_viz = Eigen::Matrix4d::Identity();
        ned_to_viz(1,1) = -1.0; // y_viz = -y_NED
        ned_to_viz(2,2) = -1.0; // z_viz = -z_NED

        // Combine transformations: first apply T (NED), then NED-to-viz
        Eigen::Matrix4d combined_T = ned_to_viz * T;

        // Transform vertices to world coordinates using homogeneous coordinates
        std::vector<Eigen::Vector3d> world_vertices;
        world_vertices.reserve(local_vertices.size());
        for (const auto& local_vertex : local_vertices) {
            // Convert to homogeneous coordinates [x, y, z, 1]
            Eigen::Vector4d homogeneous_vertex(local_vertex.x(), local_vertex.y(), local_vertex.z(), 1.0);
            // Apply combined transformation
            Eigen::Vector4d transformed = combined_T * homogeneous_vertex;
            // Extract 3D coordinates
            world_vertices.emplace_back(transformed.head<3>());
        }

        // Assign vertices to mesh
        vehicle_mesh->vertices_ = world_vertices;

        // Define single triangular face (vertices 0, 1, 2)
        vehicle_mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 2));

        // Compute triangle normal in visualization coordinates
        Eigen::Vector3d v0 = world_vertices[1] - world_vertices[0];
        Eigen::Vector3d v1 = world_vertices[2] - world_vertices[0];
        Eigen::Vector3d normal = v0.cross(v1).normalized();
        vehicle_mesh->triangle_normals_.push_back(normal);

        // Assign green color to all vertices
        vehicle_mesh->vertex_colors_.reserve(local_vertices.size());
        vehicle_mesh->vertex_colors_ = {
            {0.0, 1.0, 0.0}, // Green for vertex 0
            {0.0, 1.0, 0.0}, // Green for vertex 1
            {0.0, 1.0, 0.0}  // Green for vertex 2
        };

        return vehicle_mesh;
    }

} // namespace dynamicMap

