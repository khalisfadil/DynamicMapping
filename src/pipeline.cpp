#include <pipeline.hpp>

namespace dynamicMap {

    std::atomic<bool> Pipeline::running_{true};
    std::condition_variable Pipeline::globalCV_;
    boost::lockfree::spsc_queue<lidarDecode::LidarDataFrame, boost::lockfree::capacity<128>> Pipeline::decodedPoint_buffer_;
    boost::lockfree::spsc_queue<lidarDecode::LidarIMUDataFrame, boost::lockfree::capacity<128>> Pipeline::decodedLidarIMU_buffer_;

    // -----------------------------------------------------------------------------

    Pipeline::Pipeline(const std::string& json_path) : lidarCallback(json_path) {}
    Pipeline::Pipeline(const nlohmann::json& json_data) : lidarCallback(json_data) {}
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
                            std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame. Buffer Nav might be full." << std::endl;
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

        try {

            while (running_.load(std::memory_order_acquire)) {

            }

        } catch (const std::exception& e) {

        }
    }
} // namespace dynamicMap

