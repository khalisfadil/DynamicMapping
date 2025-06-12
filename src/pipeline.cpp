#include <pipeline.hpp>

std::atomic<bool> Pipeline::running_{true};
std::condition_variable Pipeline::globalCV_;
boost::lockfree::spsc_queue<LidarDataFrame, boost::lockfree::capacity<128>> Pipeline::decodedPoint_buffer_;
boost::lockfree::spsc_queue<LidarIMUDataFrame, boost::lockfree::capacity<128>> Pipeline::decodedLidarIMU_buffer_;

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
    UdpSocket listener(ioContext, host, port,
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
    UdpSocket listener(ioContext, host, port,
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

void Pipeline::runOusterLidarIMUListener(boost::asio::io_context& ioContext,
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
    UdpSocket listener(ioContext, host, port,
        // Lambda callback:
        [&](const std::vector<uint8_t>& packet_data) {
            LidarIMUDataFrame frame_data_IMU_copy; // 1. Local DataFrame created.
                                       //    It will hold a deep copy of the lidar data.

            // 2. lidarCallback processes the packet.
            //    Inside decode_packet_single_return, frame_data_copy is assigned
            //    (via DataFrame::operator=) the contents of lidarCallback's completed buffer.
            //    This results in a deep copy into frame_data_copy.
            lidarCallback.decode_packet_LidarIMU(packet_data, frame_data_IMU_copy);

            // 3. Now frame_data_copy is an independent, deep copy of the relevant frame.
            //    We can safely use it and then move it into the queue.
            if (frame_data_IMU_copy.Accelerometer_Read_Time > 0 && frame_data_IMU_copy.Gyroscope_Read_Time > 0 && frame_data_IMU_copy.Accelerometer_Read_Time != this->Accelerometer_Read_Time_ && frame_data_IMU_copy.Gyroscope_Read_Time != this->Gyroscope_Read_Time_) {

                this->Accelerometer_Read_Time_ = frame_data_IMU_copy.Accelerometer_Read_Time;
                this->Gyroscope_Read_Time_ = frame_data_IMU_copy.Gyroscope_Read_Time;
                
                // 4. Move frame_data_copy into the SPSC queue.
                //    This transfers ownership of frame_data_copy's internal resources (vector data)
                //    to the element constructed in the queue, avoiding another full copy.
                //    frame_data_copy is left in a valid but unspecified (likely empty) state.
                if (!decodedLidarIMU_buffer_.push(std::move(frame_data_IMU_copy))) {
                    std::lock_guard<std::mutex> lock(consoleMutex);
                    std::cerr << "[Pipeline] Listener: SPSC buffer push failed for frame " 
                              << this->frame_id_ // Use this->frame_id_ as frame_data_copy might be moved-from
                              << ". Buffer Lidar(IMU) might be full." << std::endl;
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
        vis.GetRenderOption().background_color_ = Eigen::Vector3d(0.05, 0.05, 0.05); // Dark grey background
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
    // Record start time for frame rate limiting (measures processing time of this function)
    // If you want to limit to an absolute FPS regardless of processing time,
    // you'd need a static `last_render_timepoint`.
    auto call_start_time = std::chrono::steady_clock::now();

    LidarDataFrame frame_to_display;
    bool new_frame_available = false;
    auto& view_control = vis_ptr->GetViewControl();

    // Consume all frames currently in the buffer, but only process the latest one for display.
    // This helps the visualizer "catch up" if the producer is faster.
    LidarDataFrame temp_frame;
    while (decodedPoint_buffer_.pop(temp_frame)) {
        frame_to_display = std::move(temp_frame); // Keep moving the latest popped frame
        new_frame_available = true;
    }

    bool geometry_needs_update = false;
    if (new_frame_available) {
        // Process the latest available frame
        // The condition `frame_to_display.numberpoints > 0` is implicitly handled
        // by updatePtCloudStream, which will clear the cloud if numberpoints is 0.
        // std::cerr << "numpoint in decoded: " << frame_to_display.numberpoints << std::endl; //both show same value
        updatePtCloudStream(point_cloud_ptr_, frame_to_display);
        geometry_needs_update = true; // Assume geometry changed if we processed a new frame
    }

    if (geometry_needs_update) {
        vis_ptr->UpdateGeometry(point_cloud_ptr_); // Tell Open3D to refresh this geometry
        view_control.Scale(0.1);
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

void Pipeline::updatePtCloudStream(std::shared_ptr<open3d::geometry::PointCloud>& ptCloud_ptr, const LidarDataFrame& frame) {
    if (!ptCloud_ptr) return; // Should not happen if initialized, but good check

    if (frame.numberpoints <= 0) {
        // If the new frame is empty, clear the displayed point cloud
        // to avoid showing stale data.
        if (!ptCloud_ptr->IsEmpty()) { // Only clear if it's not already empty
            ptCloud_ptr->Clear();
        }
        return;
    }

    // Resize point cloud vectors to match the new frame's data.
    // This is efficient as it reuses memory if capacity is sufficient.
    ptCloud_ptr->points_.resize(frame.numberpoints);
    ptCloud_ptr->colors_.resize(frame.numberpoints);

    // Pre-calculate for color mapping (already a constexpr, which is good)
    constexpr double log2_255_div_100 = 1.35049869499; // log2(255.0/100.0)

    for (size_t i = 0; i < frame.numberpoints; ++i) {
        // Assign transformed coordinates for visualization
        // OusterLidarCallback output: x=front, y=right, z=down
        // Visualization coordinates: x=front, y=-right (left), z=-down (up)
        ptCloud_ptr->points_[i] = Eigen::Vector3d(frame.x[i], -frame.y[i], -frame.z[i]);

        // Map reflectivity to parameter t for color interpolation
        double reflectivity_val = static_cast<double>(frame.reflectivity[i]); // Ensure double for calculations
        double t; // Normalized parameter [0, 1] for color interpolation

        if (reflectivity_val <= 100.0) {
            // Linear mapping for lambertian targets (reflectivity 0 to 100) -> t [0, 0.5]
            t = reflectivity_val / 200.0; 
        } else {
            // Logarithmic-based mapping for retroreflective targets (reflectivity 101 to 255) -> t (0.5, 1.0]
            // Ensure reflectivity_val / 100.0 is > 0 for log2, which it will be if reflectivity_val > 100
            double log_val = std::log2(reflectivity_val / 100.0);
            double s_param = 1.0 + 863.0 * (log_val / log2_255_div_100);
            t = 0.5 + 0.5 * (s_param - 1.0) / 863.0;
        }
        t = std::max(0.0, std::min(1.0, t)); // Clamp t to [0, 1] to be safe

        // Interpolate color based on t
        if (t < 0.5) { // Royal blue (t=0) to Neon green (t=0.5)
            double u = t * 2.0; // u normalized from 0 to 1 for this segment
            ptCloud_ptr->colors_[i] = Eigen::Vector3d(
                0.12 * (1.0 - u),             // R: 0.12 -> 0.0
                0.29 + (1.0 - 0.29) * u,      // G: 0.29 -> 1.0
                0.69 * (1.0 - u)              // B: 0.69 -> 0.0
            );
        } else { // Neon green (t=0.5) to White (t=1.0)
            double u = (t - 0.5) * 2.0; // u normalized from 0 to 1 for this segment
            ptCloud_ptr->colors_[i] = Eigen::Vector3d(
                u,                            // R: 0.0 -> 1.0
                1.0,                          // G: 1.0 (constant)
                u                             // B: 0.0 -> 1.0
            );
        }
    }
    

    // VoxelDownSample:
    // The original call ptCloud_ptr->VoxelDownSample(1); was problematic:
    // 1. It returns a new point cloud, does not modify in place.
    // 2. A voxel size of 1.0 (meter) is very large and would decimate the cloud heavily.
    // If downsampling is truly needed for performance or visual clarity, apply it here correctly.
    // Example:
    // double desired_voxel_size = 0.05; // 5cm, adjust as needed
    // auto downsampled_cloud = ptCloud_ptr->VoxelDownSample(desired_voxel_size);
    // *ptCloud_ptr = *downsampled_cloud; // This replaces the current cloud with the downsampled one.
                                        // This involves a copy.
    // For now, it's removed as its original intent/usage was unclear and potentially incorrect.
    // If performance is an issue with many points, consider uncommenting and adjusting the above.
}

