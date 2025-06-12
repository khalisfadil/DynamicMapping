#pragma once

#include <boost/asio.hpp>
#include <boost/lockfree/spsc_queue.hpp>
#include <memory>
#include <thread>
#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>
#include <open3d/Open3D.h>

#include <UdpSocket.hpp>
#include <OusterLidarCallback.hpp>
#include <LidarDataframe.hpp>
#include <LidarIMUDataFrame.hpp>

class Pipeline {
    public:

        static std::atomic<bool> running_;
        static std::condition_variable globalCV_;
        static boost::lockfree::spsc_queue<LidarDataFrame, boost::lockfree::capacity<128>> decodedPoint_buffer_;
        static boost::lockfree::spsc_queue<LidarIMUDataFrame, boost::lockfree::capacity<128>> decodedLidarIMU_buffer_;

        Pipeline(const std::string& json_path); // Constructor with JSON file path
        Pipeline(const nlohmann::json& json_data); // Constructor with JSON data
        // ~Pipeline();
        static void signalHandler(int signal);
        void setThreadAffinity(const std::vector<int>& coreIDs);
        void runOusterLidarListenerSingleReturn(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores);
        void runOusterLidarListenerLegacy(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores);  
        void runOusterLidarIMUListener(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores); 
        void runVisualizer(const std::vector<int>& allowedCores);

        open3d::visualization::Visualizer vis;

    private:

        std::mutex consoleMutex;
        OusterLidarCallback lidarCallback;
        uint16_t frame_id_= 0;
        uint64_t Accelerometer_Read_Time_ = 0.0;
        uint64_t Gyroscope_Read_Time_ = 0.0;

        LidarDataFrame frame_data_copy_;

        std::shared_ptr<open3d::geometry::PointCloud> point_cloud_ptr_;

        bool updateVisualizer(open3d::visualization::Visualizer* vis);
        void updatePtCloudStream(std::shared_ptr<open3d::geometry::PointCloud>& ptCloud_ptr, const LidarDataFrame& frame);
        
        std::chrono::milliseconds targetFrameDuration{100}; //30 FPS

};