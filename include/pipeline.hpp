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
#include <OusterLidarCallback_c.hpp>
#include <LidarDataframe.hpp>
#include <navdataframe.hpp>
#include <vizudataframe.hpp>
#include <LidarIMUDataFrame.hpp>
#include <DataFrame_NavMsg.hpp>
#include <callback_navMsg.hpp>
#include <map.hpp>
#include <vizuutils.hpp>


namespace dynamicMap {
    class Pipeline {
        public:

            static std::atomic<bool> running_;
            static std::condition_variable globalCV_;
            static boost::lockfree::spsc_queue<lidarDecode::LidarDataFrame, boost::lockfree::capacity<128>> decodedPoint_buffer_;
            static boost::lockfree::spsc_queue<lidarDecode::LidarIMUDataFrame, boost::lockfree::capacity<128>> decodedLidarIMU_buffer_;
            static boost::lockfree::spsc_queue<std::vector<decodeNav::DataFrameNavMsg>, boost::lockfree::capacity<128>> decodedNav_buffer_;
            static boost::lockfree::spsc_queue<dynamicMap::NavDataFrame, boost::lockfree::capacity<128>> interpolatedNav_buffer_;
            static boost::lockfree::spsc_queue<VizuDataFrame, boost::lockfree::capacity<128>> vizu_buffer_;

            Pipeline(const std::string& json_path); // Constructor with JSON file path
            Pipeline(const nlohmann::json& json_data); // Constructor with JSON data
            // ~Pipeline();
            static void signalHandler(int signal);
            void setThreadAffinity(const std::vector<int>& coreIDs);
            void runOusterLidarListenerSingleReturn(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores);
            void runOusterLidarListenerLegacy(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores);  
            void runOusterLidarIMUListener(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores); 
            void runVisualizer(const std::vector<int>& allowedCores);

            //application
            void runNavMsgListener(boost::asio::io_context& ioContext, const std::string& host, uint16_t port, uint32_t bufferSize, const std::vector<int>& allowedCores);
            void runDataAlignment(const std::vector<int>& allowedCores);
            void updateOccMap(const std::vector<int>& allowedCores);

            open3d::visualization::Visualizer vis;

        private:

            std::mutex consoleMutex;
            OusterLidarCallback lidarCallback;
            uint16_t frame_id_= 0;

            lidarDecode::LidarDataFrame frame_data_copy_;

            decodeNav::DataFrameNavMsg frame_data_Nav_copy; // Local DataFrame created
            decodeNav::NavMsgCallback navMsgCallback;
            double timestampNav_ = 0.0;
            std::vector<decodeNav::DataFrameNavMsg> frame_buffer_Nav_vec;
            const size_t VECTOR_SIZE_NAV = 15;
            bool first_pose = true;
            double oriLat_ = 0.0;
            double oriLon_ = 0.0;
            double oriAlt_ = 0.0;

            Map map_{10};

            VizuDataFrame vizuFrame_;
            Eigen::Vector3d currentLookat_ = {0, 0, 0};
            std::chrono::milliseconds targetFrameDuration{30};

            std::shared_ptr<open3d::geometry::PointCloud> point_cloud_ptr_;
            std::shared_ptr<open3d::geometry::TriangleMesh> vehiclemesh_ptr_;

            bool updateVisualizer(open3d::visualization::Visualizer* vis);
            void updateStream(std::shared_ptr<open3d::geometry::PointCloud>& ptCloud_ptr,
                    std::shared_ptr<open3d::geometry::TriangleMesh>& vehiclemesh_ptr,
                    const dynamicMap::VizuDataFrame& frame);

    };
} // namespace dynamicMap