// MIT License

// Copyright (c) 2024 Muhammad Khalis bin Mohd Fadil

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#pragma once

#include "clusterExtractor.hpp"

constexpr int maxPointsPerVoxel_ = 20;

class OccupancyMap {
    public:
        //##############################################################################
        // Struct to store point data within each voxel
        struct PointData {
            Eigen::Vector3f position;  // Position of the point
            float reflectivity;
            float intensity;
            float NIR;
        };
        //##############################################################################
        // Struct to store RemovalReason
        enum class RemovalReason {
            None,
            Raycasting,
            MaxRangeExceeded,
            Dynamic
        };
        //##############################################################################
        // Struct to store voxel data, including a fixed-size vector of points and aggregate metrics
        struct VoxelData {
            std::vector<PointData> points;  // Changed from std::deque to std::vector
            Eigen::Vector3f centerPosition;
            float totalReflectivity = 0.0f;
            float totalIntensity = 0.0f;
            float totalNIR = 0.0f;
            float avgReflectivity = 0.0f;
            float avgIntensity = 0.0f;
            float avgNIR = 0.0f;
            uint32_t lastSeenFrame = 0;       // Last frame this voxel was updated
            mutable bool isDynamic = false;       // Flag indicating dynamic status
            mutable RemovalReason removalReason = RemovalReason::None;  // Reason for voxel removal

            // Default constructor initializes all members
            VoxelData() 
                : centerPosition(Eigen::Vector3f::Zero()),  // Initialize to (0, 0, 0)
                totalReflectivity(0.0f),
                totalIntensity(0.0f),
                totalNIR(0.0f),
                avgReflectivity(0.0f),
                avgIntensity(0.0f),
                avgNIR(0.0f),
                lastSeenFrame(0),
                isDynamic(false),
                removalReason(RemovalReason::None) {
                points.reserve(maxPointsPerVoxel_);  // Reserve capacity for MAX_POINTS_PER_VOXEL
            }
        };
        //##############################################################################
        // Constructor with one-time parameters
        OccupancyMap(float mapRes,
                        float reachingDistance,
                        Eigen::Vector3f mapCenter);
        //##############################################################################
        // Main Pipeline runOccupancyMapPipeline
        void runOccupancyMapPipeline(const std::vector<Eigen::Vector3f>& pointCloud,
                                        const std::vector<float>& reflectivity,
                                        const std::vector<float>& intensity,
                                        const std::vector<float>& NIR,
                                        const Eigen::Vector3f& newPosition,
                                        uint32_t newFrame);
        //##############################################################################
        // getDynamicVoxels
        std::vector<VoxelData> getDynamicVoxels() const;
        //##############################################################################
        // getStaticVoxels
        std::vector<VoxelData> getStaticVoxels() const;
        //##############################################################################
        // getAllVoxelCenters
        std::vector<Eigen::Vector3f> getVoxelCenters(const std::vector<VoxelData>& voxels);
        //##############################################################################
        // computeVoxelColors
        std::tuple<std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>> computeVoxelColors(const std::vector<VoxelData>& voxels);
        //##############################################################################
        // Assign voxel colour red
        std::vector<Eigen::Vector3i> assignVoxelColorsRed(const std::vector<VoxelData>& voxels);
    private:
        //##############################################################################
        // Persistent member variables (one-time defined parameters)
        float mapRes_;
        float reachingDistance_;
        Eigen::Vector3f mapCenter_;
        Eigen::Vector3f vehiclePosition_;
        uint32_t currentFrame_;
        //##############################################################################
        // Custom hash function for Eigen::Vector3i
        struct Vector3iHash {
            std::size_t operator()(const Eigen::Vector3i& vec) const {
                return std::hash<int>()(vec.x()) ^ (std::hash<int>()(vec.y()) << 1) ^ (std::hash<int>()(vec.z()) << 2);}};
        //##############################################################################
        // Custom equality function for Eigen::Vector3i
        struct Vector3iEqual {
            bool operator()(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs) const {
                return lhs == rhs;}};
        //##############################################################################
        // Persistent member occupancyMap_
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> occupancyMap_;
        //##############################################################################
        // Persistent data structure to track inserted or modified voxels in each frame
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> insertedVoxels_;
        //##############################################################################
        // updateVehiclePosition
        void updateVehiclePosition(const Eigen::Vector3f& newPosition);
        //##############################################################################
        // updateCurrentFrame
        void updateCurrentFrame(uint32_t newFrame);
        //##############################################################################
        // Convert position to voxel grid index
        Eigen::Vector3i posToGridIndex(const Eigen::Vector3f& pos) const;
        //##############################################################################
        // Convert grid index back to world position (center of voxel)
        Eigen::Vector3f gridToWorld(const Eigen::Vector3i& gridIndex) const;
        //##############################################################################
        // Calculate average values for a voxel
        void updateVoxelAverages(VoxelData& voxel) const;
        //##############################################################################
        // Insert point Cloud into occupancy Map
        void insertPointCloud(const std::vector<Eigen::Vector3f>& pointCloud,
                                    const std::vector<float>& reflectivity,
                                    const std::vector<float>& intensity,
                                    const std::vector<float>& NIR);
        //##############################################################################
        // Perform raycast to remove voxel
        std::vector<Eigen::Vector3i> performRaycast(const Eigen::Vector3f& start, const Eigen::Vector3f& end);
        //##############################################################################
        // Perform markVoxelsForClearing
        void markVoxelsForClearing();
        //##############################################################################
        // Perform markVoxelsForClearing
        void markDynamicVoxels(const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud);
        //##############################################################################
        // Perform removeFlaggedVoxels
        void removeFlaggedVoxels();
        //##############################################################################
        // Calculate occupancy-based grayscale color
        Eigen::Vector3i calculateOccupancyColor(const VoxelData& voxel);
        //##############################################################################
        // Calculate reflectivity color
        Eigen::Vector3i calculateReflectivityColor(float avgReflectivity);
        //##############################################################################
        // Calculate intensity-based color
        Eigen::Vector3i calculateIntensityColor(float avgIntensity);
        //##############################################################################
        // Calculate NIR-based color
        Eigen::Vector3i calculateNIRColor(float avgNIR);
};