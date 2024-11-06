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

#include <cstddef>
#include <functional>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include <Eigen/Eigen>

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <tbb/tbb.h>          // Ensure tbb/tbb.h is included for parallel constructs
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/concurrent_vector.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>


// typedef unsigned long size_t;  // Define size_t for compatibility if needed

// Custom hash function for Eigen::Vector3i
struct Vector3iHash {
    std::size_t operator()(const Eigen::Vector3i& vec) const {
        return std::hash<int>()(vec.x()) ^ (std::hash<int>()(vec.y()) << 1) ^ (std::hash<int>()(vec.z()) << 2);
    }
};

// Custom equality function for Eigen::Vector3i
struct Vector3iEqual {
    bool operator()(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs) const {
        return lhs == rhs;
    }
};

// Struct to store point data within each voxel
struct PointData {
    Eigen::Vector3d position;
    float reflectivity;
    float intensity;
    float NIR;
};

enum class RemovalReason {
    None,
    Raycasting,
    MaxRangeExceeded
};

// Struct to store voxel data, including a fixed-size queue of points and aggregate metrics
struct VoxelData {
    std::deque<PointData> points;
    Eigen::Vector3d centerPosition;
    float totalReflectivity = 0.0f;
    float totalIntensity = 0.0f;
    float totalNIR = 0.0f;
    float avgReflectivity = 0.0f;
    float avgIntensity = 0.0f;
    float avgNIR = 0.0f;

    int stabilityScore = 0;       // Number of frames voxel has remained stable
    int lastSeenFrame = -1;       // Last frame this voxel was updated
    bool isDynamic = false;       // Flag indicating dynamic status
    RemovalReason removalReason = RemovalReason::None;  // Reason for voxel removal
};

class OccupancyMap {
public:
    OccupancyMap(double resolution, const Eigen::Vector3d& map_center, double max_distance);

    void updateOccupancy(const std::vector<Eigen::Vector3d>& pointCloud,
                         const Eigen::Vector3d& vehiclePosition,
                         const std::vector<float>& reflectivity,
                         const std::vector<float>& intensity,
                         const std::vector<float>& NIR,
                         int currentFrame,
                         int stabilityThreshold,
                         int decayFactor,
                         int neighborRange);

    std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i>> getOccupiedCellsWithColors() const;
    
private:
    double mapRes_;
    Eigen::Vector3d mapCenter_;
    double maxDistance_;

    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> occupancyMap_;
    static const int MAX_POINTS_PER_VOXEL = 255;

    bool isWithinDistance(const Eigen::Vector3d& pos) const;
    Eigen::Vector3i posToGridIndex(const Eigen::Vector3d& pos) const;
    Eigen::Vector3d gridToWorld(const Eigen::Vector3i& gridIndex) const;
    void updateVoxelAverages(VoxelData& voxel) const;
    std::vector<Eigen::Vector3i> performRaycast(const Eigen::Vector3d& start, const Eigen::Vector3d& end);

    bool checkNeighborConsistency(const Eigen::Vector3i& gridIndex, int stabilityThreshold, int neighborRange) const;
    double calculateMaxRadius(const std::vector<Eigen::Vector3d>& pointCloud, const Eigen::Vector3d& vehiclePosition) const;
    int calculateReflectivityColor(float reflectivity) const;
};



