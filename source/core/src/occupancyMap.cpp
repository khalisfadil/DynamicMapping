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
#include "occupancyMap.hpp"
//##############################################################################
// Constructor
OccupancyMap::OccupancyMap(float mapRes, float reachingDistance, Eigen::Vector3f mapCenter)
    : mapRes_(mapRes), reachingDistance_(reachingDistance), mapCenter_(mapCenter)  {
    // constructor body
}
//##############################################################################
// Main Pipeline
void OccupancyMap::runOccupancyMapPipeline(const std::vector<Eigen::Vector3f>& pointCloud,
                                           const std::vector<float>& reflectivity,
                                           const std::vector<float>& intensity,
                                           const std::vector<float>& NIR,
                                           const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud,
                                           const Eigen::Vector3f& newPosition,
                                           uint32_t newFrame) {
    // Step 0: Update persistent state
    updateVehiclePosition(newPosition);
    updateCurrentFrame(newFrame);

    // Step 1: Insert the point cloud
    insertPointCloud(pointCloud, reflectivity, intensity, NIR);

    // Step 2: Mark voxels for clearing
    markVoxelsForClearing();

    // Step 3: Remove flagged voxels
    removeFlaggedVoxels();

    // Step 4: Mark dynamic voxels only if dynamicCloud is not empty
    if (!dynamicCloud.empty()) {
        markDynamicVoxels(dynamicCloud);
    }
}
//##############################################################################
// Function to update the vehicle position
void OccupancyMap::updateVehiclePosition(const Eigen::Vector3f& newPosition) {
    vehiclePosition_ = newPosition;
}
//##############################################################################
// Function to update the current frame
void OccupancyMap::updateCurrentFrame(uint32_t newFrame) {
    currentFrame_ = newFrame;
}
//##############################################################################
// Convert position to voxel grid index
Eigen::Vector3i OccupancyMap::posToGridIndex(const Eigen::Vector3f& pos) const {
    Eigen::Array3f scaledPos = (pos - mapCenter_).array() * (1.0 / mapRes_);
    return Eigen::Vector3i(std::floor(scaledPos.x()), std::floor(scaledPos.y()), std::floor(scaledPos.z()));
}
//##############################################################################
// Convert grid index back to world position (center of voxel)
Eigen::Vector3f OccupancyMap::gridToWorld(const Eigen::Vector3i& gridIndex) const {
    static const Eigen::Vector3f halfRes(mapRes_ / 2, mapRes_ / 2, mapRes_ / 2);
    return mapCenter_ + gridIndex.cast<float>() * mapRes_ + halfRes;
}
//##############################################################################
// Calculate average values for a voxel
void OccupancyMap::updateVoxelAverages(VoxelData& voxel) const {
    const uint64_t pointCount = voxel.points.size();
    if (pointCount == 0) return;

    double invPointCount = 1.0 / pointCount;
    voxel.avgReflectivity = voxel.totalReflectivity * invPointCount;
    voxel.avgIntensity = voxel.totalIntensity * invPointCount;
    voxel.avgNIR = voxel.totalNIR * invPointCount;
}
//##############################################################################
// insert the point cloud into occupancy map
void OccupancyMap::insertPointCloud(const std::vector<Eigen::Vector3f>& pointCloud,
                                   const std::vector<float>& reflectivity,
                                   const std::vector<float>& intensity,
                                   const std::vector<float>& NIR) {
    insertedVoxels_.clear();

    // Use TBB parallel_reduce to handle the point cloud in parallel
    auto localMap = tbb::parallel_reduce(
        tbb::blocked_range<uint64_t>(0, pointCloud.size(), 1024),
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>(),
        [&](const tbb::blocked_range<uint64_t>& range,
            tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> localMap) {

            for (uint64_t i = range.begin(); i < range.end(); ++i) {
                const Eigen::Vector3f& point = pointCloud[i];
                Eigen::Vector3i gridIndex = posToGridIndex(point);
                auto& voxel = localMap[gridIndex];

                if (voxel.points.empty()) {
                    voxel.centerPosition = gridToWorld(gridIndex);
                }
                voxel.lastSeenFrame = currentFrame_;

                // Add the point if voxel has space
                if (voxel.points.size() < maxPointsPerVoxel_) {
                    voxel.points.push_back({point, reflectivity[i], intensity[i], NIR[i]});
                    voxel.totalReflectivity += reflectivity[i];
                    voxel.totalIntensity += intensity[i];
                    voxel.totalNIR += NIR[i];
                }
            }
            return localMap;
        },
        // Combine local maps into one by merging each entry
        [](auto a, auto b) {
            for (auto& [gridIndex, localVoxel] : b) {
                auto& voxel = a[gridIndex];
                if (voxel.points.empty()) {
                    voxel = std::move(localVoxel);
                } else {
                    // **Change 1: Calculate pointsToAdd only once in the combine step**
                    uint64_t availableSpace = maxPointsPerVoxel_ - voxel.points.size();
                    uint64_t pointsToAdd = std::min(availableSpace, localVoxel.points.size());
                    // Insert points from localVoxel, limiting to pointsToAdd
                    voxel.points.insert(voxel.points.end(), localVoxel.points.begin(), 
                                        localVoxel.points.begin() + pointsToAdd);
                    // Update aggregate values only for the added points
                    for (uint64_t i = 0; i < pointsToAdd; ++i) {
                        voxel.totalReflectivity += localVoxel.points[i].reflectivity;
                        voxel.totalIntensity += localVoxel.points[i].intensity;
                        voxel.totalNIR += localVoxel.points[i].NIR;
                    }
                }
            }
            return a;
        }
    );

    // Final merge from localMap into the main occupancy map and track inserted voxels
    for (auto& [gridIndex, localVoxel] : localMap) {
        auto& voxel = occupancyMap_[gridIndex];

        if (voxel.points.empty()) {
            voxel = std::move(localVoxel);
        } else {
            // **Change 2: Calculate pointsToAdd only once in the final merge**
            uint64_t availableSpace = maxPointsPerVoxel_ - voxel.points.size();
            uint64_t pointsToAdd = std::min(availableSpace, localVoxel.points.size());

            voxel.lastSeenFrame = currentFrame_;
            // Insert points from localVoxel, limiting to pointsToAdd
            voxel.points.insert(voxel.points.end(), localVoxel.points.begin(), 
                                localVoxel.points.begin() + pointsToAdd);
            // Update aggregate values only for the added points
            for (uint64_t i = 0; i < pointsToAdd; ++i) {
                voxel.totalReflectivity += localVoxel.points[i].reflectivity;
                voxel.totalIntensity += localVoxel.points[i].intensity;
                voxel.totalNIR += localVoxel.points[i].NIR;
            }
        }

        // Update average values after merging
        updateVoxelAverages(voxel);

        // Track this voxel’s gridIndex in insertedVoxels_ for the current frame
        insertedVoxels_[gridIndex] = voxel;
    }
}
//##############################################################################
// Perform raycast to remove voxel
std::vector<Eigen::Vector3i> OccupancyMap::performRaycast(const Eigen::Vector3f& start, const Eigen::Vector3f& end) {
    // Define a small threshold for "close" points
    const float closeThreshold = 1e-3;

    // If start and end points are too close, return the starting voxel only
    if ((end - start).squaredNorm() < closeThreshold * closeThreshold) {
        return { posToGridIndex(start) };
    }
    // Precompute values for the loop to avoid recalculating in each iteration
    Eigen::Vector3f direction = (end - start).normalized();
    float distance = (end - start).norm();
    // Initialize voxel collection
    tsl::robin_set<Eigen::Vector3i, Vector3iHash, Vector3iEqual> uniqueVoxels;
    Eigen::Vector3i lastVoxelIndex = posToGridIndex(start);
    uniqueVoxels.insert(lastVoxelIndex);
    // Use a single position variable and avoid unnecessary recalculations in each step
    Eigen::Vector3f currentPos = start;
    for (float step = mapRes_; step < distance; step += mapRes_) {
        currentPos += direction * mapRes_;
        Eigen::Vector3i voxelIndex = posToGridIndex(currentPos);

        // Insert only if the voxel index changes
        if (voxelIndex != lastVoxelIndex) {
            uniqueVoxels.insert(voxelIndex);
            lastVoxelIndex = voxelIndex;
        }
    }
    // Return unique voxels as a vector
    return std::vector<Eigen::Vector3i>(uniqueVoxels.begin(), uniqueVoxels.end());
}
//##############################################################################
// Perform marking Voxels For Clearing
void OccupancyMap::markVoxelsForClearing() {
    // First parallel task: Mark voxels beyond the maximum reaching distance
    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), [&](auto& mapEntry) {
        auto& [gridIndex, targetVoxel] = mapEntry;

        // Check if the voxel is beyond the maximum reaching distance
        if ((targetVoxel.centerPosition - vehiclePosition_).norm() > reachingDistance_) {
            targetVoxel.removalReason = RemovalReason::MaxRangeExceeded;
        }
    });

    // Second parallel task: Perform raycasting for each voxel in insertedVoxels_
    tbb::parallel_for_each(insertedVoxels_.begin(), insertedVoxels_.end(), [&](const auto& insertedEntry) {
        const auto& [gridIndex, voxel] = insertedEntry;

        // Perform raycasting from vehiclePosition_ to voxel center
        for (const auto& rayVoxel : performRaycast(vehiclePosition_, voxel.centerPosition)) {
            auto& targetVoxel = occupancyMap_[rayVoxel];

            // Flag the voxel for removal due to raycasting
            targetVoxel.removalReason = RemovalReason::Raycasting;
        }
    });