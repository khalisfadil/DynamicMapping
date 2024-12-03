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

    // Create a thread-local map for each parallel thread
    std::vector<tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>> threadLocalMaps;

    #pragma omp parallel
    {
        // Each thread gets its own local map
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> localMap;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < pointCloud.size(); ++i) {
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

        // Append the local map to the thread-local map vector
        #pragma omp critical
        {
            threadLocalMaps.push_back(std::move(localMap));
        }
    }

    // Merge thread-local maps into the global occupancy map
    for (auto& localMap : threadLocalMaps) {
        for (auto& [gridIndex, localVoxel] : localMap) {
            auto& voxel = occupancyMap_[gridIndex];

            if (voxel.points.empty()) {
                voxel = std::move(localVoxel);
            } else {
                uint64_t availableSpace = maxPointsPerVoxel_ - voxel.points.size();
                uint64_t pointsToAdd = std::min(availableSpace, localVoxel.points.size());

                voxel.lastSeenFrame = currentFrame_;
                voxel.points.insert(voxel.points.end(), localVoxel.points.begin(), 
                                    localVoxel.points.begin() + pointsToAdd);

                for (uint64_t i = 0; i < pointsToAdd; ++i) {
                    voxel.totalReflectivity += localVoxel.points[i].reflectivity;
                    voxel.totalIntensity += localVoxel.points[i].intensity;
                    voxel.totalNIR += localVoxel.points[i].NIR;
                }
            }

            updateVoxelAverages(voxel);
            insertedVoxels_[gridIndex] = voxel;
        }
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

    // Calculate direction and distance
    Eigen::Vector3f direction = (end - start).normalized();
    float distance = (end - start).norm();

    // Safeguard against invalid distance
    if (distance <= 0) {
        return { posToGridIndex(start) };
    }

    // Initialize voxel collection
    tsl::robin_set<Eigen::Vector3i, Vector3iHash, Vector3iEqual> uniqueVoxels;
    Eigen::Vector3i lastVoxelIndex = posToGridIndex(start);
    uniqueVoxels.insert(lastVoxelIndex);

    // Use a single position variable and avoid unnecessary recalculations in each step
    Eigen::Vector3f currentPos = start;
    for (float step = mapRes_; step < distance; step += mapRes_) {
        currentPos = start + direction * step;
        Eigen::Vector3i voxelIndex = posToGridIndex(currentPos);

        // Insert only if the voxel index changes
        if (voxelIndex != lastVoxelIndex) {
            uniqueVoxels.insert(voxelIndex);
            lastVoxelIndex = voxelIndex;
        }
    }

    // Add the final voxel index explicitly
    Eigen::Vector3i finalVoxelIndex = posToGridIndex(end);
    if (finalVoxelIndex != lastVoxelIndex) {
        uniqueVoxels.insert(finalVoxelIndex);
    }

    // Return unique voxels as a vector
    return std::vector<Eigen::Vector3i>(uniqueVoxels.begin(), uniqueVoxels.end());
}
//##############################################################################
// Perform raycast to remove voxel
void OccupancyMap::markVoxelsForClearing() {
    // Step 1: Collect voxels to flag due to exceeding maximum range
    std::vector<Eigen::Vector3i> farVoxels;

    #pragma omp parallel
    {
        std::vector<Eigen::Vector3i> localFarVoxels;

        // Use an index-based loop instead of iterator for compatibility
        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < occupancyMap_.size(); ++idx) {
            auto it = std::next(occupancyMap_.begin(), idx);
            const auto& targetVoxel = it->second;

            // Check if the voxel exceeds the maximum range
            if ((targetVoxel.centerPosition - vehiclePosition_).norm() > reachingDistance_) {
                localFarVoxels.push_back(it->first);  // Directly use the key
            }
        }

        // Combine thread-local results safely
        #pragma omp critical
        farVoxels.insert(farVoxels.end(), localFarVoxels.begin(), localFarVoxels.end());
    }

    // Apply the removal flag sequentially
    for (const auto& gridIndex : farVoxels) {
        auto it = occupancyMap_.find(gridIndex);
        if (it != occupancyMap_.end()) {
            it->second.removalReason = RemovalReason::MaxRangeExceeded;
        }
    }

    // Step 2: Perform raycasting for all inserted voxels and collect flagged voxels
    std::vector<Eigen::Vector3i> raycastedVoxels;

    #pragma omp parallel
    {
        std::vector<Eigen::Vector3i> localRaycastedVoxels;

        // Use an index-based loop instead of iterator for compatibility
        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < insertedVoxels_.size(); ++idx) {
            auto it = std::next(insertedVoxels_.begin(), idx);
            const auto& voxel = it->second;

            // Perform raycasting from vehiclePosition_ to voxel center
            auto raycastResult = performRaycast(vehiclePosition_, voxel.centerPosition);

            // Append results to thread-local collection
            localRaycastedVoxels.insert(localRaycastedVoxels.end(),
                                        raycastResult.begin(),
                                        raycastResult.end());
        }

        // Combine thread-local results safely
        #pragma omp critical
        raycastedVoxels.insert(raycastedVoxels.end(), localRaycastedVoxels.begin(), localRaycastedVoxels.end());
    }

    // Mark all raycasted voxels
    for (const auto& gridIndex : raycastedVoxels) {
        auto it = occupancyMap_.find(gridIndex);
        if (it != occupancyMap_.end()) {  // Ensure the voxel exists
            it->second.removalReason = RemovalReason::Raycasting;
        }
    }
}
//##############################################################################
// Perform raycast to remove voxel
void OccupancyMap::markDynamicVoxels(const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud) {
    // Step 1: Collect unique grid indices from the dynamic point cloud
    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> insertedDynamicVoxels_;

    #pragma omp parallel
    {
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> localDynamicVoxels;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < dynamicCloud.size(); ++i) {
            Eigen::Vector3i gridIndex = posToGridIndex(dynamicCloud[i].position);
            localDynamicVoxels[gridIndex] = {};
        }

        // Merge thread-local maps into the global map
        #pragma omp critical
        {
            for (const auto& [gridIndex, voxel] : localDynamicVoxels) {
                insertedDynamicVoxels_[gridIndex] = voxel;
            }
        }
    }

    // Step 2: Collect grid indices of dynamic voxels in a thread-safe manner
    std::vector<Eigen::Vector3i> dynamicVoxelIndices;

    #pragma omp parallel
    {
        std::vector<Eigen::Vector3i> localDynamicVoxelIndices;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < dynamicCloud.size(); ++i) {
            Eigen::Vector3i gridIndex = posToGridIndex(dynamicCloud[i].position);

            auto mapIt = occupancyMap_.find(gridIndex);
            if (mapIt != occupancyMap_.end()) {
                localDynamicVoxelIndices.push_back(gridIndex);
            }
        }

        // Merge thread-local indices into the global list
        #pragma omp critical
        dynamicVoxelIndices.insert(dynamicVoxelIndices.end(), localDynamicVoxelIndices.begin(), localDynamicVoxelIndices.end());
    }

    // Step 3: Mark the collected grid indices as dynamic in occupancyMap_
    for (const auto& gridIndex : dynamicVoxelIndices) {
        auto mapIt = occupancyMap_.find(gridIndex);
        if (mapIt != occupancyMap_.end()) {
            mapIt->second.isDynamic = true;
            mapIt->second.removalReason = RemovalReason::Dynamic;
        }
    }
}

//##############################################################################
// Perform pruning voxel
void OccupancyMap::removeFlaggedVoxels() {
    // Step 1: Collect flagged keys in parallel
    std::vector<Eigen::Vector3i> flaggedKeys;

    #pragma omp parallel
    {
        // Thread-local storage for flagged keys
        std::vector<Eigen::Vector3i> localFlaggedKeys;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < occupancyMap_.size(); ++idx) {
            auto it = std::next(occupancyMap_.begin(), idx);
            if (it->second.removalReason != RemovalReason::None) {
                localFlaggedKeys.push_back(it->first);
            }
        }

        // Combine thread-local results into the global list
        #pragma omp critical
        {
            flaggedKeys.insert(flaggedKeys.end(), localFlaggedKeys.begin(), localFlaggedKeys.end());
        }
    }

    // Step 2: Remove flagged keys sequentially
    for (const auto& key : flaggedKeys) {
        occupancyMap_.erase(key);
    }
}
//##############################################################################
// getDynamicVoxels
std::vector<OccupancyMap::VoxelData> OccupancyMap::getDynamicVoxels() const {
    if (occupancyMap_.empty()) {
        return {};  // Return an empty vector if the map is empty
    }

    // Parallel reduction for dynamic voxels
    std::vector<VoxelData> dynamicVoxels;

    #pragma omp parallel
    {
        // Thread-local storage for dynamic voxels
        std::vector<VoxelData> localDynamicVoxels;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < occupancyMap_.size(); ++idx) {
            auto it = std::next(occupancyMap_.begin(), idx);  // Safe iterator access
            const auto& voxel = it->second;
            if (voxel.isDynamic) {
                localDynamicVoxels.push_back(voxel);
            }
        }

        // Combine thread-local results into the global list
        #pragma omp critical
        {
            dynamicVoxels.insert(dynamicVoxels.end(), localDynamicVoxels.begin(), localDynamicVoxels.end());
        }
    }

    return dynamicVoxels;
}
//##############################################################################
// getStaticVoxels
std::vector<OccupancyMap::VoxelData> OccupancyMap::getStaticVoxels() const {
    // Return an empty vector if occupancyMap_ is empty
    if (occupancyMap_.empty()) {
        return {};  // Return an empty vector
    }

    // Parallel reduction for static voxels
    std::vector<VoxelData> staticVoxels;

    #pragma omp parallel
    {
        // Thread-local storage for static voxels
        std::vector<VoxelData> localStaticVoxels;

        #pragma omp for schedule(static)
        for (size_t idx = 0; idx < occupancyMap_.size(); ++idx) {
            auto it = std::next(occupancyMap_.begin(), idx);  // Safe iterator advancement
            const auto& voxel = it->second;
            if (!voxel.isDynamic) {
                localStaticVoxels.push_back(voxel);
            }
        }

        // Merge thread-local results into the global list
        #pragma omp critical
        {
            staticVoxels.insert(staticVoxels.end(), localStaticVoxels.begin(), localStaticVoxels.end());
        }
    }

    return staticVoxels;
}
//##############################################################################
// getVoxelCenters
std::vector<Eigen::Vector3f> OccupancyMap::getVoxelCenters(const std::vector<OccupancyMap::VoxelData>& voxels) {
    // Preallocate the exact size needed to avoid resizing and reserve overhead
    std::vector<Eigen::Vector3f> voxelCenters(voxels.size());

    // Use std::transform to fill voxelCenters with center positions from each voxel
    std::transform(voxels.begin(), voxels.end(), voxelCenters.begin(),
                   [](const VoxelData& voxel) { return voxel.centerPosition; });

    return voxelCenters;
}
//##############################################################################
// Calculate occupancy-based grayscale color
Eigen::Vector3i OccupancyMap::calculateOccupancyColor(const OccupancyMap::VoxelData& voxel) {
    int occupancyColorValue = static_cast<int>(255.0 * std::min(static_cast<int>(voxel.points.size()), maxPointsPerVoxel_) / maxPointsPerVoxel_);
    return Eigen::Vector3i(occupancyColorValue, occupancyColorValue, occupancyColorValue);
}
//##############################################################################
// Calculate reflectivity color
Eigen::Vector3i OccupancyMap::calculateReflectivityColor(float avgReflectivity) {
    int reflectivityColorValue;

    if (avgReflectivity <= 100.0f) {
        reflectivityColorValue = static_cast<int>(avgReflectivity * 2.55f);  // Linear scale 0–100 to 0–255
    } else {
        float transitionFactor = 0.2f;
        if (avgReflectivity <= 110.0f) {
            float linearComponent = 2.55f * avgReflectivity;
            float logComponent = 155.0f + (100.0f * (std::log2(avgReflectivity - 100.0f + 1.0f) / std::log2(156.0f)));
            reflectivityColorValue = static_cast<int>((1.0f - transitionFactor) * linearComponent + transitionFactor * logComponent);
        } else {
            float logReflectivity = std::log2(avgReflectivity - 100.0f + 1.0f) / std::log2(156.0f);  
            reflectivityColorValue = static_cast<int>(155.0f + logReflectivity * 100.0f); 
        }
    }
    return Eigen::Vector3i(std::clamp(reflectivityColorValue, 0, 255), 
                           std::clamp(reflectivityColorValue, 0, 255), 
                           std::clamp(reflectivityColorValue, 0, 255));
}
//##############################################################################
// Calculate intensity-based color
Eigen::Vector3i OccupancyMap::calculateIntensityColor(float avgIntensity) {
    int intensityColorValue = static_cast<int>(std::clamp(avgIntensity, 0.0f, 255.0f));  // Clamped 0–255
    return Eigen::Vector3i(intensityColorValue, intensityColorValue, intensityColorValue);
}
//##############################################################################
// Calculate NIR-based color
Eigen::Vector3i OccupancyMap::calculateNIRColor(float avgNIR) {
    // Clamp avgNIR directly to 0-255 range, as NIR values over 255 should still map to 255
    int NIRColorValue = static_cast<int>(std::clamp(avgNIR, 0.0f, 255.0f));
    return Eigen::Vector3i(NIRColorValue, NIRColorValue, NIRColorValue);
}
//##############################################################################
// Assuming calculateOccupancyColor, calculateReflectivityColor, calculateIntensityColor, calculateNIRColor are defined as shown in previous examples
std::tuple<std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>> 
OccupancyMap::computeVoxelColors(const std::vector<OccupancyMap::VoxelData>& voxels) {
    // Check if input is empty
    if (voxels.empty()) {
        return {{}, {}, {}, {}};  // Return empty vectors if no data
    }

    // Initialize color vectors for each characteristic with reserved space
    std::vector<Eigen::Vector3i> occupancyColors(voxels.size());
    std::vector<Eigen::Vector3i> reflectivityColors(voxels.size());
    std::vector<Eigen::Vector3i> intensityColors(voxels.size());
    std::vector<Eigen::Vector3i> NIRColors(voxels.size());

    // Parallel processing to compute colors for each voxel
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < voxels.size(); ++i) {
        const VoxelData& voxel = voxels[i];
        // Calculate each characteristic color for the voxel
        occupancyColors[i] = calculateOccupancyColor(voxel);
        reflectivityColors[i] = calculateReflectivityColor(voxel.avgReflectivity);
        intensityColors[i] = calculateIntensityColor(voxel.avgIntensity);
        NIRColors[i] = calculateNIRColor(voxel.avgNIR);
    }

    // Return all color vectors as a tuple
    return std::make_tuple(occupancyColors, reflectivityColors, intensityColors, NIRColors);
}
//##############################################################################
// Assign all voxels a red color
std::vector<Eigen::Vector3i> OccupancyMap::assignVoxelColorsRed(const std::vector<OccupancyMap::VoxelData>& voxels) {
    if (voxels.empty()) {
        return {};  // Return empty vectors if no data
    }
    std::vector<Eigen::Vector3i> colors;
    colors.assign(voxels.size(), Eigen::Vector3i(255, 0, 0));  // Direct assignment of red color to all entries
    return colors;
}





