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

// -----------------------------------------------------------------------------
// Section: OccupancyMap
// -----------------------------------------------------------------------------

OccupancyMap::OccupancyMap(float mapRes, float reachingDistance, Eigen::Vector3f mapCenter)
    : mapRes_(mapRes), reachingDistance_(reachingDistance), mapCenter_(mapCenter)  {
    // constructor body
}

// -----------------------------------------------------------------------------
// Section: runOccupancyMapPipeline
// -----------------------------------------------------------------------------

void OccupancyMap::runOccupancyMapPipeline(const std::vector<Eigen::Vector3f>& pointCloud,
                                           const std::vector<float>& reflectivity,
                                           const std::vector<float>& intensity,
                                           const std::vector<float>& NIR,
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

    std::cout << "Function OccupancyMap running okay.\n";
}

// -----------------------------------------------------------------------------
// Section: updateVehiclePosition
// -----------------------------------------------------------------------------

void OccupancyMap::updateVehiclePosition(const Eigen::Vector3f& newPosition) {
    vehiclePosition_ = newPosition;
}

// -----------------------------------------------------------------------------
// Section: updateCurrentFrame
// -----------------------------------------------------------------------------

void OccupancyMap::updateCurrentFrame(uint32_t newFrame) {
    currentFrame_ = newFrame;
}

// -----------------------------------------------------------------------------
// Section: worldToGrid
// -----------------------------------------------------------------------------

Eigen::Vector3i OccupancyMap::worldToGrid(const Eigen::Vector3f& pos) const {
    Eigen::Array3f scaledPos = (pos - mapCenter_).array() * (1.0 / mapRes_);
    return Eigen::Vector3i(std::floor(scaledPos.x()), std::floor(scaledPos.y()), std::floor(scaledPos.z()));
}

// -----------------------------------------------------------------------------
// Section: gridToWorld
// -----------------------------------------------------------------------------

Eigen::Vector3f OccupancyMap::gridToWorld(const Eigen::Vector3i& gridIndex) const {
    static const Eigen::Vector3f halfRes(mapRes_ / 2, mapRes_ / 2, mapRes_ / 2);
    return mapCenter_ + gridIndex.cast<float>() * mapRes_ + halfRes;
}

// -----------------------------------------------------------------------------
// Section: updateVoxelAverages
// -----------------------------------------------------------------------------

void OccupancyMap::updateVoxelAverages(VoxelData& voxel) const {
    const uint64_t pointCount = voxel.points.size();
    if (pointCount == 0) return;

    double invPointCount = 1.0 / pointCount;
    voxel.avgReflectivity = voxel.totalReflectivity * invPointCount;
    voxel.avgIntensity = voxel.totalIntensity * invPointCount;
    voxel.avgNIR = voxel.totalNIR * invPointCount;
}

// -----------------------------------------------------------------------------
// Section: insertPointCloud
// -----------------------------------------------------------------------------

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
                Eigen::Vector3i gridIndex = worldToGrid(point);
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

// -----------------------------------------------------------------------------
// Section: performRaycast
// -----------------------------------------------------------------------------

std::vector<Eigen::Vector3i> OccupancyMap::performRaycast(const Eigen::Vector3f& start, const Eigen::Vector3f& end) {
    // Define a small threshold for "close" points
    const float closeThreshold = 1e-3;

    // If start and end points are too close, return the starting voxel only
    if ((end - start).squaredNorm() < closeThreshold * closeThreshold) {
        return { worldToGrid(start) };
    }

    // Convert start and end positions to voxel indices
    Eigen::Vector3i startVoxel = worldToGrid(start);
    Eigen::Vector3i endVoxel = worldToGrid(end);

    // Initialize voxel collection using a concurrent container
    tbb::concurrent_vector<Eigen::Vector3i> voxelIndices;

    // Initialize Bresenham's algorithm parameters
    Eigen::Vector3i delta = (endVoxel - startVoxel).cwiseAbs();
    Eigen::Vector3i step = (endVoxel - startVoxel).cwiseSign();
    Eigen::Vector3i currentVoxel = startVoxel;

    voxelIndices.push_back(currentVoxel); // Include the starting voxel

    // Determine the dominant axis
    int maxAxis = delta.maxCoeff();
    int primaryAxis = (delta.x() >= delta.y() && delta.x() >= delta.z()) ? 0 : (delta.y() >= delta.z() ? 1 : 2);

    // Initialize error terms for Bresenham's algorithm
    int error1 = 2 * delta[(primaryAxis + 1) % 3] - delta[primaryAxis];
    int error2 = 2 * delta[(primaryAxis + 2) % 3] - delta[primaryAxis];

    // Traverse the line using Bresenham's algorithm
    for (int i = 0; i < delta[primaryAxis]; ++i) {
        currentVoxel[primaryAxis] += step[primaryAxis];

        if (error1 > 0) {
            currentVoxel[(primaryAxis + 1) % 3] += step[(primaryAxis + 1) % 3];
            error1 -= 2 * delta[primaryAxis];
        }

        if (error2 > 0) {
            currentVoxel[(primaryAxis + 2) % 3] += step[(primaryAxis + 2) % 3];
            error2 -= 2 * delta[primaryAxis];
        }

        error1 += 2 * delta[(primaryAxis + 1) % 3];
        error2 += 2 * delta[(primaryAxis + 2) % 3];

        // Add the current voxel to the result
        voxelIndices.push_back(currentVoxel);
    }

    // Return results as a standard vector
    return std::vector<Eigen::Vector3i>(voxelIndices.begin(), voxelIndices.end());
}

// -----------------------------------------------------------------------------
// Section: markVoxelsForClearing
// -----------------------------------------------------------------------------

void OccupancyMap::markVoxelsForClearing() {
    // Create per-frame caches
    tbb::concurrent_hash_map<std::pair<Eigen::Vector3i, Eigen::Vector3i>, std::vector<Eigen::Vector3i>, VoxelPairHash> startEndCache;

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

        // Use start-end pair cache
        auto rayKey = std::make_pair(worldToGrid(vehiclePosition_), worldToGrid(voxel.centerPosition));
        tbb::concurrent_hash_map<std::pair<Eigen::Vector3i, Eigen::Vector3i>, std::vector<Eigen::Vector3i>, VoxelPairHash>::accessor startEndAccessor;

        // Check the start-end pair cache first
        if (startEndCache.find(startEndAccessor, rayKey)) {
            auto& raycastVoxels = startEndAccessor->second;
            startEndAccessor.release();

            // Process cached raycast result
            for (const auto& rayVoxel : raycastVoxels) {
                auto& targetVoxel = occupancyMap_[rayVoxel];
                targetVoxel.removalReason = RemovalReason::Raycasting;
            }
            return;
        }

        // If no cache hit, compute raycast
        auto raycastVoxels = performRaycast(vehiclePosition_, voxel.centerPosition);

        // Update both caches
        startEndCache.insert({rayKey, raycastVoxels});

        // Process raycast result
        for (const auto& rayVoxel : raycastVoxels) {
            auto& targetVoxel = occupancyMap_[rayVoxel];
            targetVoxel.removalReason = RemovalReason::Raycasting;
        }
    });
}

// -----------------------------------------------------------------------------
// Section: markDynamicVoxels
// -----------------------------------------------------------------------------

void OccupancyMap::markDynamicVoxels(const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud) {
    // Step 1: Collect unique grid indices from the dynamic point cloud
    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> insertedDynamicVoxels_;

    for (const auto& point : dynamicCloud) {
        Eigen::Vector3i gridIndex = worldToGrid(point.position);
        insertedDynamicVoxels_[gridIndex] = {};
    }

    // Step 2: Mark corresponding voxels in occupancyMap_ as dynamic if they exist
    for (const auto& [gridIndex, voxel] : insertedDynamicVoxels_) {
        if (occupancyMap_.find(gridIndex) != occupancyMap_.end()) { // Ensure voxel exists
            auto& targetVoxel = occupancyMap_[gridIndex];
            targetVoxel.isDynamic = true;
            targetVoxel.removalReason = RemovalReason::Dynamic;
        }
    }
}

// -----------------------------------------------------------------------------
// Section: removeFlaggedVoxels
// -----------------------------------------------------------------------------

void OccupancyMap::removeFlaggedVoxels() {
    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> newOccupancyMap;

    tbb::parallel_for(occupancyMap_.begin(), occupancyMap_.end(), [&](const auto& item) {
        if (item.second.removalReason == RemovalReason::None) {
            std::lock_guard<std::mutex> lock(dataMutex); // Protect writes to newOccupancyMap
            newOccupancyMap[item.first] = item.second;
        }
    });

    // Replace the old map with the new map
    std::swap(occupancyMap_, newOccupancyMap);
}

// -----------------------------------------------------------------------------
// Section: getDynamicVoxels
// -----------------------------------------------------------------------------

std::vector<OccupancyMap::VoxelData> OccupancyMap::getDynamicVoxels() const {
    std::vector<VoxelData> dynamicVoxels;
    dynamicVoxels.reserve(occupancyMap_.size());  // Estimate 10% of voxels are dynamic

    // Use a concurrent vector if TBB is available for parallel processing
    tbb::concurrent_vector<VoxelData> concurrentDynamicVoxels;

    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), [&](const auto& entry) {
        const auto& voxel = entry.second;
        if (voxel.isDynamic) {
            concurrentDynamicVoxels.push_back(voxel);
        }
    });

    // Convert concurrent vector to standard vector for output
    dynamicVoxels.assign(concurrentDynamicVoxels.begin(), concurrentDynamicVoxels.end());
    return dynamicVoxels;
}

// -----------------------------------------------------------------------------
// Section: getStaticVoxels
// -----------------------------------------------------------------------------

std::vector<OccupancyMap::VoxelData> OccupancyMap::getStaticVoxels() const {
    std::vector<VoxelData> staticVoxels;
    staticVoxels.reserve(occupancyMap_.size());

    // Use a concurrent vector to allow parallel writes without locks
    tbb::concurrent_vector<VoxelData> concurrentStaticVoxels;

    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), [&](const auto& entry) {
        const auto& voxel = entry.second;
        if (!voxel.isDynamic) {
            concurrentStaticVoxels.push_back(voxel);
        }
    });

    // Convert concurrent vector to standard vector for output
    staticVoxels.assign(concurrentStaticVoxels.begin(), concurrentStaticVoxels.end());
    return staticVoxels;
}

// -----------------------------------------------------------------------------
// Section: getVoxelCenters
// -----------------------------------------------------------------------------

std::vector<Eigen::Vector3f> OccupancyMap::getVoxelCenters(const std::vector<OccupancyMap::VoxelData>& voxels) {
    // Preallocate the exact size needed to avoid resizing and reserve overhead
    std::vector<Eigen::Vector3f> voxelCenters(voxels.size());

    // Use std::transform to fill voxelCenters with center positions from each voxel
    std::transform(voxels.begin(), voxels.end(), voxelCenters.begin(),
                   [](const VoxelData& voxel) { return voxel.centerPosition; });

    return voxelCenters;
}

// -----------------------------------------------------------------------------
// Section: calculateOccupancyColor
// -----------------------------------------------------------------------------

Eigen::Vector3i OccupancyMap::calculateOccupancyColor(const OccupancyMap::VoxelData& voxel) {
    int occupancyColorValue = static_cast<int>(255.0 * std::min(static_cast<int>(voxel.points.size()), maxPointsPerVoxel_) / maxPointsPerVoxel_);
    return Eigen::Vector3i(occupancyColorValue, occupancyColorValue, occupancyColorValue);
}

// -----------------------------------------------------------------------------
// Section: calculateReflectivityColor
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// Section: calculateIntensityColor
// -----------------------------------------------------------------------------

Eigen::Vector3i OccupancyMap::calculateIntensityColor(float avgIntensity) {
    int intensityColorValue = static_cast<int>(std::clamp(avgIntensity, 0.0f, 255.0f));  // Clamped 0–255
    return Eigen::Vector3i(intensityColorValue, intensityColorValue, intensityColorValue);
}

// -----------------------------------------------------------------------------
// Section: calculateNIRColor
// -----------------------------------------------------------------------------

Eigen::Vector3i OccupancyMap::calculateNIRColor(float avgNIR) {
    // Clamp avgNIR directly to 0-255 range, as NIR values over 255 should still map to 255
    int NIRColorValue = static_cast<int>(std::clamp(avgNIR, 0.0f, 255.0f));
    return Eigen::Vector3i(NIRColorValue, NIRColorValue, NIRColorValue);
}

// -----------------------------------------------------------------------------
// Section: computeVoxelColors
// -----------------------------------------------------------------------------

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
    tbb::parallel_for(uint64_t(0), voxels.size(), [&](uint64_t i) {
        const VoxelData& voxel = voxels[i];
        // Calculate each characteristic color for the voxel
        occupancyColors[i] = calculateOccupancyColor(voxel);
        reflectivityColors[i] = calculateReflectivityColor(voxel.avgReflectivity);
        intensityColors[i] = calculateIntensityColor(voxel.avgIntensity);
        NIRColors[i] = calculateNIRColor(voxel.avgNIR);
    });

    // Return all color vectors as a tuple
    return std::make_tuple(occupancyColors, reflectivityColors, intensityColors, NIRColors);
}

// -----------------------------------------------------------------------------
// Section: assignVoxelColorsRed
// -----------------------------------------------------------------------------

std::vector<Eigen::Vector3i> OccupancyMap::assignVoxelColorsRed(const std::vector<OccupancyMap::VoxelData>& voxels) {
    std::vector<Eigen::Vector3i> colors;
    colors.assign(voxels.size(), Eigen::Vector3i(255, 0, 0));  // Direct assignment of red color to all entries
    return colors;
}