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
OccupancyMap::OccupancyMap(double mapRes,
                            double reachingDistance,
                            Eigen::Vector3d mapCenter,
                            uint32_t decayFactor)
    : mapRes_(mapRes), 
      reachingDistance_(reachingDistance),
      mapCenter_(mapCenter),
      decayFactor_(decayFactor)
{}
//##############################################################################
// Main Pipeline
void OccupancyMap::runOccupancyMapPipeline(const std::vector<Eigen::Vector3d>& pointCloud,
                                                   const std::vector<float>& reflectivity,
                                                   const std::vector<float>& intensity,
                                                   const std::vector<float>& NIR,
                                                   const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud,
                                                   const Eigen::Vector3d& newPosition,
                                                   uint32_t newFrame) {
    // Step 0: update persistant
    updateVehiclePosition(newPosition);
    updateCurrentFrame(newFrame);
    // Step 1: inserting the point cloud
    insertPointCloud(pointCloud, reflectivity, intensity, NIR);
    // Step 2: markVoxelsForClearing
    markVoxelsForClearing();
    // Step 3: removeFlaggedVoxels
    removeFlaggedVoxels();
    // Step 4: markDynamicVoxels
    markDynamicVoxels(dynamicCloud);
}
//##############################################################################
// Function to update the vehicle position
void OccupancyMap::updateVehiclePosition(const Eigen::Vector3d& newPosition) {
    vehiclePosition_ = newPosition;
}
//##############################################################################
// Function to update the current frame
void OccupancyMap::updateCurrentFrame(uint32_t newFrame) {
    currentFrame_ = newFrame;
}
//##############################################################################
// Convert position to voxel grid index
Eigen::Vector3i OccupancyMap::posToGridIndex(const Eigen::Vector3d& pos) const {
    return Eigen::Vector3i(
        std::floor((pos.x() - mapCenter_.x()) / mapRes_),
        std::floor((pos.y() - mapCenter_.y()) / mapRes_),
        std::floor((pos.z() - mapCenter_.z()) / mapRes_)
    );
}
//##############################################################################
// Convert grid index back to world position (center of voxel)
Eigen::Vector3d OccupancyMap::gridToWorld(const Eigen::Vector3i& gridIndex) const {
    return mapCenter_ + Eigen::Vector3d(gridIndex.x() * mapRes_,
                                        gridIndex.y() * mapRes_,
                                        gridIndex.z() * mapRes_) + 
                                        Eigen::Vector3d(mapRes_ / 2, mapRes_ / 2, mapRes_ / 2); // Center offset
}
//##############################################################################
// Calculate average values for a voxel
void OccupancyMap::updateVoxelAverages(VoxelData& voxel) const {
    if (voxel.points.empty()) return;
    voxel.avgReflectivity = voxel.totalReflectivity / voxel.points.size();
    voxel.avgIntensity = voxel.totalIntensity / voxel.points.size();
    voxel.avgNIR = voxel.totalNIR / voxel.points.size();
}
//##############################################################################
// insert the point cloud into occupancy map
void OccupancyMap::insertPointCloud(const std::vector<Eigen::Vector3d>& pointCloud,
                                   const std::vector<float>& reflectivity,
                                   const std::vector<float>& intensity,
                                   const std::vector<float>& NIR) {
    // Clear the list of inserted voxels for the new frame
    insertedVoxels_.clear();

    // Use TBB parallel_reduce to handle the point cloud in parallel
    auto localMap = tbb::parallel_reduce(
        tbb::blocked_range<uint64_t>(0, pointCloud.size(), 1024),
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>(),
        [&](const tbb::blocked_range<uint64_t>& range,
            tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> localMap) {

            for (uint64_t i = range.begin(); i < range.end(); ++i) {
                const Eigen::Vector3d& point = pointCloud[i];
                Eigen::Vector3i gridIndex = posToGridIndex(point);
                auto& voxel = localMap[gridIndex];

                // Initialize the voxel center position if it's empty
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
            }
            return a;
        }
    );

    // Final merge from localMap into the main occupancy map and track inserted voxels
    for (auto& [gridIndex, localVoxel] : localMap) {
        auto& voxel = occupancyMap_[gridIndex];

        if (voxel.points.empty()) {
            // New voxel case: move localVoxel to occupancyMap_
            voxel = std::move(localVoxel);
        } else {
            // Determine the number of points that can be added without exceeding maxPointsPerVoxel_
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
        insertedVoxels_[gridIndex] = {};  // Store gridIndex alone if no additional data is needed
    }
}
//##############################################################################
// Perform raycast to remove voxel
std::vector<Eigen::Vector3i> OccupancyMap::performRaycast(const Eigen::Vector3d& start, const Eigen::Vector3d& end) {
    // Define a small tolerance for "close" points
    const double closeThreshold = 1e-3; // or any small distance relevant to your map resolution

    // Check if points are too close
    if ((end - start).norm() < closeThreshold) {
        // If close, return just the starting voxel
        return { posToGridIndex(start) };
    }
    tsl::robin_set<Eigen::Vector3i, Vector3iHash, Vector3iEqual> uniqueVoxels;
    Eigen::Vector3d direction = (end - start).normalized();
    double distance = (end - start).norm();

    Eigen::Vector3i lastVoxelIndex = posToGridIndex(start);  // Initialize with starting voxel
    uniqueVoxels.insert(lastVoxelIndex);                     // Add start voxel to set

    for (double step = mapRes_; step < distance; step += mapRes_) {
        Eigen::Vector3d currentPos = start + direction * step;
        Eigen::Vector3i voxelIndex = posToGridIndex(currentPos);

        // Add only if the voxel index has changed to avoid redundant checks
        if (voxelIndex != lastVoxelIndex) {
            uniqueVoxels.insert(voxelIndex);
            lastVoxelIndex = voxelIndex;  // Update last visited voxel
        }
    }
    // Convert the set of unique voxels to a vector and return it
    return std::vector<Eigen::Vector3i>(uniqueVoxels.begin(), uniqueVoxels.end());
}
//##############################################################################
// Perform marking Voxels For Clearing
void OccupancyMap::markVoxelsForClearing() {
    // First parallel task: Mark voxels beyond the maximum reaching distance
    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), [&](auto& mapEntry) {
        auto& [gridIndex, voxel] = mapEntry;

        // Check if the voxel is beyond the maximum reaching distance
        if ((voxel.centerPosition - vehiclePosition_).norm() > reachingDistance_) {
            voxel.removalReason = RemovalReason::MaxRangeExceeded;
        }
    });

    // Second parallel task: Perform raycasting for each voxel in insertedVoxels_
    tbb::parallel_for_each(insertedVoxels_.begin(), insertedVoxels_.end(), [&](const auto& insertedEntry) {
        const auto& [gridIndex, voxel] = insertedEntry;

        // Perform raycasting from vehiclePosition_ to voxel center
        for (const auto& rayVoxel : performRaycast(vehiclePosition_, voxel.centerPosition)) {
            // Access the intersected voxel in occupancyMap_
            auto& targetVoxel = occupancyMap_[rayVoxel];

            // Flag the voxel for removal due to raycasting and adjust stability score
            targetVoxel.removalReason = RemovalReason::Raycasting;
        }
    });
}
//##############################################################################
// Perform raycast to remove voxel
void OccupancyMap::markDynamicVoxels(const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud) {
    // Step 1: Collect unique grid indices from the dynamic point cloud
    tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> insertedDynamicVoxels_;

    for (const auto& point : dynamicCloud) {
        // Calculate the voxel index for this point's position
        Eigen::Vector3i gridIndex = posToGridIndex(point.position);
        insertedDynamicVoxels_[gridIndex] = {};  // Insert unique grid indices only
    }

    // Step 2: Parallel task to mark corresponding voxels in occupancy map as dynamic
    tbb::parallel_for_each(insertedDynamicVoxels_.begin(), insertedDynamicVoxels_.end(), [&](const auto& insertedEntry) {
        const auto& [gridIndex, voxel] = insertedEntry;

        // Find the target voxel in occupancyMap_ and mark it as dynamic if it exists
        auto mapIt = occupancyMap_.find(gridIndex);
        if (mapIt != occupancyMap_.end()) {
            mapIt->second.isDynamic = true;
            mapIt->second.isDynamic = RemovalReason::Dynamic;
        }
    });
}
//##############################################################################
// Perform pruning voxel
void OccupancyMap::removeFlaggedVoxels() {
    for (auto it = occupancyMap_.begin(); it != occupancyMap_.end();) {
        // Check if the voxel is flagged for removal
        if (it->second.removalReason != RemovalReason::None) {
            // Erase the voxel and get the iterator to the next element
            it = occupancyMap_.erase(it);
        } else {
            // Move to the next element
            ++it;
        }
    }
}
//##############################################################################
// getDynamicVoxels
std::vector<OccupancyMap::VoxelData> OccupancyMap::getDynamicVoxels() const {
    std::vector<VoxelData> dynamicVoxels;

    for (const auto& [gridIndex, voxel] : occupancyMap_) {
        if (voxel.isDynamic) {
            dynamicVoxels.push_back(voxel);  // Add the dynamic voxel to the result vector
        }
    }

    return dynamicVoxels;  // Return the list of dynamic voxels
}
//##############################################################################
// getStaticVoxels
std::vector<OccupancyMap::VoxelData> OccupancyMap::getStaticVoxels() const {
    std::vector<VoxelData> staticVoxels;

    for (const auto& [gridIndex, voxel] : occupancyMap_) {
        if (!voxel.isDynamic) {
            staticVoxels.push_back(voxel);  // Add the static voxel to the result vector
        }
    }

    return staticVoxels;  // Return the list of static voxels
}
//##############################################################################
// getVoxelCenters
std::vector<Eigen::Vector3d> OccupancyMap::getVoxelCenters(const std::vector<OccupancyMap::VoxelData>& voxels) {
    std::vector<Eigen::Vector3d> voxelCenters;
    voxelCenters.reserve(voxels.size());  // Reserve space to improve performance

    for (const auto& voxel : voxels) {
        voxelCenters.push_back(voxel.centerPosition);  // Collect the center position
    }
    return voxelCenters;  // Return the list of center positions
}
//##############################################################################
// Calculate occupancy-based grayscale color
Eigen::Vector3i OccupancyMap::calculateOccupancyColor(const OccupancyMap::VoxelData& voxel) {
    int occupancy = std::min(static_cast<int>(voxel.points.size()), maxPointsPerVoxel_);
    int occupancyColorValue = static_cast<int>(255.0 * occupancy / maxPointsPerVoxel_);
    return Eigen::Vector3i(occupancyColorValue, occupancyColorValue, occupancyColorValue);
}
//##############################################################################
// Calculate reflectivity color
Eigen::Vector3i OccupancyMap::calculateReflectivityColor(double avgReflectivity) {
    int reflectivityColorValue;

    if (avgReflectivity <= 100) {
        // Linear mapping for values 0-100
        reflectivityColorValue = static_cast<int>(avgReflectivity * 2.55);  // Scale to 0-255
    } else {
        // Smooth transition zone between linear and logarithmic scales
        float transitionFactor = 0.2f;  // Adjust factor for smoothness between 100-110
        if (avgReflectivity <= 110) {
            // Blend linear and logarithmic for a smooth transition
            float linearComponent = 2.55 * avgReflectivity;
            float logComponent = 155 + (100 * (std::log2(avgReflectivity - 100 + 1) / std::log2(156)));
            reflectivityColorValue = static_cast<int>((1 - transitionFactor) * linearComponent + transitionFactor * logComponent);
        } else {
            // Logarithmic mapping for values above 110
            float logReflectivity = std::log2(avgReflectivity - 100 + 1) / std::log2(156);  // Normalized log scaling
            reflectivityColorValue = static_cast<int>(155 + logReflectivity * 100);  // Maps to 155-255
        }
    }

    // Clamp to ensure the final color is in the 0-255 range
    reflectivityColorValue = std::clamp(reflectivityColorValue, 0, 255);

    // Return the color as an RGB vector with equal values for grayscale
    return Eigen::Vector3i(reflectivityColorValue, reflectivityColorValue, reflectivityColorValue);
}
//##############################################################################
// Calculate intensity-based color
Eigen::Vector3i OccupancyMap::calculateIntensityColor(double avgIntensity) {
    // Cap avgIntensity at 300 and then scale it down to fit within the 0-255 range
    double clampedIntensity = std::min(255.0, avgIntensity);
    int intensityColorValue = static_cast<int>((clampedIntensity / 255.0) * 255.0);  // Scale 0-300 to 0-255

    return Eigen::Vector3i(intensityColorValue, intensityColorValue, intensityColorValue);
}

//##############################################################################
// Calculate NIR-based color
Eigen::Vector3i OccupancyMap::calculateNIRColor(double avgNIR) {

    double clampedNIR = std::min(255.0, avgNIR);
    int NIRColorValue = static_cast<int>((clampedNIR / 255.0) * 255.0);  // Scale 0-300 to 0-255

    return Eigen::Vector3i(NIRColorValue, NIRColorValue, NIRColorValue);
}
//##############################################################################
// Assuming calculateOccupancyColor, calculateReflectivityColor, calculateIntensityColor, calculateNIRColor are defined as shown in previous examples
std::tuple<std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>> OccupancyMap::computeVoxelColors(const std::vector<OccupancyMap::VoxelData>& voxels) {
    // Initialize color vectors for each characteristic
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
//##############################################################################
// Assuming dynamic voxel and give red colour
std::vector<Eigen::Vector3i> OccupancyMap::assignVoxelColorsRed(const std::vector<OccupancyMap::VoxelData>& voxels) {
    // Create a vector to hold the red color for each voxel
    std::vector<Eigen::Vector3i> colors(voxels.size(), Eigen::Vector3i(255, 0, 0));
    
    // All voxels are assigned the color red (255, 0, 0)
    return colors;
}








