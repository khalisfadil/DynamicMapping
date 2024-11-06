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
OccupancyMap::OccupancyMap(double resolution, const Eigen::Vector3d& map_center, double max_distance)
    : mapRes_(resolution), mapCenter_(map_center), maxDistance_(max_distance) {}

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
bool OccupancyMap::checkNeighborConsistency(const Eigen::Vector3i& gridIndex, int stabilityThreshold, int neighborRange) const {
    int neighborCount = 0;
    int weightedDynamicScore = 0;
    int totalWeight = 0;

    for (int dx = -neighborRange; dx <= neighborRange; ++dx) {
        for (int dy = -neighborRange; dy <= neighborRange; ++dy) {
            for (int dz = -neighborRange; dz <= neighborRange; ++dz) {
                if (dx == 0 && dy == 0 && dz == 0) continue;  // Skip the center voxel
                
                Eigen::Vector3i neighborIndex = gridIndex + Eigen::Vector3i(dx, dy, dz);
                auto neighborIt = occupancyMap_.find(neighborIndex);

                if (neighborIt != occupancyMap_.end()) {
                    const VoxelData& neighborVoxel = neighborIt->second;
                    neighborCount++;

                    // Calculate weight based on stability score
                    int stabilityScore = neighborVoxel.stabilityScore;
                    int weight = std::max(1, stabilityThreshold - stabilityScore); // Higher weight for lower stability

                    // Update weighted score and total weight
                    if (stabilityScore < stabilityThreshold) {
                        weightedDynamicScore += weight;
                    }
                    totalWeight += weight;

                    // Early exit if dynamic neighbors dominate
                    if (weightedDynamicScore > totalWeight / 2) {
                        return true;
                    }
                }
            }
        }
    }

    // Return true if dynamic neighbors exceed 50% of the weighted total
    return weightedDynamicScore > totalWeight / 2;
}

// ##############################################################################
// function to compute this max distance
double OccupancyMap::calculateMaxRadius(const std::vector<Eigen::Vector3d>& pointCloud, 
                                        const Eigen::Vector3d& vehiclePosition) const {
    double maxRadius = 0.0;

    for (const auto& point : pointCloud) {
        // Calculate Euclidean distance between point and vehicle position
        double distance = (point - vehiclePosition).norm();

        // Update maxRadius if this distance is greater
        maxRadius = std::max(maxRadius, distance);
    }

    return maxRadius;
}

//##############################################################################
// function to compute this max distance
void OccupancyMap::updateOccupancy(const std::vector<Eigen::Vector3d>& pointCloud,
                                   const Eigen::Vector3d& vehiclePosition,
                                   const std::vector<float>& reflectivity,
                                   const std::vector<float>& intensity,
                                   const std::vector<float>& NIR,
                                   int currentFrame,
                                   int stabilityThreshold,
                                   int decayFactor,
                                   int neighborRange) {
    // Validate input sizes
    if (pointCloud.size() != reflectivity.size() || 
        pointCloud.size() != intensity.size() || 
        pointCloud.size() != NIR.size()) {
        throw std::invalid_argument("Input vectors must have the same size.");
    }

    double reachingDistance = calculateMaxRadius(pointCloud, vehiclePosition);
    uint64_t chunkSize = 1024;

    // Step 1: Thread-local occupancy map for reducing contention
    tbb::enumerable_thread_specific<tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual>> localOccupancyMaps;

    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, pointCloud.size(), chunkSize),
                      [&](const tbb::blocked_range<uint64_t>& range) {
        auto& localMap = localOccupancyMaps.local();

        for (uint64_t i = range.begin(); i < range.end(); ++i) {
            const Eigen::Vector3d& point = pointCloud[i];
            Eigen::Vector3i gridIndex = posToGridIndex(point);

            auto& voxel = localMap[gridIndex];
            if (voxel.points.empty()) {
                voxel.centerPosition = gridToWorld(gridIndex);
            }

            // Update stability score
            voxel.stabilityScore = (voxel.lastSeenFrame == currentFrame - 1)
                                    ? voxel.stabilityScore + 1
                                    : std::max(0, voxel.stabilityScore - decayFactor);
            voxel.lastSeenFrame = currentFrame;

            // Manage points within each voxel
            if (voxel.points.size() >= MAX_POINTS_PER_VOXEL) {
                voxel.totalReflectivity -= voxel.points.front().reflectivity;
                voxel.totalIntensity -= voxel.points.front().intensity;
                voxel.totalNIR -= voxel.points.front().NIR;
                voxel.points.pop_front();
            }

            voxel.points.push_back({point, reflectivity[i], intensity[i], NIR[i]});
            voxel.totalReflectivity += reflectivity[i];
            voxel.totalIntensity += intensity[i];
            voxel.totalNIR += NIR[i];
        }
    });

    // Merge local occupancy maps into the main map
    for (auto& localMap : localOccupancyMaps) {
        for (auto& [gridIndex, localVoxel] : localMap) {
            auto& voxel = occupancyMap_[gridIndex];

            if (voxel.points.empty()) {
                voxel = std::move(localVoxel); // Move if voxel is new
            } else {
                // Accumulate if voxel exists
                voxel.totalReflectivity += localVoxel.totalReflectivity;
                voxel.totalIntensity += localVoxel.totalIntensity;
                voxel.totalNIR += localVoxel.totalNIR;
                voxel.points.insert(voxel.points.end(), localVoxel.points.begin(), localVoxel.points.end());

                // Trim excess points if necessary
                while (voxel.points.size() > MAX_POINTS_PER_VOXEL) {
                    voxel.totalReflectivity -= voxel.points.front().reflectivity;
                    voxel.totalIntensity -= voxel.points.front().intensity;
                    voxel.totalNIR -= voxel.points.front().NIR;
                    voxel.points.pop_front();
                }
            }
            updateVoxelAverages(voxel); // Only compute averages once after merging
        }
    }

    // Step 2: Mark voxels for clearing and adjust stability score
    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), [&](auto& mapEntry) {
        auto& [gridIndex, voxel] = mapEntry;

        // If beyond max distance, mark for removal due to MaxRangeExceeded
        if ((voxel.centerPosition - vehiclePosition).norm() > reachingDistance) {
            voxel.removalReason = RemovalReason::MaxRangeExceeded;
        } else {
            // Perform raycast and mark all intersected voxels for potential dynamic removal
            for (const auto& rayVoxel : performRaycast(vehiclePosition, voxel.centerPosition)) {
                auto& targetVoxel = occupancyMap_[rayVoxel];
                targetVoxel.removalReason = RemovalReason::Raycasting;
                targetVoxel.stabilityScore = std::max(0, targetVoxel.stabilityScore - decayFactor); // Reduce stability score
            }
        }
    });

    // Step 3: Clear points in each voxel marked for clearing
    tbb::parallel_for_each(occupancyMap_.begin(), occupancyMap_.end(), [&](auto& mapEntry) {
        auto& voxel = mapEntry.second;
        if (voxel.removalReason != RemovalReason::None) {
            voxel.points.clear();
        }
    });

    // Step 4: Neighbor consistency check for dynamic/static classification in parallel
    std::vector<Eigen::Vector3i> keys;
    keys.reserve(occupancyMap_.size());
    for (const auto& item : occupancyMap_) {
        keys.push_back(item.first);
    }

    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, keys.size()), [&](const tbb::blocked_range<uint64_t>& range) {
    for (uint64_t i = range.begin(); i < range.end(); ++i) {
        const auto& gridIndex = keys[i];
        auto& voxel = occupancyMap_[gridIndex];

        // Adjust stabilityScore based on removal reason
        if (voxel.removalReason == RemovalReason::Raycasting) {
            // Reduce stability score significantly to suggest potential dynamic nature
            voxel.stabilityScore = std::max(0, voxel.stabilityScore - decayFactor * 2); // Adjust multiplier as needed
        } else if (voxel.removalReason == RemovalReason::MaxRangeExceeded) {
            // Set stability score to a high value since it's out of range, treating it as static
            voxel.stabilityScore = stabilityThreshold + 1;
        } else {
            // Perform neighbor consistency check only if not marked for removal
            voxel.isDynamic = (voxel.stabilityScore < stabilityThreshold)
                              ? checkNeighborConsistency(gridIndex, stabilityThreshold, neighborRange)
                              : false;
        }
    }
});

    // Step 5: Prune voxels flagged for removal
    for (auto it = occupancyMap_.begin(); it != occupancyMap_.end();) {
        if (it->second.removalReason != RemovalReason::None) {
            it = occupancyMap_.erase(it);
        } else {
            ++it;
        }
    }
}

//##############################################################################
// get the colors for each voxel
std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i>> OccupancyMap::getOccupiedCellsWithColors() const {
    
    // Use tbb::concurrent_vector for thread-safe access during parallelism
    tbb::concurrent_vector<std::tuple<Eigen::Vector3d, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i>> coloredCells;

    // Parallel for loop over the occupancy map using an integer index
    tbb::parallel_for((unsigned long int)0, (unsigned long int)occupancyMap_.size(), [&](unsigned long int i) {
        auto it = std::next(occupancyMap_.begin(), i); // Get iterator to the current element
        const auto& [gridIndex, voxel] = *it;

        // Calculate occupancy-based grayscale color
        int occupancy = std::min(static_cast<int>(voxel.points.size()), MAX_POINTS_PER_VOXEL);
        int occupancyColorValue = static_cast<int>(255.0 * occupancy / MAX_POINTS_PER_VOXEL);
        Eigen::Vector3i occupancyColor(occupancyColorValue, occupancyColorValue, occupancyColorValue);

        // Reflectivity color mapping using updated method
        int reflectivityColorValue = calculateReflectivityColor(voxel.avgReflectivity);
        Eigen::Vector3i reflectivityColor(reflectivityColorValue, reflectivityColorValue, reflectivityColorValue);

        // Signal intensity color (scaled 0-65535 to 0-255)
        int intensityColorValue = static_cast<int>(255.0 * voxel.avgIntensity / 65535.0);
        Eigen::Vector3i intensityColor(intensityColorValue, intensityColorValue, intensityColorValue);

        // NIR color (scaled 0-65535 to 0-255)
        int NIRColorValue = static_cast<int>(255.0 * voxel.avgNIR / 65535.0);
        Eigen::Vector3i NIRColor(NIRColorValue, NIRColorValue, NIRColorValue);

        // Add the center position and each color metric to the output list
        coloredCells.emplace_back(voxel.centerPosition, occupancyColor, reflectivityColor, intensityColor, NIRColor);
    });

    // Convert the concurrent vector to a standard vector and return
    return std::vector<std::tuple<Eigen::Vector3d, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i, Eigen::Vector3i>>(coloredCells.begin(), coloredCells.end());
}


//##############################################################################
int OccupancyMap::calculateReflectivityColor(float reflectivity) const {
    int reflectivityColorValue;

    if (reflectivity <= 100) {
        // Linear mapping for values 0-100
        reflectivityColorValue = static_cast<int>(reflectivity * 2.55); // Scale to 0-255
    } else {
        // Smooth transition zone between linear and logarithmic scales
        float transitionFactor = 0.2f;  // Adjust factor for smoothness between 100-110
        if (reflectivity <= 110) {
            // Blend linear and logarithmic for a smooth transition
            float linearComponent = 2.55 * reflectivity;
            float logComponent = 155 + (100 * (std::log2(reflectivity - 100 + 1) / std::log2(156)));
            reflectivityColorValue = static_cast<int>((1 - transitionFactor) * linearComponent + transitionFactor * logComponent);
        } else {
            // Logarithmic mapping for values above 110
            float logReflectivity = std::log2(reflectivity - 100 + 1) / std::log2(156); // Normalized log scaling
            reflectivityColorValue = static_cast<int>(155 + logReflectivity * 100); // Maps to 155-255
        }
    }

    // Clamp to ensure the final color is in the 0-255 range
    return std::clamp(reflectivityColorValue, 0, 255);
}
