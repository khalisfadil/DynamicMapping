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

// -----------------------------------------------------------------------------
/**
 * @brief Maximum number of points allowed in a single voxel.
 * 
 * @details
 * This constant defines the upper limit on the number of points that can be stored 
 * in a single voxel. It is used to cap the memory usage per voxel and ensure consistent 
 * behavior across the occupancy map. Points exceeding this limit are ignored or aggregated 
 * into the voxel's metrics without being explicitly stored.
 */
constexpr int maxPointsPerVoxel_ = 20;

// -----------------------------------------------------------------------------
// Section: Class OccupancyMap
// -----------------------------------------------------------------------------

/**
 * @class OccupancyMap
 * @brief Manages a 3D occupancy map for voxel-based localization and mapping.
 */
class OccupancyMap {

    // -----------------------------------------------------------------------------
    // Section: public Class OccupancyMap
    // -----------------------------------------------------------------------------
    public:

        // -----------------------------------------------------------------------------
        /**
         * @brief Represents a single point's data attributes within a voxel.
         * 
         * @details
         * This structure encapsulates the attributes of a point in the occupancy map, 
         * including its 3D position and associated properties like reflectivity, intensity, 
         * and near-infrared (NIR) value. Each point contributes to the aggregated metrics 
         * of its containing voxel.
         * 
         * Members:
         * - **position**: The 3D position of the point in world coordinates (`Eigen::Vector3f`).
         * - **reflectivity**: The reflectivity value of the point (`float`).
         * - **intensity**: The intensity value of the point (`float`).
         * - **NIR**: The near-infrared value of the point (`float`).
         */
        struct PointData {
            Eigen::Vector3f position;  // Position of the point
            float reflectivity;
            float intensity;
            float NIR;
        };
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Enumerates the reasons for marking a voxel for removal from the occupancy map.
         * 
         * @details
         * This enumeration defines the possible reasons a voxel may be flagged for removal 
         * during the map's processing pipeline. It helps categorize and document why specific 
         * voxels are no longer relevant.
         * 
         * Enumerators:
         * - **None**: The voxel is not marked for removal.
         * - **Raycasting**: The voxel was flagged for removal due to raycasting operations.
         * - **MaxRangeExceeded**: The voxel is beyond the maximum reaching distance from the vehicle.
         * - **Dynamic**: The voxel is flagged as dynamic, indicating it contains moving or changing points.
         */
        enum class RemovalReason {
            None,
            Raycasting,
            MaxRangeExceeded,
            Dynamic
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Represents the data and attributes of a voxel in the occupancy map.
         * 
         * @details
         * This structure encapsulates all the information related to a single voxel, including 
         * its point data, aggregate metrics, and flags for dynamic status or removal. Each voxel 
         * is a fundamental unit in the occupancy map used for spatial representation and analysis.
         * 
         * Members:
         * - **points**: A collection of `PointData` objects representing individual points contained 
         *   within the voxel.
         * - **centerPosition**: The world coordinates of the voxel's center (`Eigen::Vector3f`).
         * - **totalReflectivity**: The accumulated reflectivity value of all points in the voxel (`float`).
         * - **totalIntensity**: The accumulated intensity value of all points in the voxel (`float`).
         * - **totalNIR**: The accumulated near-infrared (NIR) value of all points in the voxel (`float`).
         * - **avgReflectivity**: The average reflectivity of the points in the voxel (`float`).
         * - **avgIntensity**: The average intensity of the points in the voxel (`float`).
         * - **avgNIR**: The average NIR value of the points in the voxel (`float`).
         * - **lastSeenFrame**: The ID of the last frame in which this voxel was updated (`uint32_t`).
         * - **isDynamic**: A flag indicating whether the voxel contains dynamic (moving or changing) points (`bool`).
         * - **removalReason**: The reason for marking the voxel for removal, defined by the `RemovalReason` enum.
         * 
         * Default Constructor:
         * - Initializes all members to their default values.
         * - Allocates memory for the `points` vector up to `maxPointsPerVoxel_` for performance optimization.
         */
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

        // -----------------------------------------------------------------------------
        /**
         * @brief Constructs an `OccupancyMap` with specified parameters.
         * 
         * @details
         * Initializes the occupancy map with user-defined parameters for resolution, maximum 
         * reaching distance, and the center position of the map. These parameters define the 
         * spatial properties of the voxel grid and determine the map's behavior during operations.
         * 
         * @param mapRes The resolution of the voxel grid (size of each voxel in world units).
         * @param reachingDistance The maximum distance from the vehicle within which voxels 
         *        are considered relevant.
         * @param mapCenter The world coordinates of the map's center position.
         */
        OccupancyMap(float mapRes,
                        float reachingDistance,
                        Eigen::Vector3f mapCenter);

       // -----------------------------------------------------------------------------
        /**
         * @brief Executes the main pipeline for updating the occupancy map.
         * 
         * @details
         * This function processes a new frame of data by updating the map's state and applying 
         * a sequence of operations on the input point cloud and voxel grid. The pipeline consists 
         * of updating the vehicle's position, inserting point cloud data, clearing outdated voxels, 
         * and removing flagged voxels.
         * 
         * Steps:
         * 1. **Update Persistent State**: Updates the vehicle's position and the current frame ID.
         * 2. **Insert Point Cloud**: Adds new points to the map, updating voxels with point data 
         *    and attributes such as reflectivity, intensity, and near-infrared (NIR) values.
         * 3. **Mark Voxels for Clearing**: Identifies voxels to be removed based on criteria like 
         *    distance from the vehicle or dynamic updates.
         * 4. **Remove Flagged Voxels**: Deletes voxels that are marked for removal, optimizing memory usage.
         * 
         * @param pointCloud A vector of 3D points (`Eigen::Vector3f`) representing the input point cloud.
         * @param reflectivity A vector of reflectivity values corresponding to the points in the point cloud.
         * @param intensity A vector of intensity values corresponding to the points in the point cloud.
         * @param NIR A vector of near-infrared (NIR) values corresponding to the points in the point cloud.
         * @param newPosition The updated position of the vehicle in world coordinates.
         * @param newFrame The ID of the new frame being processed.
         * 
         * @return None.
         */
        void runOccupancyMapPipeline(const std::vector<Eigen::Vector3f>& pointCloud,
                                        const std::vector<float>& reflectivity,
                                        const std::vector<float>& intensity,
                                        const std::vector<float>& NIR,
                                        const Eigen::Vector3f& newPosition,
                                        uint32_t newFrame);

        // -----------------------------------------------------------------------------
        /**
         * @brief Retrieves all dynamic voxels from the occupancy map.
         * 
         * @details
         * This function identifies and returns all voxels in the occupancy map that are marked 
         * as dynamic (`isDynamic = true`). It uses parallel processing with Intel TBB for 
         * efficient iteration and data collection, leveraging `tbb::concurrent_vector` to 
         * avoid race conditions during concurrent writes.
         * 
         * Steps:
         * 1. Reserve memory in a `tbb::concurrent_vector` to collect dynamic voxels in parallel.
         * 2. Iterate over the occupancy map using `tbb::parallel_for_each` to identify dynamic voxels.
         * 3. Add dynamic voxels to the concurrent vector.
         * 4. Convert the concurrent vector to a standard vector for return.
         * 
         * @return A `std::vector<VoxelData>` containing all dynamic voxels from the occupancy map.
         */
        std::vector<VoxelData> getDynamicVoxels() const;

        // -----------------------------------------------------------------------------
        /**
         * @brief Retrieves all static voxels from the occupancy map.
         * 
         * @details
         * This function identifies and returns all voxels in the occupancy map that are not marked 
         * as dynamic (`isDynamic = false`). It employs parallel processing with Intel TBB to 
         * efficiently iterate over the map and collect static voxels using a thread-safe 
         * `tbb::concurrent_vector`.
         * 
         * Steps:
         * 1. Reserve memory in a `tbb::concurrent_vector` to collect static voxels in parallel.
         * 2. Iterate over the occupancy map using `tbb::parallel_for_each` to identify static voxels.
         * 3. Add static voxels to the concurrent vector.
         * 4. Convert the concurrent vector to a standard vector for return.
         * 
         * @return A `std::vector<VoxelData>` containing all static voxels from the occupancy map.
         */
        std::vector<VoxelData> getStaticVoxels() const;

        // -----------------------------------------------------------------------------
        /**
         * @brief Retrieves the center positions of a given set of voxels.
         * 
         * @details
         * This function extracts the center positions from a list of `VoxelData` objects and 
         * returns them as a vector of 3D points (`Eigen::Vector3f`). It preallocates memory 
         * for the output vector to optimize performance and uses `std::transform` for 
         * efficient processing of the input voxels.
         * 
         * Steps:
         * 1. Preallocate a vector of the same size as the input voxels to store the centers.
         * 2. Use `std::transform` to iterate over the input voxels and extract the `centerPosition` 
         *    from each voxel.
         * 3. Return the vector of voxel center positions.
         * 
         * @param voxels A vector of `VoxelData` objects from which to extract center positions.
         * 
         * @return A `std::vector<Eigen::Vector3f>` containing the center positions of the input voxels.
         */
        std::vector<Eigen::Vector3f> getVoxelCenters(const std::vector<VoxelData>& voxels);

        // -----------------------------------------------------------------------------
        /**
         * @brief Computes color representations for a set of voxels based on their attributes.
         * 
         * @details
         * This function calculates colors for each voxel in a given list based on its occupancy, 
         * reflectivity, intensity, and near-infrared (NIR) values. It leverages parallel processing 
         * with Intel TBB to efficiently compute colors for large datasets. The colors are returned 
         * as four separate vectors, each representing a specific characteristic.
         * 
         * Steps:
         * 1. If the input vector is empty, return four empty vectors.
         * 2. Preallocate vectors for occupancy, reflectivity, intensity, and NIR colors, ensuring 
         *    efficient memory usage.
         * 3. Use `tbb::parallel_for` to iterate over the input voxels and compute each type of color 
         *    for every voxel.
         * 4. Return the results as a tuple containing the four color vectors.
         * 
         * @param voxels A vector of `VoxelData` objects representing the input voxels.
         * 
         * @return A tuple containing four `std::vector<Eigen::Vector3i>` objects:
         * - **occupancyColors**: Grayscale colors representing voxel occupancy.
         * - **reflectivityColors**: Grayscale colors based on average reflectivity.
         * - **intensityColors**: Grayscale colors based on average intensity.
         * - **NIRColors**: Grayscale colors based on average near-infrared (NIR) values.
         */
        std::tuple<std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>, std::vector<Eigen::Vector3i>> computeVoxelColors(const std::vector<VoxelData>& voxels);

        // -----------------------------------------------------------------------------
        /**
         * @brief Assigns a red color to all voxels in the input list.
         * 
         * @details
         * This function creates a vector of colors, where each voxel in the input list is assigned 
         * a red color represented as `Eigen::Vector3i(255, 0, 0)`. The function ensures all entries 
         * in the returned vector have the same red color.
         * 
         * @param voxels A vector of `VoxelData` objects for which colors are to be assigned.
         * 
         * @return A vector of `Eigen::Vector3i`, with all entries set to the color red.
         */
        std::vector<Eigen::Vector3i> assignVoxelColorsRed(const std::vector<VoxelData>& voxels);

    // -----------------------------------------------------------------------------
    // Section: private Class OccupancyMap
    // -----------------------------------------------------------------------------
    private:
        // -----------------------------------------------------------------------------
        /**
         * @brief The resolution of the voxel grid.
         * 
         * @details
         * Specifies the size of each voxel in world units. This parameter determines the spatial 
         * granularity of the occupancy map, with smaller values providing higher precision at 
         * the cost of increased computational complexity and memory usage.
         */
        float mapRes_;

        // -----------------------------------------------------------------------------
        /**
         * @brief The maximum distance from the vehicle for relevant voxels.
         * 
         * @details
         * Defines the maximum distance from the vehicle within which voxels are considered 
         * relevant for processing. Voxels beyond this distance may be flagged for removal 
         * or ignored to optimize performance.
         */
        float reachingDistance_;

        // -----------------------------------------------------------------------------
        /**
         * @brief The center of the occupancy map in world coordinates.
         * 
         * @details
         * Represents the world position of the map's center. All voxel indices and positions 
         * are calculated relative to this center, allowing alignment with the global coordinate system.
         */
        Eigen::Vector3f mapCenter_;

        // -----------------------------------------------------------------------------
        /**
         * @brief The current position of the vehicle in world coordinates.
         * 
         * @details
         * Tracks the vehicle's position, used for dynamic voxel updates, raycasting, and 
         * determining which voxels are within the relevant processing range.
         */
        Eigen::Vector3f vehiclePosition_;

        // -----------------------------------------------------------------------------
        /**
         * @brief The ID of the current processing frame.
         * 
         * @details
         * Used to manage temporal data in the occupancy map. Tracks when voxels were 
         * last updated and helps distinguish changes between successive frames.
         */
        uint32_t currentFrame_;

        std::mutex occupancyMapMutex;

        // -----------------------------------------------------------------------------
        /**
         * @brief Custom hash function for `Eigen::Vector3i` used in hash maps.
         * 
         * This struct provides a hash function for `Eigen::Vector3i`, generating a unique hash 
         * value by combining the hashes of the vector's components (x, y, z). The components are 
         * individually hashed and combined using bitwise XOR and shifts to reduce collisions. 
         */
        struct Vector3iHash {
            std::size_t operator()(const Eigen::Vector3i& vec) const {
                std::size_t seed = 0;
                seed ^= std::hash<int>()(vec.x()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= std::hash<int>()(vec.y()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= std::hash<int>()(vec.z()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Custom equality comparator for `Eigen::Vector3i` used in hash maps.
         * 
         * This struct provides an equality comparison operator for `Eigen::Vector3i`, ensuring that 
         * two 3D integer vectors are considered equal if all their corresponding components (x, y, z) 
         * are identical.
         */
        struct Vector3iEqual {
            bool operator()(const Eigen::Vector3i& lhs, const Eigen::Vector3i& rhs) const {
                return lhs == rhs;
            }
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Hash function for a pair of `Eigen::Vector3i`.
         * 
         * This structure provides a custom hash implementation for a pair of 3D integer vectors 
         * (start and end voxel indices). The hash combines the individual hashes of both vectors 
         * to create a unique hash for the pair.
         */
        struct VoxelPairHash {
            std::size_t operator()(const std::pair<Eigen::Vector3i, Eigen::Vector3i>& pair) const {
                std::size_t hash1 = Vector3iHash{}(pair.first);
                std::size_t hash2 = Vector3iHash{}(pair.second);

                // Combine the two hashes into a single hash
                return hash1 ^ (hash2 << 1);
            }
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Equality comparator for a pair of `Eigen::Vector3i`.
         * 
         * Ensures that two pairs of 3D integer vectors are considered equal if both their start and 
         * end vectors are identical.
         */
        struct VoxelPairEqual {
            bool operator()(const std::pair<Eigen::Vector3i, Eigen::Vector3i>& a,
                            const std::pair<Eigen::Vector3i, Eigen::Vector3i>& b) const {
                return a.first == b.first && a.second == b.second;
            }
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief Hash function for `Eigen::Vector3f`.
         * 
         * Provides a custom hash implementation for a 3D floating-point vector (voxel center position).
         * Combines the individual hashes of the x, y, and z components to create a unique hash.
         */
        struct Vector3fHash {
            std::size_t operator()(const Eigen::Vector3f& vec) const {
                std::size_t seed = 0;
                seed ^= std::hash<float>()(vec.x()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= std::hash<float>()(vec.y()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= std::hash<float>()(vec.z()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        // -----------------------------------------------------------------------------
        /**
         * @brief The main data structure representing the occupancy map.
         * 
         * @details
         * This hash map stores the state of the voxel grid, where each key is a voxel grid index 
         * (`Eigen::Vector3i`) and the corresponding value is `VoxelData` containing aggregated point 
         * and attribute information for that voxel. The occupancy map supports efficient insertion, 
         * lookup, and modification of voxel data and is the central component for managing spatial 
         * information in the map.
         * 
         * The map uses a custom hash function (`Vector3iHash`) and equality comparison (`Vector3iEqual`) 
         * to handle the 3D grid indices efficiently.
         */
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> occupancyMap_;
        
        // -----------------------------------------------------------------------------
        /**
         * @brief A hash map to track voxels inserted or modified during the current frame.
         * 
         * @details
         * This data structure stores voxel grid indices (`Eigen::Vector3i`) as keys and their corresponding 
         * `VoxelData` as values. It is used to keep track of all voxels that were inserted or updated during 
         * the current frame's processing. The `insertedVoxels_` map helps optimize operations that need to 
         * work specifically on recently modified voxels, such as raycasting or dynamic clearing tasks.
         * 
         * The map uses a custom hash (`Vector3iHash`) and equality comparison (`Vector3iEqual`) for 
         * efficient key management.
         */
        tsl::robin_map<Eigen::Vector3i, VoxelData, Vector3iHash, Vector3iEqual> insertedVoxels_;
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Update the vehicle position in occupancy map.
         */
        void updateVehiclePosition(const Eigen::Vector3f& newPosition);

        // -----------------------------------------------------------------------------
        /**
         * @brief Update the current frame in occupancy map.
         */
        void updateCurrentFrame(uint32_t newFrame);

        // -----------------------------------------------------------------------------
        /**
         * @brief Converts a world position to a voxel grid index.
         * 
         * This function calculates the corresponding 3D grid index of a voxel for a given 
         * world position based on the map resolution and the map's center position.
         * 
         * @details
         * - The function scales the world position relative to the map center by the inverse 
         *   of the map resolution to determine the grid index.
         * - Uses `std::floor` to ensure that positions within a voxel are mapped to the 
         *   same grid index.
         * 
         * @param pos The world position as an `Eigen::Vector3f`.
         * 
         * @return The 3D grid index as an `Eigen::Vector3i`.
         */
        Eigen::Vector3i worldToGrid(const Eigen::Vector3f& pos) const;

        // -----------------------------------------------------------------------------
        /**
         * @brief Converts a voxel grid index to its corresponding world position.
         * 
         * This function calculates the center position of a voxel in the world coordinate system 
         * based on its grid index, the resolution of the map, and the map's center position.
         * 
         * @details
         * - The function adds half of the voxel resolution to ensure the returned position is the 
         *   center of the voxel.
         * - The grid index is converted from integer to floating-point representation for precise 
         *   calculations in the world coordinate system.
         * 
         * @param gridIndex The 3D grid index of the voxel as an `Eigen::Vector3i`.
         * 
         * @return The world position of the voxel center as an `Eigen::Vector3f`.
         */
        Eigen::Vector3f gridToWorld(const Eigen::Vector3i& gridIndex) const;

        // -----------------------------------------------------------------------------
        /**
         * @brief Updates the average attributes for a voxel based on its stored points.
         * 
         * This function calculates the average reflectivity, intensity, and near-infrared (NIR) 
         * values for a given voxel using its accumulated total values and the number of points 
         * stored within it.
         * 
         * @details
         * - The function efficiently computes averages using the inverse of the point count 
         *   to avoid multiple divisions.
         * - If the voxel contains no points, the function exits early without updating any averages.
         * 
         * @param voxel The `VoxelData` object to update. It contains aggregated totals and the 
         *              points that are used to compute the averages.
         * 
         * @return None.
         */
        void updateVoxelAverages(VoxelData& voxel) const;
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Inserts a point cloud into the occupancy map, aggregating data into voxels.
         * 
         * This function processes a point cloud and updates the occupancy map by assigning points 
         * to their respective voxels in a 3D grid. It uses parallel processing with Intel TBB 
         * to efficiently handle large point clouds and merges results into the global occupancy map.
         * 
         * @details
         * - **Parallelism:** Utilizes `tbb::parallel_reduce` to process the point cloud in parallel, 
         *   aggregating intermediate results into a local map (`localMap`) before merging them.
         * - **Voxel Aggregation:** Each voxel accumulates points, reflectivity, intensity, and NIR values 
         *   while respecting a maximum point capacity per voxel (`maxPointsPerVoxel_`).
         * - **Final Merge:** The local maps are merged into the global `occupancyMap_`, and newly 
         *   inserted voxels are tracked in `insertedVoxels_` for subsequent processing.
         * - **Efficiency Improvements:**
         *   1. Avoids redundant calculations for merging points into voxels.
         *   2. Ensures voxel averages are updated after merging.
         * 
         * @param pointCloud A vector of 3D points representing the input point cloud.
         * @param reflectivity A vector of reflectivity values corresponding to the points.
         * @param intensity A vector of intensity values corresponding to the points.
         * @param NIR A vector of near-infrared (NIR) values corresponding to the points.
         * 
         * @return None.
         */
        void insertPointCloud(const std::vector<Eigen::Vector3f>& pointCloud,
                                    const std::vector<float>& reflectivity,
                                    const std::vector<float>& intensity,
                                    const std::vector<float>& NIR);

        // -----------------------------------------------------------------------------
        /**
         * @brief Performs raycasting between two points in a voxel grid.
         * 
         * This function calculates the set of voxels intersected by a ray originating 
         * from the `start` position and terminating at the `end` position. It uses 
         * Bresenham's line algorithm to efficiently traverse voxels along the ray.
         * 
         * @details
         * - If the start and end points are very close (below a threshold), only the 
         *   voxel containing the starting point is returned.
         * - The function converts the start and end points into voxel indices and 
         *   applies Bresenham's line algorithm to determine the voxels intersected 
         *   by the ray.
         * - The result is collected using a thread-safe `tbb::concurrent_vector` and 
         *   returned as a `std::vector`.
         * 
         * @param start The starting position of the ray in 3D space.
         * @param end The ending position of the ray in 3D space.
         * @return A `std::vector<Eigen::Vector3i>` containing the voxel indices 
         *         intersected by the ray, including the start and end voxels.
         */
        std::vector<Eigen::Vector3i> performRaycast(const Eigen::Vector3f& start, const Eigen::Vector3f& end);

        // -----------------------------------------------------------------------------
        /**
         * @brief Marks voxels for clearing based on distance and raycasting.
         * 
         * This function performs two tasks in parallel:
         * 1. Identifies and marks voxels beyond the maximum reaching distance from the vehicle's position.
         * 2. Performs raycasting for newly inserted voxels to determine which voxels should be cleared 
         *    (e.g., due to occlusion or line-of-sight checks).
         * 
         * Caching is used to optimize raycasting:
         * - Per-frame `startEndCache` stores results for unique start-end voxel pairs.
         * - Per-frame `voxelCenterCache` stores results for voxel center positions.
         * 
         * @note The caches are cleared after each frame to avoid stale data and excessive memory usage.
         * @note This implementation assumes the boat is in motion, and start-end pairs change frequently.
         * 
         * @param None
         * @return None
         */
        void markVoxelsForClearing();
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Marks voxels for clearing based on various conditions, such as exceeding a maximum distance or being flagged during raycasting.
         * 
         * This function performs two parallel tasks:
         * 
         * 1. Marks voxels that are beyond the maximum reaching distance (`reachingDistance_`) from the vehicle's position (`vehiclePosition_`) by updating their `removalReason` to `MaxRangeExceeded`.
         * 2. Performs raycasting for each voxel in `insertedVoxels_` to determine which voxels are in the line of sight from the vehicle's position to the voxel's center. 
         *    Results of raycasting are cached using a per-frame `startEndCache` to avoid redundant computations.
         * 
         * @details 
         * - **Parallel Task 1**:
         *   Iterates over the `occupancyMap_` in parallel to mark voxels that exceed the reaching distance. 
         * 
         * - **Parallel Task 2**:
         *   For each voxel in `insertedVoxels_`, it performs raycasting and marks voxels along the ray's path for removal. 
         *   Raycasting results are cached using `startEndCache` to avoid recomputation for identical start-end pairs within the same frame.
         * 
         * @note 
         * - The function ensures thread safety by using `tbb::concurrent_hash_map` for caching raycasting results.
         * - Raycasting results are stored and reused only for the current frame.
         * - This function leverages Intel TBB for efficient parallel processing.
         */
        void markDynamicVoxels(const std::vector<ClusterExtractor::PointWithAttributes>& dynamicCloud);
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Removes all flagged voxels from the occupancy map.
         * 
         * This function filters out all voxels flagged for removal (i.e., those with 
         * `removalReason` set to a value other than `RemovalReason::None`). It utilizes 
         * parallel processing with thread-local maps to efficiently handle large occupancy maps.
         * 
         * Steps:
         * 1. Creates a thread-local map for each thread to avoid contention during parallel processing.
         * 2. Iterates over the existing occupancy map in parallel, filtering out flagged voxels and storing 
         *    non-flagged voxels in the corresponding thread-local map.
         * 3. Merges all thread-local maps into a single final map.
         * 4. Replaces the old map with the newly constructed map containing only non-flagged voxels.
         * 
         * @note This implementation ensures thread safety by using thread-local storage
         * and utilizes efficient merging of thread-local maps to construct the final map.
         */
        void removeFlaggedVoxels();

        // -----------------------------------------------------------------------------
        /**
         * @brief Calculates an occupancy-based grayscale color for a voxel.
         * 
         * @details
         * This function generates a grayscale color value for a voxel based on the number 
         * of points it contains. The color intensity ranges from 0 to 255, where 255 
         * corresponds to a fully occupied voxel (i.e., containing `maxPointsPerVoxel_` points). 
         * The calculated color is returned as an `Eigen::Vector3i` with identical values for 
         * the red, green, and blue components to represent grayscale.
         * 
         * @param voxel The `VoxelData` object for which the occupancy color is calculated.
         * 
         * @return A grayscale color as an `Eigen::Vector3i`, with values ranging from 0 to 255.
         */
        Eigen::Vector3i calculateOccupancyColor(const VoxelData& voxel);
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Calculates a grayscale color based on the average reflectivity of a voxel.
         * 
         * @details
         * This function maps the average reflectivity value of a voxel to a grayscale color 
         * intensity in the range of 0–255. The mapping follows a piecewise function:
         * - For reflectivity values ≤ 100.0, a linear scale is applied.
         * - For reflectivity values > 100.0, a logarithmic scale is used to emphasize higher 
         *   reflectivity values with a smoother gradient.
         * - A transition region (100.0 to 110.0) blends the linear and logarithmic scales 
         *   using a weighted combination for a seamless transition.
         * 
         * The calculated intensity is clamped between 0 and 255 to ensure valid grayscale values. 
         * The returned value is an `Eigen::Vector3i` with the same intensity applied to the 
         * red, green, and blue components.
         * 
         * @param avgReflectivity The average reflectivity value of the voxel.
         * 
         * @return A grayscale color as an `Eigen::Vector3i`, with intensity based on reflectivity.
         */
        Eigen::Vector3i calculateReflectivityColor(float avgReflectivity);
        
        // -----------------------------------------------------------------------------
        /**
         * @brief Calculates a grayscale color based on the average intensity of a voxel.
         * 
         * @details
         * This function maps the average intensity of a voxel to a grayscale color value 
         * in the range of 0–255. The intensity is clamped to ensure it falls within this 
         * range, preventing invalid values. The resulting intensity is applied equally to 
         * the red, green, and blue components to produce a grayscale color.
         * 
         * @param avgIntensity The average intensity value of the voxel.
         * 
         * @return A grayscale color as an `Eigen::Vector3i`, with intensity derived from the average intensity.
         */
        Eigen::Vector3i calculateIntensityColor(float avgIntensity);

        // -----------------------------------------------------------------------------
        /**
         * @brief Calculates a grayscale color based on the average near-infrared (NIR) value of a voxel.
         * 
         * @details
         * This function maps the average NIR value of a voxel to a grayscale color intensity 
         * in the range of 0–255. The input value is clamped to ensure it lies within this range. 
         * The resulting intensity is applied equally to the red, green, and blue components to 
         * produce a grayscale color representation.
         * 
         * @param avgNIR The average NIR value of the voxel.
         * 
         * @return A grayscale color as an `Eigen::Vector3i`, with intensity derived from the average NIR value.
         */
        Eigen::Vector3i calculateNIRColor(float avgNIR);
};