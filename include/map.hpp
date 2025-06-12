#pragma once

#include <queue>
#include <Eigen/Dense>
#include <robin_map.h>
#include <point.hpp>
#include <voxel.hpp>

namespace dynamicMap {

    class Map {

        public:

            // -----------------------------------------------------------------------------

            Map() = default;

            // -----------------------------------------------------------------------------

            explicit Map(int default_lifetime = 10) : default_lifetime_(default_lifetime) {}

            // -----------------------------------------------------------------------------

            [[nodiscard]] ArrayVector3d pointcloud() const {
                ArrayVector3d points;
                points.reserve(total_num_points_); // Use tracked total number of points
                for (const auto& pair : voxel_map_) {
                    const auto& block = pair.second;
                    points.insert(points.end(), block.points.begin(), block.points.end());
                }
                return points;
            }

            // -----------------------------------------------------------------------------

            [[nodiscard]] size_t size() const {
                return total_num_points_;
            }

            // -----------------------------------------------------------------------------
            
            // This method removes voxels where the *first point* in the voxel is *further* than 'distance'
            // from 'location'. Effectively, it keeps voxels within a certain radius.
            void removeOutliers(const Eigen::Vector3d& location, double distance) {
                std::vector<Voxel> voxels_to_erase;
                voxels_to_erase.reserve(voxel_map_.size() / 10); // Heuristic reservation
                const double sq_distance = distance * distance;
                for (const auto& pair : voxel_map_) {
                    const auto& voxel = pair.first;
                    const auto& block = pair.second;
                    if (!block.points.empty() && (block.points[0] - location).squaredNorm() > sq_distance) {
                        voxels_to_erase.push_back(voxel);
                    }
                }
                for (const auto& voxel : voxels_to_erase) {
                    auto it = voxel_map_.find(voxel);
                    if (it != voxel_map_.end()) {
                        total_num_points_ -= it->second.NumPoints();
                        voxel_map_.erase(it);
                    }
                }
            }

            // -----------------------------------------------------------------------------

            void update_and_filter_lifetimes() {
                std::vector<Voxel> voxels_to_erase;
                // Reserve based on a heuristic, e.g. 10% of map or a fixed typical number
                voxels_to_erase.reserve(voxel_map_.size() / 10 + 1); 
                for (auto it = voxel_map_.begin(); it != voxel_map_.end(); ++it) {
                    auto& voxel_block = it.value();
                    voxel_block.life_time -= 1;
                    if (voxel_block.life_time <= 0) voxels_to_erase.push_back(it->first);
                }
                for (const auto &vox_key : voxels_to_erase) {
                    auto it = voxel_map_.find(vox_key);
                    if (it != voxel_map_.end()) {
                        total_num_points_ -= it->second.NumPoints();
                        voxel_map_.erase(it);
                    }
                }
            }

            // -----------------------------------------------------------------------------

            void setDefaultLifeTime(int16_t default_lifetime) {
                default_lifetime_ = default_lifetime; 
            }

            // -----------------------------------------------------------------------------

            void clear() { 
                voxel_map_.clear();
                total_num_points_ = 0; 
            }

            // -----------------------------------------------------------------------------

            void add(const std::vector<Points3D>& points, double voxel_size, int max_num_points_in_voxel,
                    double min_distance_points, int min_num_points = 0) {
                for (const auto& point : points) {
                    add(point.pt, voxel_size, max_num_points_in_voxel, min_distance_points, min_num_points);
                }
            }

            // -----------------------------------------------------------------------------

            void add(const ArrayVector3d& points, double voxel_size, int max_num_points_in_voxel, 
                    double min_distance_points) {
                for (const auto& point : points) {
                    add(point, voxel_size, max_num_points_in_voxel, min_distance_points);
                }
            }

            // -----------------------------------------------------------------------------

            void add(const Eigen::Vector3d &point, double voxel_size, int max_num_points_in_voxel, double min_distance_points, int min_num_points = 0) {
                Voxel voxel_key = Voxel::Coordinates(point, voxel_size);
                auto it = voxel_map_.find(voxel_key);
                
                if (it != voxel_map_.end()) {
                    auto &voxel_block = it.value();

                    if (!voxel_block.IsFull()) {
                        double sq_dist_min_to_points = std::numeric_limits<double>::max();

                        for (const auto& existing_point : voxel_block.points) {
                            double sq_dist = (existing_point - point).squaredNorm();
                            if (sq_dist < sq_dist_min_to_points) {
                                sq_dist_min_to_points = sq_dist;
                            }
                        }

                        if (sq_dist_min_to_points > (min_distance_points * min_distance_points)) {
                            // Add point if min_num_points requirement is met OR not applicable
                            if (min_num_points <= 0 || voxel_block.NumPoints() >= min_num_points) {
                                if (voxel_block.AddPoint(point)) {
                                    total_num_points_++;
                                }
                            }
                        }
                    }
                    voxel_block.life_time = default_lifetime_;
                } else {
                    // Always create a new voxel with the first point.
                    // The min_num_points parameter applies to adding more points to an existing voxel.
                    VoxelBlock block(max_num_points_in_voxel);
                    if (block.AddPoint(point)) { // Should always be true for a new block
                        total_num_points_++;
                        block.life_time = default_lifetime_;
                        voxel_map_.emplace(voxel_key, std::move(block));
                    }
                }
            }

            // -----------------------------------------------------------------------------

            using pair_distance_t = std::tuple<double, Eigen::Vector3d, Voxel>;

            // -----------------------------------------------------------------------------

            struct Comparator {
                bool operator()(const pair_distance_t& left, const pair_distance_t& right) const {
                    return std::get<0>(left) > std::get<0>(right); // Min-heap
                }
            };

            // -----------------------------------------------------------------------------

            using priority_queue_t = std::priority_queue<pair_distance_t, std::vector<pair_distance_t>, Comparator>;

            // -----------------------------------------------------------------------------

            ArrayVector3d searchNeighbors(const Eigen::Vector3d& point, int nb_voxels_visited, double size_voxel_map,
                                            int max_num_neighbors, int threshold_voxel_capacity = 1, std::vector<Voxel>* voxels = nullptr) {
                // Reserve space for output
                if (voxels) voxels->reserve(max_num_neighbors);

                // Compute center voxel coordinates
                const Voxel center = Voxel::Coordinates(point, size_voxel_map);
                const int16_t kx = center.x;
                const int16_t ky = center.y;
                const int16_t kz = center.z;

                // Initialize min-heap for closest points
                priority_queue_t priority_queue;

                // Track max distance for pruning
                double max_distance = std::numeric_limits<double>::max();

                // Spiral traversal: process voxels layer by layer
                // The loop structure inherently keeps voxels within the desired cubic radius.
                for (int16_t d = 0; d <= nb_voxels_visited; ++d) {
                    for (int16_t dx = -d; dx <= d; ++dx) {
                        for (int16_t dy = -d; dy <= d; ++dy) {
                            for (int16_t dz = -d; dz <= d; ++dz) {
                                // Only process boundary voxels at distance d
                                if (std::abs(dx) != d && std::abs(dy) != d && std::abs(dz) != d) continue;

                                Voxel voxel{kx + dx, ky + dy, kz + dz};

                                // Early pruning: skip voxels too far away
                                Eigen::Vector3d voxel_center(
                                    voxel.x * size_voxel_map + size_voxel_map / 2.0,
                                    voxel.y * size_voxel_map + size_voxel_map / 2.0,
                                    voxel.z * size_voxel_map + size_voxel_map / 2.0
                                );
                                
                                if ((voxel_center - point).norm() > max_distance + size_voxel_map) continue;

                                // Look up voxel in map
                                auto search = voxel_map_.find(voxel);
                                if (search == voxel_map_.end()) continue;

                                const auto& voxel_block = search->second;
                                if (voxel_block.NumPoints() < threshold_voxel_capacity) continue;

                                // Process points in voxel
                                for (const auto& neighbor : voxel_block.points) {
                                    double distance = (neighbor - point).norm();
                                    if (priority_queue.size() < static_cast<size_t>(max_num_neighbors)) {
                                        priority_queue.emplace(distance, neighbor, voxel);
                                        if (priority_queue.size() == static_cast<size_t>(max_num_neighbors)) {
                                            max_distance = std::get<0>(priority_queue.top());
                                        }
                                    } else if (distance < max_distance) {
                                        priority_queue.pop();
                                        priority_queue.emplace(distance, neighbor, voxel);
                                        max_distance = std::get<0>(priority_queue.top());
                                    }
                                }
                            }
                        }
                    }
                }

                // Extract results
                const auto size = priority_queue.size();
                ArrayVector3d closest_neighbors(size);
                if (voxels) voxels->resize(size);

                for (size_t i = size; i > 0; --i) {
                    closest_neighbors[i - 1] = std::get<1>(priority_queue.top());
                    if (voxels) (*voxels)[i - 1] = std::get<2>(priority_queue.top());
                    priority_queue.pop();
                }

                return closest_neighbors;
            }

            // -----------------------------------------------------------------------------

            void raycast(const Eigen::Vector3d& start_point, const Eigen::Vector3d& end_point, double voxel_size) {
                // Early exit for invalid voxel size
                if (voxel_size <= 0.0) return;

                constexpr double close_threshold_sq = 1e-6; // Compile-time constant

                // Convert points to voxel coordinates
                Voxel start_key = Voxel::Coordinates(start_point, voxel_size);
                Voxel end_key = Voxel::Coordinates(end_point, voxel_size);

                // Early exit for close points or same voxel
                if ((end_point - start_point).squaredNorm() < close_threshold_sq || start_key == end_key) {
                    voxel_map_.erase(start_key); // Erase start voxel
                    return;
                }

                // Bresenham setup
                Eigen::Vector3i current_voxel(start_key.x, start_key.y, start_key.z);
                Eigen::Vector3i end_voxel(end_key.x, end_key.y, end_key.z);
                Eigen::Vector3i delta = (end_voxel - current_voxel).cwiseAbs();
                Eigen::Vector3i step = (end_voxel - current_voxel).cwiseSign();

                // Select primary axis
                int primary_axis = delta[0] >= delta[1] && delta[0] >= delta[2] ? 0 :
                                delta[1] >= delta[2] ? 1 : 2;
                int secondary_axis = (primary_axis + 1) % 3;
                int tertiary_axis = (primary_axis + 2) % 3;

                // Precompute Bresenham constants
                int delta_primary = delta[primary_axis];
                int delta_secondary = delta[secondary_axis];
                int delta_tertiary = delta[tertiary_axis];
                int delta_primary_2 = delta_primary << 1;
                int delta_secondary_2 = delta_secondary << 1;
                int delta_tertiary_2 = delta_tertiary << 1;

                // Initialize error terms
                int error1 = delta_secondary_2 - delta_primary;
                int error2 = delta_tertiary_2 - delta_primary;

                // Reserve vector size
                std::vector<Voxel> voxel_indices;
                voxel_indices.reserve(static_cast<size_t>(delta_primary + 1));

                // Add start voxel
                voxel_indices.push_back(start_key);

                // Bresenham loop until just before end voxel
                while (current_voxel != end_voxel) {
                    current_voxel[primary_axis] += step[primary_axis];
                    if (error1 > 0) {
                        current_voxel[secondary_axis] += step[secondary_axis];
                        error1 -= delta_primary_2;
                    }
                    if (error2 > 0) {
                        current_voxel[tertiary_axis] += step[tertiary_axis];
                        error2 -= delta_primary_2;
                    }
                    error1 += delta_secondary_2;
                    error2 += delta_tertiary_2;

                    Voxel current_key{
                        static_cast<int16_t>(current_voxel[0]),
                        static_cast<int16_t>(current_voxel[1]),
                        static_cast<int16_t>(current_voxel[2])
                    };

                    // Add voxel only if it's not the end voxel and not a duplicate
                    if (current_key != end_key && (voxel_indices.empty() || current_key != voxel_indices.back())) {
                        voxel_indices.push_back(current_key);
                    } else if (current_key == end_key) {
                        break; // Stop before adding end voxel
                    }
                }

                // Remove all collected voxels from voxel_map_
                for (const auto& voxel : voxel_indices) {
                    auto it = voxel_map_.find(voxel);
                    if (it != voxel_map_.end()) {
                        total_num_points_ -= it->second.NumPoints();
                        voxel_map_.erase(it);
                    }
                }
            }

            // -----------------------------------------------------------------------------

            void removeIsolatedVoxels(int neighbor_search_radius = 1) {
                if (voxel_map_.empty() || neighbor_search_radius < 0) {
                    return;
                }

                std::vector<Voxel> voxels_to_erase;
                // Reserve conservatively, actual number of isolated voxels might be small
                voxels_to_erase.reserve(voxel_map_.size() / 20 + 1); 

                for (const auto& pair : voxel_map_) {
                    const Voxel& current_voxel_key = pair.first;
                    bool has_neighbor = false;

                    for (int16_t dx = -neighbor_search_radius; dx <= neighbor_search_radius; ++dx) {
                        for (int16_t dy = -neighbor_search_radius; dy <= neighbor_search_radius; ++dy) {
                            for (int16_t dz = -neighbor_search_radius; dz <= neighbor_search_radius; ++dz) {
                                if (dx == 0 && dy == 0 && dz == 0) {
                                    continue; // Skip the voxel itself
                                }
                                Voxel neighbor_key(current_voxel_key.x + dx, 
                                                   current_voxel_key.y + dy, 
                                                   current_voxel_key.z + dz);
                                if (voxel_map_.count(neighbor_key)) {
                                    has_neighbor = true;
                                    goto next_voxel_check; // Found a neighbor, move to the next voxel
                                }
                            }
                        }
                    }
                    next_voxel_check:;
                    if (!has_neighbor) {
                        voxels_to_erase.push_back(current_voxel_key);
                    }
                }

                for (const auto& voxel_key : voxels_to_erase) {
                    auto it = voxel_map_.find(voxel_key);
                    if (it != voxel_map_.end()) {
                        total_num_points_ -= it->second.NumPoints();
                        voxel_map_.erase(it);
                    }
                }
            }

        private:
            VoxelHashMap voxel_map_;
            int default_lifetime_ = 10;
            size_t total_num_points_ = 0;
    };

} // namespace dynamicMap