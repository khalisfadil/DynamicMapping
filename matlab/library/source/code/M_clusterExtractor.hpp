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

#include "M_EKFVelocity2D.hpp"

#include <cstdint>
#include <deque>
#include <algorithm>
#include <cstddef>

#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>

//##############################################################################
// ClusterExtractor class to manage cluster extraction, tracking, and association
class ClusterExtractor {
public:
    //##############################################################################
    // Constructor with one-time parameters
    ClusterExtractor(double clusterTolerance,
                     int minClusterSize,
                     int maxClusterSize,
                     double staticThreshold,
                     double dynamicScoreThreshold,
                     double densityThreshold,
                     double velocityThreshold,
                     double similarityThreshold,
                     double maxDistanceThreshold,
                     double dt);
    //##############################################################################
    // Structure to hold point information within each cluster
    struct PointWithAttributes {
        Eigen::Vector3d position;float reflectivity;float intensity;float NIR;
    };
    //##############################################################################
    // Structure to store properties of each cluster
    struct ClusterProperties {
        int clusterID = -1; 
        std::vector<PointWithAttributes> data;
        Eigen::Vector3d centroid; 
        Eigen::Vector3d boundingBoxMin; 
        Eigen::Vector3d boundingBoxMax;
        Eigen::Vector3d velocity; 
        double density; 
        float avgReflectivity; 
        float avgIntensity;
        float avgNIR; 
        int pointCount; 
        bool isDynamic = false;
        double dynamicScore = 0.0;                          // Overall dynamic confidence score
        double velocityConsistencyScore = 0.0;              // Score from EKF velocity prediction consistency
    };
    //##############################################################################
    // Main Pipeline
    void runClusterExtractorPipeline(const std::vector<Eigen::Vector3d>& pointCloud,
                                        const std::vector<float>& reflectivity,
                                        const std::vector<float>& intensity,
                                        const std::vector<float>& NIR);
    //##############################################################################
    // Function getDynamicClusters
    std::vector<ClusterProperties> getDynamicClusters() const;
    //##############################################################################
    // Function getStaticClusters
    std::vector<ClusterProperties> getStaticClusters() const;
    //##############################################################################
    // Function to retrieve points from all dynamic clusters
    std::vector<PointWithAttributes> getDynamicClusterPoints() const;
private:
    //##############################################################################
    // Persistent member variables (one-time defined parameters)
    double clusterTolerance_;
    int minClusterSize_;
    int maxClusterSize_;
    double staticThreshold_;
    double dynamicScoreThreshold_;
    double densityThreshold_;
    double velocityThreshold_;
    double similarityThreshold_;
    double maxDistanceThreshold_;
    double dt_;
    int newClusterID;

    //##############################################################################
    // Persistent previous clusters
    std::deque<tsl::robin_map<int, ClusterProperties>> prevClusterMap_;
    //##############################################################################
    // Persistent ekf clusters
    tsl::robin_map<int, std::unique_ptr<EKFVelocity2D>> ekfInstances_;
    //##############################################################################
    // Stored current extracted clusters of points
    std::vector<std::vector<PointWithAttributes>> clusters_;
    //##############################################################################
    // Stored current property list based on current cluster
    std::vector<ClusterProperties> propertiesList_;
    //##############################################################################
    // Persistent associations between previous and current ClusterProperties
    std::vector<std::tuple<ClusterProperties*, ClusterProperties*, int>> persistentAssociations_;
    //##############################################################################
    // Extract cluster
    void extractClusters(const std::vector<Eigen::Vector3d>& pointCloud,
                                       const std::vector<float>& reflectivity,
                                       const std::vector<float>& intensity,
                                       const std::vector<float>& NIR);
    //##############################################################################
    // Calculate properties for the given cluster
    void calculateClusterProperties();
    //##############################################################################
    // Calculate bounding box consistency score between two clusters
    double calculateBoundingBoxScore(const ClusterProperties& clusterA,
                                     const ClusterProperties& clusterB) const;
    //##############################################################################
    // Calculate similarity score between two clusters
    double calculateSimilarityScore(const ClusterProperties& clusterA,
                                    const ClusterProperties& clusterB) const;
    //##############################################################################
    // Function to associate clusters across frames
    void associateClusters();
    //##############################################################################
    // Function to updateEKFForClusters
    void updateEKFForClusters();
    //##############################################################################
    // Function to evaluate dynamic
    void evaluateClusterDynamics();
    //##############################################################################
    // Function to updatePrevClusterMap
    void updatePrevClusterMap();
    //##############################################################################
    // Function to initialize prevClusterMap_ with propertieslist
    void initializePrevClusterMap();
};

