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
#include "clusterExtractor.hpp"

//##############################################################################
// Constructor
ClusterExtractor::ClusterExtractor(float clusterTolerance,
                                   uint32_t minClusterSize,
                                   uint32_t maxClusterSize,
                                   float staticThreshold,
                                   float dynamicScoreThreshold,
                                   float densityThreshold,
                                   float velocityThreshold,
                                   float similarityThreshold,
                                   float maxDistanceThreshold,
                                   double dt)
    : clusterTolerance_(clusterTolerance),
      minClusterSize_(minClusterSize),
      maxClusterSize_(maxClusterSize),
      staticThreshold_(staticThreshold),
      dynamicScoreThreshold_(dynamicScoreThreshold),
      densityThreshold_(densityThreshold),
      velocityThreshold_(velocityThreshold),
      similarityThreshold_(similarityThreshold),
      maxDistanceThreshold_(maxDistanceThreshold),
      dt_(dt),
      newClusterID(1)  // Initialize newClusterID to 1
{}
//##############################################################################
// Main pipeline  
void ClusterExtractor::runClusterExtractorPipeline(const std::vector<Eigen::Vector3f>& pointCloud,
                                                   const std::vector<float>& reflectivity,
                                                   const std::vector<float>& intensity,
                                                   const std::vector<float>& NIR) {
    // Step 1: Perform clustering and calculate properties
    extractClusters(pointCloud, reflectivity, intensity, NIR);
    calculateClusterProperties();

    // Step 2: Handle first-frame initialization
    if (prevClusterMap_.empty()) {
        initializePrevClusterMap();
        return; // Exit early, no further processing needed on first frame
    }

    // Step 3: Associate clusters with previous frames
    associateClusters();
    if (persistentAssociations_.empty()) {
        initializePrevClusterMap();
        return; // Exit early if no associations found
    }

    // Step 4: Process clusters with EKF if associations are present
    updateEKFForClusters();
    evaluateClusterDynamics();

    // Step 5: Update prevClusterMap_ with the latest frame data
    updatePrevClusterMap();
}
//##############################################################################
// Function to extract clusters using PCL's Euclidean Cluster Extraction
void ClusterExtractor::extractClusters(const std::vector<Eigen::Vector3f>& pointCloud,
                                       const std::vector<float>& reflectivity,
                                       const std::vector<float>& intensity,
                                       const std::vector<float>& NIR) {
    clusters_.clear();

    // Convert points to a PCL PointCloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.resize(pointCloud.size());

    // Use tbb::parallel_for to populate the PCL PointCloud
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, pointCloud.size()), [&](const tbb::blocked_range<uint64_t>& range) {
        for (uint64_t i = range.begin(); i < range.end(); ++i) {
            cloud->points[i].x = pointCloud[i].x();
            cloud->points[i].y = pointCloud[i].y();
            cloud->points[i].z = pointCloud[i].z();
        }
    });

    // Setup KdTree and EuclideanClusterExtraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(clusterTolerance_);
    ec.setMinClusterSize(minClusterSize_);
    ec.setMaxClusterSize(maxClusterSize_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);

    std::vector<pcl::PointIndices> clusterIndices;
    ec.extract(clusterIndices);

    // Convert clustered indices back to clusters with attributes in parallel
    clusters_.resize(clusterIndices.size());
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, clusterIndices.size()), [&](const tbb::blocked_range<uint64_t>& range) {
        for (uint64_t i = range.begin(); i < range.end(); ++i) {
            const auto& indices = clusterIndices[i];
            std::vector<PointWithAttributes>& cluster = clusters_[i];
            cluster.reserve(indices.indices.size());
            for (int index : indices.indices) {
                Eigen::Vector3f position(cloud->points[index].x, cloud->points[index].y, cloud->points[index].z);
                PointWithAttributes pointWithAttr = {
                    position,
                    reflectivity[index],  // Use the corresponding reflectivity value
                    intensity[index],     // Use the corresponding intensity value
                    NIR[index]            // Use the corresponding NIR value
                };
                cluster.push_back(pointWithAttr);
            }
        }
    });
}
//##############################################################################
// Function to calculate properties of each cluster (centroid, bounding box, etc.)
void ClusterExtractor::calculateClusterProperties() {
    // Resize propertiesList_ to match the number of clusters, allowing direct index access
    propertiesList_.resize(clusters_.size());

    // Use TBB to parallelize the calculation for each cluster
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, clusters_.size()), [&](const tbb::blocked_range<uint64_t>& range) {
        for (uint64_t i = range.begin(); i < range.end(); ++i) {
            const auto& cluster = clusters_[i];

            // Check for empty clusters
            if (cluster.empty()) {
                continue;
            }

            ClusterProperties properties;

            properties.data = cluster; // include all point inside the properties.

            Eigen::Vector3f sum(0, 0, 0);
            Eigen::Vector3f minBound = cluster[0].position;
            Eigen::Vector3f maxBound = cluster[0].position;

            float totalReflectivity = 0.0f;
            float totalIntensity = 0.0f;
            float totalNIR = 0.0f;

            for (const auto& point : cluster) {
                // Update centroid sum and bounding box
                sum += point.position;
                minBound = minBound.cwiseMin(point.position);
                maxBound = maxBound.cwiseMax(point.position);

                // Accumulate attribute values
                totalReflectivity += point.reflectivity;
                totalIntensity += point.intensity;
                totalNIR += point.NIR;
            }
            // Calculate final properties for this cluster
            properties.centroid = sum / cluster.size();
            properties.boundingBoxMin = minBound;
            properties.boundingBoxMax = maxBound;
            properties.pointCount = cluster.size();
            // Calculate averages for reflectivity, intensity, and NIR
            properties.avgReflectivity = totalReflectivity / cluster.size();
            properties.avgIntensity = totalIntensity / cluster.size();
            properties.avgNIR = totalNIR / cluster.size();
            // Calculate the bounding box volume for 3D (or area for 2D)
            float volume = (maxBound.x() - minBound.x()) *
                            (maxBound.y() - minBound.y()) *
                            (maxBound.z() - minBound.z());
            
            // Ensure the volume is positive to avoid division by zero
            volume = std::max(volume, 1e-6f);
            // Calculate density as the ratio of point count to volume
            properties.density = cluster.size() / volume;
            // Directly assign computed properties to propertiesList_ at index i
            propertiesList_[i] = std::move(properties);
        }
    });