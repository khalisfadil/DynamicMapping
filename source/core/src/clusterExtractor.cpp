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

    // // Step 2: Handle first-frame initialization
    // if (prevClusterMap_.empty()) {
    //     initializePrevClusterMap();
    //     std::cout << "Initialize cluster because prevClusterMap is empty \n";
    //     return; // Exit early, no further processing needed on first frame
    // }

    // // Step 3: Associate clusters with previous frames
    // associateClusters();
    // if (persistentAssociations_.empty()) {
    //     initializePrevClusterMap();
    //     std::cout << "Initialize cluster because associateClusters is empty \n";
    //     return; // Exit early if no associations found
    // }

    // // Step 4: Process clusters with EKF if associations are present
    // updateEKFForClusters();
    // evaluateClusterDynamics();

    // // Step 5: Update prevClusterMap_ with the latest frame data
    // updatePrevClusterMap();
    // std::cout << "Function ClusterExtractor running okay.\n";
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
}
//##############################################################################
// Function to calculate bounding box overlap score
float ClusterExtractor::calculateBoundingBoxScore(const ClusterProperties& clusterA, const ClusterProperties& clusterB) const {
    Eigen::Vector3f overlapMin = clusterA.boundingBoxMin.cwiseMax(clusterB.boundingBoxMin);
    Eigen::Vector3f overlapMax = clusterA.boundingBoxMax.cwiseMin(clusterB.boundingBoxMax);
    Eigen::Vector3f overlapDimensions = (overlapMax - overlapMin).cwiseMax(0);

    float overlapVolume = overlapDimensions.prod();
    float minVolume = std::min((clusterA.boundingBoxMax - clusterA.boundingBoxMin).prod(),
                                (clusterB.boundingBoxMax - clusterB.boundingBoxMin).prod());

    // Return normalized overlap volume if minVolume > 0, else return 0
    return (minVolume > 0) ? (overlapVolume / minVolume) : 0.0f;
}
//##############################################################################
// Function to calculate similarity score based on position, bounding box, etc.
float ClusterExtractor::calculateSimilarityScore(const ClusterProperties& clusterA, const ClusterProperties& clusterB) const {
    float score = 0.0f;

    // Weights for each component
    const float distanceWeight = 0.4f;
    const float bboxWeight = 0.3f;
    const float reflectivityWeight = 0.1f;
    const float intensityWeight = 0.1f;
    const float nirWeight = 0.1f;

    // Centroid distance similarity
    float distance = (clusterA.centroid - clusterB.centroid).norm();
    score += distanceWeight * std::exp(-distance);

    // Bounding box similarity
    score += bboxWeight * calculateBoundingBoxScore(clusterA, clusterB);

    // Reflectivity similarity
    float reflectivityDifference = std::abs(clusterA.avgReflectivity - clusterB.avgReflectivity);
    score += reflectivityWeight * std::exp(-reflectivityDifference);

    // Intensity similarity
    float intensityDifference = std::abs(clusterA.avgIntensity - clusterB.avgIntensity);
    score += intensityWeight * std::exp(-intensityDifference);

    // NIR similarity
    float NIRDifference = std::abs(clusterA.avgNIR - clusterB.avgNIR);
    score += nirWeight * std::exp(-NIRDifference);

    return score;
}
//##############################################################################
// Function to associate clusters across frames
void ClusterExtractor::associateClusters() {
    if (propertiesList_.empty() || prevClusterMap_.empty()) {
        return;  // No clusters to associate
    }

    tbb::concurrent_bounded_queue<std::tuple<float, int, int>> scoreQueue;

    // Clear persistent associations for the current frame
    persistentAssociations_.clear();

    // Step 2: Parallel loop for computing similarity scores
    tbb::parallel_for(tbb::blocked_range<uint64_t>(0, propertiesList_.size()), [&](const tbb::blocked_range<uint64_t>& range) {
        for (uint64_t i = range.begin(); i < range.end(); ++i) {
            auto& currCluster = propertiesList_[i];

            // Compare with clusters from the previous 5 frames
            for (auto& frameMap : prevClusterMap_) {
                for (auto& [prevID, prevCluster] : frameMap) {
                    float distance = (currCluster.centroid - prevCluster.centroid).norm();
                    if (distance > maxDistanceThreshold_) continue;

                    float similarityScore = calculateSimilarityScore(currCluster, prevCluster);

                    if (similarityScore > similarityThreshold_) {
                        scoreQueue.push(std::make_tuple(similarityScore, prevID, static_cast<int>(i)));
                    }
                }
            }
        }
    });

    // Step 3: Match clusters based on similarity scores
    tbb::concurrent_unordered_map<int, bool> matchedPrevClusters;
    tbb::concurrent_unordered_map<int, bool> matchedCurrClusters;

    std::tuple<float, int, int> item;
    while (scoreQueue.try_pop(item)) {
        auto [similarityScore, prevID, currIdx] = item;

        if (!matchedPrevClusters[prevID] && !matchedCurrClusters[currIdx]) {
            matchedPrevClusters[prevID] = true;
            matchedCurrClusters[currIdx] = true;

            // Update the clusterID directly in propertiesList_
            propertiesList_[currIdx].clusterID = prevID;

            // Search for prevID starting from the most recent frame in prevClusterMap_
            int frameDiff = 1;  // Start frameDiff at 1
            for (auto frameIt = prevClusterMap_.rbegin(); frameIt != prevClusterMap_.rend(); ++frameIt, ++frameDiff) {
                auto& frameMap = *frameIt;
                if (frameMap.count(prevID) > 0) {
                    // Add the latest matching cluster and frame difference to persistentAssociations_
                    persistentAssociations_.emplace_back(&frameMap[prevID], &propertiesList_[currIdx], frameDiff);
                    break;  // Stop after finding the latest match
                }
            }
        }
    }

    // Step 4: Assign new IDs to unmatched clusters
    for (uint64_t i = 0; i < propertiesList_.size(); ++i) {
        if (!matchedCurrClusters[static_cast<int>(i)]) {
            propertiesList_[i].clusterID = newClusterID++;  // Increment persistent ID and assign to unmatched clusters
        }
    }
}
//##############################################################################
// EKF update function
void ClusterExtractor::updateEKFForClusters() {
    // Iterate over each association in persistentAssociations_
    for (auto& [prevClusterPtr, currClusterPtr, frameDiff] : persistentAssociations_) {
        // Ensure both previous and current clusters are valid
        if (prevClusterPtr && currClusterPtr) {
            int clusterID = prevClusterPtr->clusterID;  // Retrieve cluster ID
            double effectiveDt = frameDiff * dt_;      // Calculate effective dt

            // Check if an EKF instance exists for this clusterID, or create one if it doesn't
            auto it = ekfInstances_.find(clusterID);
            if (it == ekfInstances_.end()) {
                it = ekfInstances_.emplace(clusterID, std::make_unique<EKFVelocity2D>(prevClusterPtr->centroid.head<2>())).first;
            }

            // Retrieve a mutable reference to the EKF instance via the unique_ptr
            auto& ekf = *(it->second);

            // Perform the predict step with the calculated dt
            ekf.predict(effectiveDt);

            // Prepare the position measurement from the current cluster's centroid [x, y]
            Eigen::Vector2f positionMeasurement = currClusterPtr->centroid.head<2>();

            // Perform the update step with the current position measurement
            ekf.update(positionMeasurement);

            // Assign the updated EKF state back to the current cluster's velocity
            Eigen::Vector2f predictedVelocity2D = ekf.getPredictedVelocity();
            currClusterPtr->velocity << predictedVelocity2D.x(), predictedVelocity2D.y(), 0.0f;
        }
    }
}
//##############################################################################
// evaluateClusterDynamics
void ClusterExtractor::evaluateClusterDynamics() {
    // Iterate over each association in persistentAssociations_
    for (auto& [prevClusterPtr, currClusterPtr, frameDiff] : persistentAssociations_) {
        // Ensure both previous and current clusters are valid
        if (prevClusterPtr && currClusterPtr) {
            int clusterID = currClusterPtr->clusterID;  // Retrieve the current cluster ID

            // Check if an EKF instance exists for this clusterID
            auto it = ekfInstances_.find(clusterID);
            if (it != ekfInstances_.end()) {
                // Retrieve a mutable reference to the EKF instance via the unique_ptr
                auto& ekf = *(it->second);

                // Step 1: Check detected velocity from EKF to detect significant movement
                float detectedSpeed = ekf.getPredictedVelocity().norm();
                if (detectedSpeed > 1.5) {
                    currClusterPtr->dynamicScore = 1.0;  // Full confidence in dynamic behavior
                    currClusterPtr->isDynamic = true;
                    continue;  // Skip further checks for this cluster
                }

                // Step 2: Calculate dynamic score based on centroid, bounding box, density, velocity consistency, and previous dynamic state

                // 1. Centroid movement score (high if centroids are far apart)
                float distance = (currClusterPtr->centroid - prevClusterPtr->centroid).norm();
                float centroidScore = 1.0f - std::exp(-distance / staticThreshold_);

                // 2. Bounding box consistency score (high if bounding boxes are inconsistent)
                float boundingBoxScore = 1.0f - calculateBoundingBoxScore(*currClusterPtr, *prevClusterPtr);

                // 3. Density change score (high if density changes significantly)
                float densityChange = std::abs(currClusterPtr->density - prevClusterPtr->density);
                float densityScore = 1.0f - std::exp(-densityChange / densityThreshold_);

                // 4. Velocity consistency score from EKF
                float velocityError = ekf.getStateVelocityError();
                currClusterPtr->velocityConsistencyScore = 1.0f - std::exp(-velocityError / velocityThreshold_);

                // 5. Previous dynamic state contribution
                float prevDynamicScore = prevClusterPtr->dynamicScore; // Confidence score from the previous cluster
                float prevDynamicWeight = 0.2f; // Adjust this weight as necessary

                // Calculate the final dynamic score, weighted by each factor
                currClusterPtr->dynamicScore = (centroidScore * 0.25f +
                                                boundingBoxScore * 0.2f +
                                                densityScore * 0.1f +
                                                currClusterPtr->velocityConsistencyScore * 0.25f +
                                                prevDynamicScore * prevDynamicWeight);

                // Set isDynamic based on dynamic score threshold
                currClusterPtr->isDynamic = currClusterPtr->dynamicScore > 0.5f;
            } else {
                // Default to dynamic if no EKF instance exists for this clusterID
                currClusterPtr->isDynamic = false;
                currClusterPtr->dynamicScore = 0.0f;  // Full confidence in dynamic behavior
            }
        } else {
            // Default to dynamic if previous cluster is missing
            currClusterPtr->isDynamic = false;
            currClusterPtr->dynamicScore = 0.0f;  // Full confidence in dynamic behavior
        }
    }
}
//##############################################################################
// update function updatePrevClusterMap
void ClusterExtractor::updatePrevClusterMap() {
    // Step 1: Remove clusters from propertiesList_ that have the same clusterID as in currClusterPtr
    for (const auto& [prevClusterPtr, currClusterPtr, frameDiff] : persistentAssociations_) {
        if (currClusterPtr) {  // Ensure currClusterPtr is valid
            // Remove any cluster from propertiesList_ that has the same clusterID as currClusterPtr
            propertiesList_.erase(std::remove_if(propertiesList_.begin(), propertiesList_.end(),
                [&currClusterPtr](const ClusterProperties& properties) {
                    return properties.clusterID == currClusterPtr->clusterID;
                }), propertiesList_.end());
        }
    }

    // Step 2: Add each currClusterPtr from persistentAssociations_ to propertiesList_
    for (const auto& [prevClusterPtr, currClusterPtr, frameDiff] : persistentAssociations_) {
        if (currClusterPtr) {
            propertiesList_.push_back(*currClusterPtr);  // Add updated cluster to propertiesList_
        }
    }

    // Step 3: Create a new map for the current frame using the updated propertiesList_
    tsl::robin_map<int, ClusterProperties> currentFrameMap;
    for (const auto& cluster : propertiesList_) {
        currentFrameMap.emplace(cluster.clusterID, cluster);  // Add each updated cluster to the frame map
    }

    // Step 4: Maintain a maximum of 5 frames in prevClusterMap_
    if (prevClusterMap_.size() == 5) {
        prevClusterMap_.pop_front();  // Remove the oldest frame if deque has reached its maximum size
    }

    // Step 5: Push the updated map for the current frame to the back of the deque
    prevClusterMap_.push_back(std::move(currentFrameMap));
}
//##############################################################################
// Function to initializePrevClusterMap
void ClusterExtractor::initializePrevClusterMap() {
    // Check if the deque has reached the maximum size (5 frames)
    if (prevClusterMap_.size() == 5) {
        prevClusterMap_.pop_front();  // Remove the oldest frame
    }

    // Insert the current frame’s clusters into prevClusterMap_
    tsl::robin_map<int, ClusterProperties> currentFrameMap;
    for (ClusterProperties& cluster : propertiesList_) {
        currentFrameMap.emplace(cluster.clusterID, cluster);  // Add the current cluster to the frame map
    }

    // Push the updated map for the current frame to the back of the deque
    prevClusterMap_.push_back(std::move(currentFrameMap));
}
//##############################################################################
// Function to retrieve dynamic clusters from propertiesList_
std::vector<ClusterExtractor::ClusterProperties> ClusterExtractor::getDynamicClusters() const {
    std::vector<ClusterProperties> dynamicClusters;

    for (const auto& cluster : propertiesList_) {
        if (cluster.isDynamic) {
            dynamicClusters.push_back(cluster);  // Add dynamic cluster to the result list
        }
    }

    return dynamicClusters;  // Return the list of dynamic clusters
}
//##############################################################################
// Function to retrieve static clusters from propertiesList_
std::vector<ClusterExtractor::ClusterProperties> ClusterExtractor::getStaticClusters() const {
    std::vector<ClusterProperties> staticClusters;

    for (const auto& cluster : propertiesList_) {
        if (!cluster.isDynamic) {
            staticClusters.push_back(cluster);  // Add static cluster to the result list
        }
    }

    return staticClusters;  // Return the list of static clusters
}
//##############################################################################
// Function to retrieve points from all dynamic clusters
std::vector<ClusterExtractor::PointWithAttributes> ClusterExtractor::getDynamicClusterPoints() const {
    std::vector<PointWithAttributes> dynamicPoints;

    for (const auto& cluster : propertiesList_) {
        if (cluster.isDynamic) {
            // Append all points from this dynamic cluster to the result vector
            dynamicPoints.insert(dynamicPoints.end(), cluster.data.begin(), cluster.data.end());
        }
    }

    return dynamicPoints;  // Return the list of points from dynamic clusters
}