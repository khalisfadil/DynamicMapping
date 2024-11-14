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
#include "dynamicMapping.hpp"

// Persistent pointers to manage the instances of OccupancyMap and ClusterExtractor
static OccupancyMap* occupancyMapInstance = nullptr;
static ClusterExtractor* clusterExtractorInstance = nullptr;

// Initialize resources for dynamic mapping with specified parameters
void CreateDynamicMapping(double mapRes, 
                            double reachingDistance,
                            double* mapCenter,
                            double clusterTolerance,
                            int minClusterSize,
                            int maxClusterSize,
                            double staticThreshold,
                            double dynamicScoreThreshold,
                            double densityThreshold,
                            double velocityThreshold,
                            double similarityThreshold,
                            double maxDistanceThreshold,
                            double dt) {

    // Convert mapCenter array to Eigen::Vector3d
    Eigen::Vector3d mapCenterVec(mapCenter[0], mapCenter[1], mapCenter[2]);

    // Initialize the OccupancyMap instance
    if (!occupancyMapInstance) {
        occupancyMapInstance = new OccupancyMap(mapRes, reachingDistance, mapCenterVec);
    }

    // Initialize the ClusterExtractor instance if needed
    if (!clusterExtractorInstance) {
        clusterExtractorInstance = new ClusterExtractor(clusterTolerance, minClusterSize, maxClusterSize,
                                                        staticThreshold, dynamicScoreThreshold, densityThreshold,
                                                        velocityThreshold, similarityThreshold, maxDistanceThreshold, dt);
    }
}

// Main function for dynamic mapping
void OutputDynamicMapping(uint32_t numInputCloud,
                          float* inputCloud, 
                          float* reflectivity,
                          float* intensity,
                          float* NIR,
                          double* vehiclePosition,
                          uint32_t currFrame,
                          double mapRes, 
                          double reachingDistance,
                          double* mapCenter,
                          double clusterTolerance,
                          int minClusterSize,
                          int maxClusterSize,
                          double staticThreshold,
                          double dynamicScoreThreshold,
                          double densityThreshold,
                          double velocityThreshold,
                          double similarityThreshold,
                          double maxDistanceThreshold,
                          double dt,
                          double* outputStaticVoxelVec, uint32_t& staticVoxelVecSize,  // Pass size by reference to update
                          double* outputDynamicVoxelVec, uint32_t& dynamicVoxelVecSize,  // Same here for dynamic size
                          int* outputStaticOccupancyColors,
                          int* outputStaticReflectivityColors,
                          int* outputStaticIntensityColors,
                          int* outputStaticNIRColors,
                          int* outputDynamicColors) {

    // Constants for max occupancy sizes
    const uint32_t staticOccupancyMaxSize = 128 * 1024 * 10;
    const uint32_t dynamicOccupancyMaxSize = 128 * 1024;

    // Initialize output arrays
    std::fill(outputStaticVoxelVec, outputStaticVoxelVec + staticOccupancyMaxSize * 3, std::numeric_limits<double>::quiet_NaN());
    std::fill(outputDynamicVoxelVec, outputDynamicVoxelVec + dynamicOccupancyMaxSize * 3, std::numeric_limits<double>::quiet_NaN());
    std::fill(outputStaticOccupancyColors, outputStaticOccupancyColors + staticOccupancyMaxSize * 3, -1);
    std::fill(outputStaticReflectivityColors, outputStaticReflectivityColors + staticOccupancyMaxSize * 3, -1);
    std::fill(outputStaticIntensityColors, outputStaticIntensityColors + staticOccupancyMaxSize * 3, -1);
    std::fill(outputStaticNIRColors, outputStaticNIRColors + staticOccupancyMaxSize * 3, -1);
    std::fill(outputDynamicColors, outputDynamicColors + dynamicOccupancyMaxSize * 3, -1);

    // Reset sizes
    staticVoxelVecSize = 0;
    dynamicVoxelVecSize = 0;

    if (!occupancyMapInstance || !clusterExtractorInstance) {
        // Initialize if instances are null
        CreateDynamicMapping(mapRes, 
                                reachingDistance,
                                mapCenter,
                                clusterTolerance,
                                minClusterSize,
                                maxClusterSize,
                                staticThreshold,
                                dynamicScoreThreshold,
                                densityThreshold,
                                velocityThreshold,
                                similarityThreshold,
                                maxDistanceThreshold,
                                dt);
    }

    // Pre-size vectors to avoid resizing during parallel processing
    std::vector<Eigen::Vector3d> pointCloud(numInputCloud);
    std::vector<float> reflectivityVec(numInputCloud);
    std::vector<float> intensityVec(numInputCloud);
    std::vector<float> NIRVec(numInputCloud);

    // Use TBB parallel_for to populate the vectors in parallel
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, numInputCloud),
        [&](const tbb::blocked_range<uint32_t>& range) {
            for (uint32_t i = range.begin(); i < range.end(); ++i) {
                // Assign values to each vector at the index i
                pointCloud[i] = Eigen::Vector3d(inputCloud[i * 3], inputCloud[i * 3 + 1], inputCloud[i * 3 + 2]);
                reflectivityVec[i] = reflectivity[i];
                intensityVec[i] = intensity[i];
                NIRVec[i] = NIR[i];
            }
        });

    if (pointCloud.empty()) {
        std::cerr << "pointCloud is empty. Exiting processing." << std::endl;
        return;
    }

    Eigen::Vector3d mapCenterVec(mapCenter[0], mapCenter[1], mapCenter[2]);
    Eigen::Vector3d vehiclePos(vehiclePosition[0], vehiclePosition[1], vehiclePosition[2]);

    // Run clustering and dynamic mapping
    clusterExtractorInstance->runClusterExtractorPipeline(pointCloud, reflectivityVec, intensityVec, NIRVec);
    std::vector<ClusterExtractor::PointWithAttributes> dynamicCloud = clusterExtractorInstance->getDynamicClusterPoints();
    occupancyMapInstance->runOccupancyMapPipeline(pointCloud, reflectivityVec, intensityVec, NIRVec, dynamicCloud, vehiclePos, currFrame);

    // Process static voxels
    std::vector<OccupancyMap::VoxelData> staticVoxel = occupancyMapInstance->getStaticVoxels();
    if (!staticVoxel.empty()) {
        auto voxelColors = occupancyMapInstance->computeVoxelColors(staticVoxel);
        std::vector<Eigen::Vector3d> staticVoxelVec = occupancyMapInstance->getVoxelCenters(staticVoxel);
        staticVoxelVecSize = std::min(static_cast<uint32_t>(staticVoxelVec.size()), staticOccupancyMaxSize);

        const auto& occupancyColors = std::get<0>(voxelColors);
        const auto& reflectivityColors = std::get<1>(voxelColors);
        const auto& intensityColors = std::get<2>(voxelColors);
        const auto& NIRColors = std::get<3>(voxelColors);

        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, staticVoxelVecSize),
            [&](const tbb::blocked_range<uint32_t>& range) {
                for (uint32_t i = range.begin(); i < range.end(); ++i) {
                    outputStaticVoxelVec[i] = staticVoxelVec[i].x();
                    outputStaticVoxelVec[i + staticOccupancyMaxSize] = staticVoxelVec[i].y();
                    outputStaticVoxelVec[i + staticOccupancyMaxSize * 2] = staticVoxelVec[i].z();
                    outputStaticOccupancyColors[i] = occupancyColors[i].x();
                    outputStaticOccupancyColors[i + staticOccupancyMaxSize] = occupancyColors[i].y();
                    outputStaticOccupancyColors[i + staticOccupancyMaxSize * 2] = occupancyColors[i].z();
                    outputStaticReflectivityColors[i] = reflectivityColors[i].x();
                    outputStaticReflectivityColors[i + staticOccupancyMaxSize] = reflectivityColors[i].y();
                    outputStaticReflectivityColors[i + staticOccupancyMaxSize * 2] = reflectivityColors[i].z();
                    outputStaticIntensityColors[i] = intensityColors[i].x();
                    outputStaticIntensityColors[i + staticOccupancyMaxSize] = intensityColors[i].y();
                    outputStaticIntensityColors[i + staticOccupancyMaxSize * 2] = intensityColors[i].z();
                    outputStaticNIRColors[i] = NIRColors[i].x();
                    outputStaticNIRColors[i + staticOccupancyMaxSize] = NIRColors[i].y();
                    outputStaticNIRColors[i + staticOccupancyMaxSize * 2] = NIRColors[i].z();
                }
            });
    }

    // Process dynamic voxels
    std::vector<OccupancyMap::VoxelData> dynamicVoxel = occupancyMapInstance->getDynamicVoxels();
    if (!dynamicVoxel.empty()) {
        std::vector<Eigen::Vector3d> dynamicVoxelVec = occupancyMapInstance->getVoxelCenters(dynamicVoxel);
        dynamicVoxelVecSize = std::min(static_cast<uint32_t>(dynamicVoxelVec.size()), dynamicOccupancyMaxSize);
        std::vector<Eigen::Vector3i> dynamicVoxelColor = occupancyMapInstance->assignVoxelColorsRed(dynamicVoxel);

        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, dynamicVoxelVecSize),
            [&](const tbb::blocked_range<uint32_t>& range) {
                for (uint32_t i = range.begin(); i < range.end(); ++i) {
                    outputDynamicVoxelVec[i] = dynamicVoxelVec[i].x();
                    outputDynamicVoxelVec[i + dynamicOccupancyMaxSize] = dynamicVoxelVec[i].y();
                    outputDynamicVoxelVec[i + dynamicOccupancyMaxSize * 2] = dynamicVoxelVec[i].z();
                    outputDynamicColors[i] = dynamicVoxelColor[i].x();
                    outputDynamicColors[i + dynamicOccupancyMaxSize] = dynamicVoxelColor[i].y();
                    outputDynamicColors[i + dynamicOccupancyMaxSize * 2] = dynamicVoxelColor[i].z();
                }
            });
    }
}
// Clean up resources for dynamic mapping
void DeleteDynamicMapping() {
    // Delete dynamically allocated instances
    if (occupancyMapInstance) {
        delete occupancyMapInstance;
        occupancyMapInstance = nullptr;  // Avoid dangling pointer
    }

    if (clusterExtractorInstance) {
        delete clusterExtractorInstance;
        clusterExtractorInstance = nullptr;  // Avoid dangling pointer
    }
}

