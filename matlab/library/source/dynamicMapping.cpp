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
#include <iostream>
#include <limits>
#include <memory>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

// Persistent pointers to manage the instances of OccupancyMap and ClusterExtractor
static std::unique_ptr<OccupancyMap> occupancyMapInstance = nullptr;
static std::unique_ptr<ClusterExtractor> clusterExtractorInstance = nullptr;

// Constants for max occupancy sizes
constexpr uint32_t MAX_STATIC_OCCUPANCY = 128 * 1024 * 5;
constexpr uint32_t MAX_DYNAMIC_OCCUPANCY = 128 * 1024;

// Initialize resources for dynamic mapping with specified parameters
void CreateDynamicMapping() {}

// Main function for dynamic mapping
void OutputDynamicMapping(uint32_t numInputCloud,               //u1
                          float* inputCloud,                    //u2
                          float* reflectivity,                  //u3
                          float* intensity,                     //u4
                          float* NIR,                           //u5
                          double* vehiclePosition,              //u6
                          uint32_t currFrame,                   //u7
                          double mapRes,                        //u8
                          double reachingDistance,              //u9
                          double* mapCenter,                    //u10
                          double clusterTolerance,              //u11
                          uint32_t minClusterSize,              //u12
                          uint32_t maxClusterSize,              //u13
                          double staticThreshold,               //u14
                          double dynamicScoreThreshold,         //u15
                          double densityThreshold,              //u16
                          double velocityThreshold,             //u17
                          double similarityThreshold,           //u18
                          double maxDistanceThreshold,          //u19
                          double dt,                            //u20
                          double* outputStaticVoxelVec, uint32_t& staticVoxelVecSize,       //y1 y2
                          double* outputDynamicVoxelVec, uint32_t& dynamicVoxelVecSize,     //y3 y4
                          int* outputStaticOccupancyColors,                                 //y5
                          int* outputStaticReflectivityColors,                              //y6
                          int* outputStaticIntensityColors,                                 //y7
                          int* outputStaticNIRColors,                                       //y8
                          int* outputDynamicColors) {                                       //y9

    // Initialize output arrays
    std::fill(outputStaticVoxelVec, outputStaticVoxelVec + MAX_STATIC_OCCUPANCY * 3, std::numeric_limits<double>::quiet_NaN());
    std::fill(outputDynamicVoxelVec, outputDynamicVoxelVec + MAX_DYNAMIC_OCCUPANCY * 3, std::numeric_limits<double>::quiet_NaN());
    std::fill(outputStaticOccupancyColors, outputStaticOccupancyColors + MAX_STATIC_OCCUPANCY * 3, -1);
    std::fill(outputStaticReflectivityColors, outputStaticReflectivityColors + MAX_STATIC_OCCUPANCY * 3, -1);
    std::fill(outputStaticIntensityColors, outputStaticIntensityColors + MAX_STATIC_OCCUPANCY * 3, -1);
    std::fill(outputStaticNIRColors, outputStaticNIRColors + MAX_STATIC_OCCUPANCY * 3, -1);
    std::fill(outputDynamicColors, outputDynamicColors + MAX_DYNAMIC_OCCUPANCY * 3, -1);

    // Reset sizes
    staticVoxelVecSize = 0;
    dynamicVoxelVecSize = 0;

    Eigen::Vector3d mapCenterVec(mapCenter[0], mapCenter[1], mapCenter[2]);
    Eigen::Vector3d vehiclePos(vehiclePosition[0], vehiclePosition[1], vehiclePosition[2]);

    try {
        if (!occupancyMapInstance || !clusterExtractorInstance) {
            occupancyMapInstance = std::make_unique<OccupancyMap>(mapRes, reachingDistance, mapCenterVec);
            clusterExtractorInstance = std::make_unique<ClusterExtractor>(clusterTolerance, minClusterSize, maxClusterSize,
                                                                          staticThreshold, dynamicScoreThreshold, densityThreshold,
                                                                          velocityThreshold, similarityThreshold, maxDistanceThreshold, dt);
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception during initialization: " << e.what() << std::endl;
        return;
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
                pointCloud[i] = Eigen::Vector3d(inputCloud[i * 3], inputCloud[i * 3 + 1], inputCloud[i * 3 + 2]);
                reflectivityVec[i] = reflectivity[i];
                intensityVec[i] = intensity[i];
                NIRVec[i] = NIR[i];
            }
        });

    if (pointCloud.empty()) {
        std::cerr << "[ERROR] PointCloud is empty. Exiting processing." << std::endl;
        return;
    }

    // Run clustering and dynamic mapping
    clusterExtractorInstance->runClusterExtractorPipeline(pointCloud, reflectivityVec, intensityVec, NIRVec);
    std::vector<ClusterExtractor::PointWithAttributes> dynamicCloud = clusterExtractorInstance->getDynamicClusterPoints();
    occupancyMapInstance->runOccupancyMapPipeline(pointCloud, reflectivityVec, intensityVec, NIRVec, dynamicCloud, vehiclePos, currFrame);

    // Process static voxels
    std::vector<OccupancyMap::VoxelData> staticVoxel = occupancyMapInstance->getStaticVoxels();
    if (!staticVoxel.empty()) {
        auto voxelColors = occupancyMapInstance->computeVoxelColors(staticVoxel);
        std::vector<Eigen::Vector3d> staticVoxelVec = occupancyMapInstance->getVoxelCenters(staticVoxel);
        staticVoxelVecSize = std::min(static_cast<uint32_t>(staticVoxelVec.size()), MAX_STATIC_OCCUPANCY);

        const auto& occupancyColors = std::get<0>(voxelColors);
        const auto& reflectivityColors = std::get<1>(voxelColors);
        const auto& intensityColors = std::get<2>(voxelColors);
        const auto& NIRColors = std::get<3>(voxelColors);

        // Fill output vectors for static voxels
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, staticVoxelVecSize),
            [&](const tbb::blocked_range<uint32_t>& range) {
                for (uint32_t i = range.begin(); i < range.end(); ++i) {
                    outputStaticVoxelVec[i] = staticVoxelVec[i].x();
                    outputStaticVoxelVec[i + MAX_STATIC_OCCUPANCY] = staticVoxelVec[i].y();
                    outputStaticVoxelVec[i + MAX_STATIC_OCCUPANCY * 2] = staticVoxelVec[i].z();
                    outputStaticOccupancyColors[i] = occupancyColors[i].x();
                    outputStaticOccupancyColors[i + MAX_STATIC_OCCUPANCY] = occupancyColors[i].y();
                    outputStaticOccupancyColors[i + MAX_STATIC_OCCUPANCY * 2] = occupancyColors[i].z();
                    outputStaticReflectivityColors[i] = reflectivityColors[i].x();
                    outputStaticReflectivityColors[i + MAX_STATIC_OCCUPANCY] = reflectivityColors[i].y();
                    outputStaticReflectivityColors[i + MAX_STATIC_OCCUPANCY * 2] = reflectivityColors[i].z();
                    outputStaticIntensityColors[i] = intensityColors[i].x();
                    outputStaticIntensityColors[i + MAX_STATIC_OCCUPANCY] = intensityColors[i].y();
                    outputStaticIntensityColors[i + MAX_STATIC_OCCUPANCY * 2] = intensityColors[i].z();
                    outputStaticNIRColors[i] = NIRColors[i].x();
                    outputStaticNIRColors[i + MAX_STATIC_OCCUPANCY] = NIRColors[i].y();
                    outputStaticNIRColors[i + MAX_STATIC_OCCUPANCY * 2] = NIRColors[i].z();
                }
            });
    }

    // Process dynamic voxels
    std::vector<OccupancyMap::VoxelData> dynamicVoxel = occupancyMapInstance->getDynamicVoxels();
    if (!dynamicVoxel.empty()) {
        std::vector<Eigen::Vector3d> dynamicVoxelVec = occupancyMapInstance->getVoxelCenters(dynamicVoxel);
        dynamicVoxelVecSize = std::min(static_cast<uint32_t>(dynamicVoxelVec.size()), MAX_DYNAMIC_OCCUPANCY);
        std::vector<Eigen::Vector3i> dynamicVoxelColor = occupancyMapInstance->assignVoxelColorsRed(dynamicVoxel);

        // Fill output vectors for dynamic voxels
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, dynamicVoxelVecSize),
            [&](const tbb::blocked_range<uint32_t>& range) {
                for (uint32_t i = range.begin(); i < range.end(); ++i) {
                    outputDynamicVoxelVec[i] = dynamicVoxelVec[i].x();
                    outputDynamicVoxelVec[i + MAX_DYNAMIC_OCCUPANCY] = dynamicVoxelVec[i].y();
                    outputDynamicVoxelVec[i + MAX_DYNAMIC_OCCUPANCY * 2] = dynamicVoxelVec[i].z();
                    outputDynamicColors[i] = dynamicVoxelColor[i].x();
                    outputDynamicColors[i + MAX_DYNAMIC_OCCUPANCY] = dynamicVoxelColor[i].y();
                    outputDynamicColors[i + MAX_DYNAMIC_OCCUPANCY * 2] = dynamicVoxelColor[i].z();
                }
            });
    }
}

// Clean up resources for dynamic mapping
void DeleteDynamicMapping() {
    occupancyMapInstance.reset();  // Automatically deletes the instance
    clusterExtractorInstance.reset();
}
