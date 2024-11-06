#include "dynamicMapping.h"
#include "occupancyMap.hpp"

// Persistent global pointer to OccupancyMap instance
static OccupancyMap* occMapInstance = nullptr;

// Initialize resources for dynamic mapping with specified parameters
void CreateDynamicMapping() {
    if (occMapInstance == nullptr) {
        double resolution = 0.1;                   // Example voxel resolution
        Eigen::Vector3d map_center(0.0, 0.0, 0.0); // Example map center
        double max_distance = 100.0;               // Example max distance

        occMapInstance = new OccupancyMap(resolution, map_center, max_distance);
    }
}

// Clean up resources for dynamic mapping
void DeleteDynamicMapping() {
    if (occMapInstance != nullptr) {
        delete occMapInstance;
        occMapInstance = nullptr;
    }
}

// Main function for dynamic mapping
void OutputDynamicMapping(float* inputCloud, uint32_t numInputCloud,
                          double maxDistance, 
                          float* occupiedPositions,
                          float* coloring, double& voxelSize) {
    if (occMapInstance == nullptr) {
        // Ensure instance is initialized with CreateDynamicMapping
        CreateDynamicMapping();
    }

    // // Update the occupancy map with input point cloud
    // occMapInstance->updateOccupancy(inputCloud, numInputCloud, maxDistance);

    // // Retrieve occupied cells and their colors for visualization
    // occupiedPositions = occMapInstance->getOccupiedCells();
    // coloring = occMapInstance->getCellColors();
    // voxelSize = occMapInstance->getVoxelSize();
}
