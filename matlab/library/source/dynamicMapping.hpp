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

#include "M_occupancyMap.hpp"
#include "M_clusterExtractor.hpp"

extern void CreateDynamicMapping();

extern void OutputDynamicMapping(uint32_t numInputCloud,
                                    float* inputCloud, 
                                    float* reflectivity,
                                    float* intensity,
                                    float* NIR,
                                    float* vehiclePosition,
                                    uint32_t currFrame,
                                    float mapRes, 
                                    float reachingDistance,
                                    float* mapCenter,
                                    float clusterTolerance,
                                    uint32_t minClusterSize,
                                    uint32_t maxClusterSize,
                                    float staticThreshold,
                                    float dynamicScoreThreshold,
                                    float densityThreshold,
                                    float velocityThreshold,
                                    float similarityThreshold,
                                    float maxDistanceThreshold,
                                    double dt,
                                    float* outputStaticVoxelVec, uint32_t& staticVoxelVecSize,  // Pass size by reference to update
                                    float* outputDynamicVoxelVec, uint32_t& dynamicVoxelVecSize,  // Same here for dynamic size
                                    int* outputStaticOccupancyColors,
                                    int* outputStaticReflectivityColors,
                                    int* outputStaticIntensityColors,
                                    int* outputStaticNIRColors,
                                    int* outputDynamicColors);

extern void DeleteDynamicMapping();