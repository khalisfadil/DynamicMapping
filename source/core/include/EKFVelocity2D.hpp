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
#include <Eigen/Dense>
#include <memory>

class EKFVelocity2D {
public:
    //##############################################################################
    // EKFVelocity2D
    EKFVelocity2D(const Eigen::Vector2f& initialPosition);
    //##############################################################################
    // predict
    void predict(double dt);  // Predict with variable dt
    //##############################################################################
    // update
    void update(const Eigen::Vector2f& positionMeasurement);
    //##############################################################################
    // clone
    std::unique_ptr<EKFVelocity2D> clone() const;
    //##############################################################################
    // getStateVelocityError
    float getStateVelocityError() const ;
    //##############################################################################
    // getPredictedVelocity
    Eigen::Vector2f getPredictedVelocity() const;

private:
    //##############################################################################
    // Persistent member variables (one-time defined parameters)
    Eigen::Vector4f x_;  // State vector [x, y, vx, vy]
    Eigen::Matrix4f P_;  // Covariance matrix
    Eigen::Matrix4f Q_;  // Process noise
    Eigen::Matrix2f R_;  // Measurement noise
    Eigen::Matrix4f F_;  // State transition matrix
    Eigen::Matrix<float, 2, 4> H_;  // Measurement matrix
    Eigen::Vector4f predictedState_;  // Store the predicted state for error calculation
    float stateVelocityError_;
};

