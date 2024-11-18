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
#include "EKFVelocity2D.hpp"

//##############################################################################
// EKFVelocity2D Constructor
EKFVelocity2D::EKFVelocity2D(const Eigen::Vector2d& initialPosition) {
    // Initialize state vector with given position and zero velocity
    x_ << initialPosition(0), initialPosition(1), 0.0, 0.0;

    // Initialize the covariance matrix with high uncertainty for velocity
    P_ = Eigen::Matrix4d::Identity();
    P_(2, 2) = 1000.0;  // High uncertainty in x velocity
    P_(3, 3) = 1000.0;  // High uncertainty in y velocity

    Q_ = Eigen::Matrix4d::Identity() * 0.1;  // Process noise
    R_ = Eigen::Matrix2d::Identity() * 0.1;  // Measurement noise

    // Initialize F_ with a default time step of 1.0 (will be updated in predict)
    F_ << 1, 0, 1.0, 0,
          0, 1, 0, 1.0,
          0, 0, 1, 0,
          0, 0, 0, 1;

    // Measurement model (we observe position only)
    H_ << 1, 0, 0, 0,
          0, 1, 0, 0;
}
//##############################################################################
// predict
void EKFVelocity2D::predict(double dt) {
    // Update the state transition matrix F with the new dt
    F_(0, 2) = dt;
    F_(1, 3) = dt;

    // Predict state
    x_ = F_ * x_;  // New state estimate
    P_ = F_ * P_ * F_.transpose() + Q_;  // Update covariance

    // Store the predicted state for later comparison
    predictedState_ = x_;
}
//##############################################################################
// update
void EKFVelocity2D::update(const Eigen::Vector2d& positionMeasurement) {
    Eigen::Vector2d y = positionMeasurement - H_ * x_;  // Measurement residual
    Eigen::Matrix2d S = H_ * P_ * H_.transpose() + R_;  // Residual covariance
    Eigen::Matrix<double, 4, 2> K = P_ * H_.transpose() * S.inverse();  // Kalman gain
    x_ += K * y;  // Update state with measurement
    P_ = (Eigen::Matrix4d::Identity() - K * H_) * P_;  // Update covariance

    // Calculate the error as the norm of the difference between predicted and updated states
    stateVelocityError_ = (x_.segment<2>(2) - predictedState_.segment<2>(2)).norm();
}

//##############################################################################
// getStateVelocityError
double EKFVelocity2D::getStateVelocityError() const {
    // `const` ensures this function doesn't modify the object state
    return stateVelocityError_;
}
//##############################################################################
// getPredictedVelocity
Eigen::Vector2d EKFVelocity2D::getPredictedVelocity() const {
    // `const` ensures this function doesn't modify the object state
    return x_.segment<2>(2);  // Extracts the velocity [vx, vy]
}
//##############################################################################
// clone
std::unique_ptr<EKFVelocity2D> EKFVelocity2D::clone() const {
    // `const` ensures the clone function doesn't modify the object state
    auto newEKF = std::make_unique<EKFVelocity2D>(Eigen::Vector2d(x_[0], x_[1])); // Copy initial position
    newEKF->x_ = this->x_;      // Copy state
    newEKF->P_ = this->P_;      // Copy covariance
    newEKF->Q_ = this->Q_;      // Copy process noise
    newEKF->R_ = this->R_;      // Copy measurement noise
    newEKF->F_ = this->F_;      // Copy state transition model
    newEKF->H_ = this->H_;      // Copy measurement model
    return newEKF;
}
