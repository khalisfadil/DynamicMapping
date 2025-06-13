#pragma once

#include <open3d/Open3D.h>

namespace dynamicMap {

        /**
     * @brief Creates a triangular vehicle mesh transformed by a 4x4 homogeneous matrix.
     * @param T 4x4 transformation matrix in NED coordinates (x=front, y=right, z=down).
     * @return Shared pointer to an Open3D TriangleMesh in visualization coordinates (x=front, y=-right, z=-down).
     */
    std::shared_ptr<open3d::geometry::TriangleMesh> createVehicleMesh(const Eigen::Matrix4d& T) {
        // Validate transformation matrix
        if (T.rows() != 4 || T.cols() != 4 || !T.allFinite()) {
            return std::make_shared<open3d::geometry::TriangleMesh>(); // Return empty mesh for invalid input
        }

        auto vehicle_mesh = std::make_shared<open3d::geometry::TriangleMesh>();

        // Define local vertices in NED coordinates (vehicle shape: 10m long, 10m wide at rear)
        std::vector<Eigen::Vector3d> local_vertices = {
            {10.0, 0.0, 0.0},    // Front tip
            {-10.0, -5.0, 0.0},  // Rear left
            {-10.0, 5.0, 0.0}    // Rear right
        };

        // Define NED-to-visualization transformation matrix
        Eigen::Matrix4d ned_to_viz = Eigen::Matrix4d::Identity();
        ned_to_viz(1,1) = -1.0; // y_viz = -y_NED
        ned_to_viz(2,2) = -1.0; // z_viz = -z_NED

        // Combine transformations: first apply T (NED), then NED-to-viz
        Eigen::Matrix4d combined_T = ned_to_viz * T;

        // Transform vertices to world coordinates using homogeneous coordinates
        std::vector<Eigen::Vector3d> world_vertices;
        world_vertices.reserve(local_vertices.size());
        for (const auto& local_vertex : local_vertices) {
            // Convert to homogeneous coordinates [x, y, z, 1]
            Eigen::Vector4d homogeneous_vertex(local_vertex.x(), local_vertex.y(), local_vertex.z(), 1.0);
            // Apply combined transformation
            Eigen::Vector4d transformed = combined_T * homogeneous_vertex;
            // Extract 3D coordinates
            world_vertices.emplace_back(transformed.head<3>());
        }

        // Assign vertices to mesh
        vehicle_mesh->vertices_ = world_vertices;

        // Define single triangular face (vertices 0, 1, 2)
        vehicle_mesh->triangles_.push_back(Eigen::Vector3i(0, 1, 2));

        // Compute triangle normal in visualization coordinates
        Eigen::Vector3d v0 = world_vertices[1] - world_vertices[0];
        Eigen::Vector3d v1 = world_vertices[2] - world_vertices[0];
        Eigen::Vector3d normal = v0.cross(v1).normalized();
        vehicle_mesh->triangle_normals_.push_back(normal);

        // Assign green color to all vertices
        vehicle_mesh->vertex_colors_.reserve(local_vertices.size());
        vehicle_mesh->vertex_colors_ = {
            {0.0, 1.0, 0.0}, // Green for vertex 0
            {0.0, 1.0, 0.0}, // Green for vertex 1
            {0.0, 1.0, 0.0}  // Green for vertex 2
        };

        return vehicle_mesh;
    }

} // namespace dynamicMap