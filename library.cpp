#include "library.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <queue>
#include <set>
#include <utility>


const int32_t inf = 10000;

//FUNCTIONS TO SWITCH BETWEEN IMAGE AND GRAPH ID'S

int32_t get_vertex_id_2d(const Image2d &image, int32_t x, int32_t y) {
    return x * static_cast<int32_t>(image[0].size()) + y;
}

int32_t get_vertex_id_3d(const Image3d &image, int32_t x, int32_t y, int32_t z) {
    return x * static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()) +
           y * static_cast<int32_t>(image[0][0].size()) + z;
}

int32_t get_x_in_image_2d(const Image2d &image, int32_t vertex_id) {
    return vertex_id / static_cast<int32_t>(image[0].size());
}

int32_t get_y_in_image_2d(const Image2d &image, int32_t vertex_id) {
    return vertex_id % static_cast<int32_t>(image[0].size());
}

int32_t get_x_in_image_3d(const Image3d &image, int32_t vertex_id) {
    return vertex_id / (static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()));
}

int32_t get_y_in_image_3d(const Image3d &image, int32_t vertex_id) {
    return (vertex_id % (static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()))) /
           static_cast<int32_t>(image[0][0].size());
}

int32_t get_z_in_image_3d(const Image3d &image, int32_t vertex_id) {
    return (vertex_id % (static_cast<int32_t>(image[0].size()) * static_cast<int32_t>(image[0][0].size()))) %
           static_cast<int32_t>(image[0][0].size());
}

//FUNCTIONS TO CALCULATE DISTANCE BETWEEN PIXELS

double calculate_distance_2d(std::pair<int32_t, int32_t> v1, std::pair<int32_t, int32_t> v2,
                             const TransformationMatrix &transformation) {
    v1.first -= v2.first;
    v1.second -= v2.second;
    return sqrt(1.0 * (v1.first * transformation[0][0] + v1.second * transformation[1][0]) * v1.first +
                (v1.first * transformation[0][1] + v1.second * transformation[1][1]) * v1.second);
}

double calculate_distance_3d(std::vector<double> &v1, std::vector<double> &v2,
                             const TransformationMatrix &transformation) {
    for (int32_t i = 0; i < 3; ++i) {
        v1[i] -= v2[i];
    }
    for (int32_t i = 0; i < 3; ++i) {
        v2[i] = v1[0] * transformation[0][i] + v1[1] * transformation[1][i] + v1[2] * transformation[2][i];
    }
    return sqrt(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}

//FUNCTIONS TO CHECK IF THE COORDINATES ARE VALID

bool is_not_out_of_borders_2d(const Image2d &image, int32_t x, int32_t y) {
    return x >= 0 && y >= 0 && x < image.size() && y < image[0].size();
}

bool is_not_out_of_borders_3d(const Image3d &image, int32_t x, int32_t y, int32_t z) {
    return x >= 0 && y >= 0 && z >= 0 && x < image.size() && y < image[0].size() && z < image[0][0].size();
}

//FUNCTIONS THAT ADD NEIGHBOURS OF THE CURRENT PIXEL TO THE GRAPH

void
get_neighbours_4_connectivity_2d(const Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, const TransformationMatrix &transformation) {
    int32_t int_x = get_x_in_image_2d(image, vertex);
    int32_t int_y = get_y_in_image_2d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i) + abs(j) == 1) {
                int32_t new_x = int_x + i;
                int32_t new_y = int_y + j;
                if (is_not_out_of_borders_2d(image, new_x, new_y)) {
                    neighbours.emplace_back(get_vertex_id_2d(image, new_x, new_y),
                                            calculate_distance_2d(std::make_pair(int_x, int_y),
                                                                  std::make_pair(new_x, new_y), transformation));
                }
            }
        }
    }
}

void
get_neighbours_6_connectivity_3d(const Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, const TransformationMatrix &transformation) {
    int32_t int_x = get_x_in_image_3d(image, vertex);
    int32_t int_y = get_y_in_image_3d(image, vertex);
    int32_t int_z = get_z_in_image_3d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(3)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            for (int32_t k = -1; k <= 1; ++k) {
                if (abs(i) + abs(j) + abs(k) == 1) {
                    int32_t new_x = int_x + i;
                    int32_t new_y = int_y + j;
                    int32_t new_z = int_z + k;
                    if (is_not_out_of_borders_3d(image, new_x, new_y, new_z)) {
                        std::vector<double> v1 = {static_cast<double >(int_x), static_cast<double >(int_y),
                                                  static_cast<double >(int_z)};
                        std::vector<double> v2 = {static_cast<double >(new_x), static_cast<double >(new_y),
                                                  static_cast<double >(new_z)};
                        neighbours.emplace_back(get_vertex_id_3d(image, new_x, new_y, new_z),
                                                calculate_distance_3d(v1, v2, transformation));
                    }
                }
            }
        }
    }
}

void
get_neighbours_diagonal_connectivity_2d(const Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                        int32_t vertex, const TransformationMatrix &transformation) {
    int32_t int_x = get_x_in_image_2d(image, vertex);
    int32_t int_y = get_y_in_image_2d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i) == 1 && abs(j) == 1) {
                int32_t new_x = int_x + i;
                int32_t new_y = int_y + j;
                if (is_not_out_of_borders_2d(image, new_x, new_y)) {
                    neighbours.emplace_back(get_vertex_id_2d(image, new_x, new_y),
                                            calculate_distance_2d(std::make_pair(int_x, int_y),
                                                                  std::make_pair(new_x, new_y), transformation));
                }
            }
        }
    }
}

void
get_neighbours_diagonal_connectivity_3d(const Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                        int32_t vertex, const TransformationMatrix &transformation) {
    int32_t int_x = get_x_in_image_3d(image, vertex);
    int32_t int_y = get_y_in_image_3d(image, vertex);
    int32_t int_z = get_z_in_image_3d(image, vertex);

#pragma omp parallel for num_threads(4) collapse(3)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            for (int32_t k = -1; k <= 1; ++k) {
                if (abs(i) + abs(j) + abs(k) >= 2) {
                    int32_t new_x = int_x + i;
                    int32_t new_y = int_y + j;
                    int32_t new_z = int_z + k;
                    if (is_not_out_of_borders_3d(image, new_x, new_y, new_z)) {
                        std::vector<double> v1 = {static_cast<double >(int_x), static_cast<double >(int_y),
                                                  static_cast<double >(int_z)};
                        std::vector<double> v2 = {static_cast<double >(new_x), static_cast<double >(new_y),
                                                  static_cast<double >(new_z)};
                        neighbours.emplace_back(get_vertex_id_3d(image, new_x, new_y, new_z),
                                                calculate_distance_3d(v1, v2, transformation));
                    }
                }
            }
        }
    }
}

void
get_neighbours_8_connectivity_2d(const Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, const TransformationMatrix &transformation) {
    get_neighbours_4_connectivity_2d(image, neighbours, vertex, transformation);
    get_neighbours_diagonal_connectivity_2d(image, neighbours, vertex, transformation);
}

void
get_neighbours_26_connectivity_3d(const Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                  int32_t vertex, const TransformationMatrix &transformation) {
    get_neighbours_6_connectivity_3d(image, neighbours, vertex, transformation);
    get_neighbours_diagonal_connectivity_3d(image, neighbours, vertex, transformation);
}

//FUNCTIONS CHECK IF THE PIXEL IS A BORDER PIXEL OF THE SET OF INTEREST

bool
is_border_2d(const Image2d &image, const std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
             int32_t vertex,
             bool black) {
#pragma omp parallel for num_threads(4)
    for (auto i: image_graph[vertex]) {
        int32_t x = get_x_in_image_2d(image, i.first);
        int32_t y = get_y_in_image_2d(image, i.first);
        if (image[x][y] == !black) {
            return true;
        }
    }
    return false;
}

bool
is_border_3d(const Image3d &image, const std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
             int32_t vertex,
             bool black) {
#pragma omp parallel for num_threads(4)
    for (auto i: image_graph[vertex]) {
        int32_t x = get_x_in_image_3d(image, i.first);
        int32_t y = get_y_in_image_3d(image, i.first);
        int32_t z = get_z_in_image_3d(image, i.first);
        if (image[x][y][z] == !black) {
            return true;
        }
    }
    return false;
}

//CHECK INPUT DATA

bool is_2d_connectivity_type_ok(std::string &connectivity_type) {
    return connectivity_type == "4-connectivity" || connectivity_type == "8-connectivity";
}


bool is_3d_connectivity_type_ok(std::string &connectivity_type) {
    return connectivity_type == "6-connectivity" || connectivity_type == "26-connectivity";
}

bool check_size_2d(const Image2d &image) {
    if (image.empty() || image[0].empty()) {
        return false;
    }
#pragma omp parallel for num_threads(4)
    for (size_t i = 1; i < image.size(); ++i) {
        if (image[i].size() != image[0].size()) {
            return false;
        }
    }
    return true;
}

bool check_size_3d(const Image3d &image) {
    if (image.empty() || image[0].empty()) {
        return false;
    }
#pragma omp parallel for num_threads(4)
    for (size_t i = 1; i < image.size(); ++i) {
        if (image[i].size() != image[0].size()) {
            return false;
        }
    }
#pragma omp parallel for num_threads(4) collapse(2)
    for (size_t i = 0; i < image.size(); ++i) {
        for (size_t j = 0; j < image[i].size(); ++j) {
            if (image[i][j].size() != image[0][0].size()) {
                return false;
            }
        }
    }
    return true;
}

bool check_transformation_matrix_2d(const TransformationMatrix &transformation) {
    if (transformation.size() != 2) {
        return false;
    }
    if (transformation[0].size() != 2 || transformation[1].size() != 2) {
        return false;
    }
    return transformation[0][0] > 0 &&
           transformation[0][0] * transformation[1][1] - transformation[1][0] * transformation[0][1] > 0;
}

bool check_transformation_matrix_3d(const TransformationMatrix &transformation) {
    if (transformation.size() != 3) {
        return false;
    }
    if (transformation[0].size() != 3 || transformation[1].size() != 3 || transformation[2].size() != 3) {
        return false;
    }
    if (transformation[0][0] <= 0 ||
        transformation[0][0] * transformation[1][1] - transformation[0][1] * transformation[1][0] <= 0) {
        return false;
    }
    double det = transformation[0][0] * transformation[1][1] * transformation[2][2] +
                 transformation[0][1] * transformation[1][2] * transformation[2][0] +
                 transformation[0][2] * transformation[1][0] * transformation[2][1] -
                 transformation[0][0] * transformation[1][2] * transformation[2][1] -
                 transformation[0][1] * transformation[1][0] * transformation[2][2] -
                 transformation[0][2] * transformation[1][1] * transformation[2][0];
    return det > 0;
}

bool check_input_2d(const Image2d &image, const TransformationMatrix &transformation, std::string &connectivity_type) {
    return check_size_2d(image) && check_transformation_matrix_2d(transformation) &&
           is_2d_connectivity_type_ok(connectivity_type);
}

bool check_input_2d(const Image2d &image, const TransformationMatrix &transformation) {
    return check_size_2d(image) && check_transformation_matrix_2d(transformation);
}

bool check_input_3d(const Image3d &image, const TransformationMatrix &transformation, std::string &connectivity_type) {
    return check_size_3d(image) && check_transformation_matrix_3d(transformation) &&
           is_3d_connectivity_type_ok(connectivity_type);
}

bool check_input_3d(const Image3d &image, const TransformationMatrix &transformation) {
    return check_size_3d(image) && check_transformation_matrix_3d(transformation);
}

//FUNCTIONS THAT BUILD GRAPH BY THE IMAGE

void build_graph_2d(const Image2d &image,
                    std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                    const TransformationMatrix &transformation, const std::string &connectivity_type) {
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        if (connectivity_type == "4-connectivity") {
            get_neighbours_4_connectivity_2d(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "8-connectivity") {
            get_neighbours_8_connectivity_2d(image, image_graph[i], i, transformation);
        } else {
            assert(0);
        }
    }
}

void build_graph_3d(Image3d &image,
                    std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                    TransformationMatrix &transformation, std::string &connectivity_type) {
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        if (connectivity_type == "6-connectivity") {
            get_neighbours_6_connectivity_3d(image, image_graph[i], i, transformation);
        } else if (connectivity_type == "26-connectivity") {
            get_neighbours_26_connectivity_3d(image, image_graph[i], i, transformation);
        } else {
            assert(0);
        }
    }
}

//DIJKSTRA ITERATOR

void update_distances(std::vector<int32_t> &border,
                      std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                      std::vector<double> &distances) {
    int32_t bucket_size = 100;
    int32_t left_bucket = 0;

    std::vector<std::set<std::pair<double, int32_t>>> buckets(inf / bucket_size + 1);
    std::set<std::pair<int32_t, int32_t>> priority_queue;
#pragma omp parallel for num_threads(4)
    for (auto &i: border) {
        buckets[0].insert({0, i});
        distances[i] = 0;
    }
    while (left_bucket < buckets.size()) {
        double current_distance = buckets[left_bucket].begin()->first;
        int32_t current_vertex = buckets[left_bucket].begin()->second;
        buckets[left_bucket].erase(buckets[left_bucket].begin());
#pragma omp parallel for num_threads(4)
        for (auto go_to: image_graph[current_vertex]) {
            if (distances[go_to.first] > current_distance + go_to.second) {
                int32_t new_bucket_id = std::floor(1.0 * (current_distance + go_to.second) / bucket_size);
                int32_t previous_bucket_id;
                previous_bucket_id = std::floor(1.0 * distances[go_to.first] / bucket_size);
                buckets[previous_bucket_id].erase({distances[go_to.first], go_to.first});
                buckets[new_bucket_id].insert({current_distance + go_to.second, go_to.first});
                distances[go_to.first] = current_distance + go_to.second;
            }
        }
#pragma omp parallel for num_threads(4)
        for (int32_t i = left_bucket; i < buckets.size(); ++i) {
            if (!buckets[i].empty()) {
                break;
            }
            left_bucket++;
        }
    }
}

//DFS FOR WINDOW ALGORITHM

void go_dfs(int32_t start_vertex, int32_t cur, double border_distance,
            std::vector<std::vector<std::pair<int32_t, double>>> &graph,
            std::vector<int32_t> &used, std::vector<int32_t> &visited_vertexes,
            std::vector<double> &distances, Image2d &image, TransformationMatrix &transformation) {
    visited_vertexes.push_back(cur);
    used[cur] = 1;

    for (int32_t i = 0; i < graph[cur].size(); ++i) {
        int32_t vertex = graph[cur][i].first;
        if (used[vertex]) {
            continue;
        }

        int32_t vertex_x = get_x_in_image_2d(image, vertex);
        int32_t vertex_y = get_y_in_image_2d(image, vertex);

        double distance = calculate_distance_2d(std::make_pair(get_x_in_image_2d(image, start_vertex), get_y_in_image_2d(image, start_vertex)),
                                                std::make_pair(vertex_x, vertex_y), transformation);

        if (distance > border_distance) {
            continue;
        }

        if (image[vertex_x][vertex_y] != 0) {
            distances[start_vertex] = std::min(distance, distances[start_vertex]);
            break;
        }
        go_dfs(start_vertex, vertex, border_distance, graph, used, visited_vertexes, distances, image, transformation);
    }
}

//WINDOW TRANSFORMATION

Image2d make_window_transformation_2d(Image2d &image, TransformationMatrix &transformation, double border_distance) {
    std::string connectivity_type = "8-connectivity";
    assert(check_input_2d(image, transformation, connectivity_type));
    assert(border_distance > 0);

    Image2d transformed_image(image.size());
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image.size(); ++i) {
        transformed_image[i].assign(image[i].size(), border_distance);
    }
    std::vector<std::vector<std::pair<int32_t, double>>> image_graph;
    image_graph.resize(image[0].size() * image.size());
    build_graph_2d(image, image_graph, transformation, connectivity_type);
    std::vector<double> distances(image_graph.size(), border_distance);

#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        if (image[current_x][current_y] != 0) {
            transformed_image[current_x][current_y] = 0;
            distances[i] = 0;
        }
    }

    std::vector<int32_t> used(distances.size(), 0);
    std::vector<int32_t> visited_vertexes;

    for (int32_t i = 0; i < distances.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        visited_vertexes.clear();
        if (image[current_x][current_y] == 0) {
            go_dfs(i, i, border_distance, image_graph, used, visited_vertexes, distances, image, transformation);
        }
        for (int32_t visited_vertex : visited_vertexes) {
            used[visited_vertex] = 0;
        }
    }

    for (int32_t i = 0; i < distances.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        transformed_image[current_x][current_y] = distances[i];
    }

    return transformed_image;
}

//DIJKSTRA

Image2d make_transformation_2d_c(Image2d &image, const TransformationMatrix& transformation,
                                 std::string connectivity_type, bool is_signed) {
    assert(check_input_2d(image, transformation, connectivity_type));
    Image2d transformed_image(image.size());
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image.size(); ++i) {
        transformed_image[i].assign(image[i].size(), inf);
    }

    std::vector<std::vector<std::pair<int32_t, double>>> image_graph;
    image_graph.resize(image[0].size() * image.size());
    build_graph_2d(image, image_graph, transformation, connectivity_type);
    std::vector<int32_t> border;
    std::vector<double> distances(image_graph.size(), inf);
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        if (image[current_x][current_y] != 0) {
            transformed_image[current_x][current_y] = 0;
            distances[i] = 0;
            if (is_border_2d(image, image_graph, i, true)) {
                border.push_back(i);
            }
        }
    }
    update_distances(border, image_graph, distances);
    for (int32_t i = 0; i < distances.size(); ++i) {
        int32_t current_x = get_x_in_image_2d(image, i);
        int32_t current_y = get_y_in_image_2d(image, i);
        transformed_image[current_x][current_y] = distances[i];
    }
    if (is_signed) {
        border.clear();
        distances.assign(distances.size(), inf);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image_graph.size(); ++i) {
            int32_t current_x = get_x_in_image_2d(image, i);
            int32_t current_y = get_y_in_image_2d(image, i);
            if (image[current_x][current_y] == 0) {
                distances[i] = 0;
                if (is_border_2d(image, image_graph, i, false)) {
                    border.push_back(i);
                }
            }
        }
        update_distances(border, image_graph, distances);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < distances.size(); ++i) {
            int32_t current_x = get_x_in_image_2d(image, i);
            int32_t current_y = get_y_in_image_2d(image, i);
            transformed_image[current_x][current_y] -= distances[i];
        }
    }
    return transformed_image;
}

//3D ALGO

Image3d make_transformation_3d(Image3d &image, TransformationMatrix transformation,
                               std::string connectivity_type, bool is_signed) {
    assert(check_size_3d(image));
    assert(check_transformation_matrix_3d(transformation));

    Image3d transformed_image(image.size());
#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = 0; i < image.size(); ++i) {
        for (int32_t j = 0; j < image.size(); ++j) {
            transformed_image[i][j].assign(image[i][j].size(), inf);
        }
    }

    std::vector<std::vector<std::pair<int32_t, double>>> image_graph;
    image_graph.resize(image[0].size() * image.size() * image[0][0].size());
    build_graph_3d(image, image_graph, transformation, connectivity_type);

    std::vector<int32_t> border;
    std::vector<double> distances(image_graph.size(), inf);
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image_graph.size(); ++i) {
        int32_t current_x = get_x_in_image_3d(image, i);
        int32_t current_y = get_y_in_image_3d(image, i);
        int32_t current_z = get_z_in_image_3d(image, i);
        if (image[current_x][current_y][current_z] != 0) {
            transformed_image[current_x][current_y][current_z] = 0;
            distances[i] = 0;
            if (is_border_3d(image, image_graph, i, true)) {
                border.push_back(i);
            }
        }
    }
    update_distances(border, image_graph, distances);
    for (int32_t i = 0; i < distances.size(); ++i) {
        int32_t current_x = get_x_in_image_3d(image, i);
        int32_t current_y = get_y_in_image_3d(image, i);
        int32_t current_z = get_z_in_image_3d(image, i);
        transformed_image[current_x][current_y][current_z] = distances[i];
    }
    if (is_signed) {
        border.clear();
        distances.assign(distances.size(), inf);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image_graph.size(); ++i) {
            int32_t current_x = get_x_in_image_3d(image, i);
            int32_t current_y = get_y_in_image_3d(image, i);
            int32_t current_z = get_z_in_image_3d(image, i);
            if (image[current_x][current_y][current_z] == 0) {
                distances[i] = 0;
                if (is_border_3d(image, image_graph, i, false)) {
                    border.push_back(i);
                }
            }
        }
        update_distances(border, image_graph, distances);
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < distances.size(); ++i) {
            int32_t current_x = get_x_in_image_3d(image, i);
            int32_t current_y = get_y_in_image_3d(image, i);
            int32_t current_z = get_z_in_image_3d(image, i);
            transformed_image[current_x][current_y][current_z] -= distances[i];
        }
    }
    return transformed_image;
}

//CHECK IF THE ELEMENT IS A BORDER ELEMENT IN IMAGE MODE

bool is_border_2d_no_graph(Image2d &image, int32_t x, int32_t y, bool black) {

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = -1; i <= 1; ++i) {
        for (int32_t j = -1; j <= 1; ++j) {
            if (abs(i + j) == 1) {
                int32_t new_x = x + i;
                int32_t new_y = y + j;
                if (new_x >= 0 && new_y >= 0 && new_x < image.size() && new_y < image[0].size()) {
                    if (image[new_x][new_y] == !black) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

//BRUTE

Image2d make_transformation_2d_brute(Image2d &image, const TransformationMatrix& transformation,
                                     bool is_signed) {
    assert(check_input_2d(image, transformation));

    Image2d transformed_image(image.size());
#pragma omp parallel for num_threads(4)
    for (int32_t i = 0; i < image.size(); ++i) {
        transformed_image[i].assign(image[i].size(), inf);
    }

    std::vector<std::pair<int32_t, int32_t>> border;

#pragma omp parallel for num_threads(4) collapse(2)
    for (int32_t i = 0; i < image.size(); ++i) {
        for (int32_t j = 0; j < image[i].size(); ++j) {
            if (image[i][j] != 0) {
                transformed_image[i][j] = 0;
                if (is_border_2d_no_graph(image, i, j, true)) {
                    border.emplace_back(i, j);
                }
            }
        }
    }
#pragma omp parallel for num_threads(4) collapse(3)
    for (int32_t i = 0; i < image.size(); ++i) {
        for (int32_t j = 0; j < image[i].size(); ++j) {
            if (image[i][j] == 0) {
                for (auto &k: border) {
                    int32_t x = k.first;
                    int32_t y = k.second;
                    transformed_image[i][j] = std::min(transformed_image[i][j],
                                                       calculate_distance_2d({i, j}, {x, y}, transformation));
                }
            }
        }
    }

    if (is_signed) {
        Image2d transformed_image_signed(image.size());
#pragma omp parallel for num_threads(4)
        for (int32_t i = 0; i < image.size(); ++i) {
            transformed_image_signed[i].assign(image[i].size(), inf);
        }
        border.clear();
#pragma omp parallel for num_threads(4) collapse(2)
        for (int32_t i = 0; i < image.size(); ++i) {
            for (int32_t j = 0; j < image[i].size(); ++j) {
                if (image[i][j] == 0) {
                    transformed_image_signed[i][j] = 0;
                    if (is_border_2d_no_graph(image, i, j, false)) {
                        border.emplace_back(i, j);
                    }
                }
            }
        }
#pragma omp parallel for num_threads(4) collapse(3)
        for (int32_t i = 0; i < image.size(); ++i) {
            for (int32_t j = 0; j < image[i].size(); ++j) {
                if (image[i][j] == 1) {
                    for (auto &k: border) {
                        int32_t x = k.first;
                        int32_t y = k.second;
                        transformed_image_signed[i][j] = std::min(transformed_image_signed[i][j],
                                                                  calculate_distance_2d({i, j}, {x, y},
                                                                                        transformation));
                    }
                }
            }
        }
        for (int32_t i = 0; i < image.size(); ++i) {
            for (int32_t j = 0; j < image[i].size(); ++j) {
                transformed_image[i][j] -= transformed_image_signed[i][j];
            }
        }
    }
    return transformed_image;
}
//ELLIPSE ALGO

Image2d make_transformation_2d_ellipse(Image2d &image, double lambda1, double lambda2, double theta,
                                       std::string &connectivity_type, bool is_signed) {
    std::vector<std::vector<double>> transformation(2, std::vector<double>(2));
    transformation[0][0] = lambda1 * lambda1 * cos(theta) * cos(theta) + lambda2 * lambda2 * sin(theta) * sin(theta);
    transformation[1][1] = lambda1 * lambda1 * sin(theta) * sin(theta) + lambda2 * lambda2 * cos(theta) * cos(theta);
    transformation[0][1] = transformation[1][0] = (lambda1 * lambda1 - lambda2 * lambda2) * sin(theta) * cos(theta);
    return make_transformation_2d_c(image, transformation, connectivity_type, is_signed);
}

//////////////////////////////////////////////////////////// PYBIND MOMENT


namespace py = pybind11;

void array2d_to_vector(const py::array_t<double> &numpy_array, Image2d &vector) {

    vector.resize(numpy_array.shape()[0]);
    for (size_t i = 0; i < vector.size(); ++i) {
        vector[i].resize(numpy_array.shape()[1]);
        for (size_t j = 0; j < vector[i].size(); ++j) {
            vector[i][j] = *numpy_array.data(i, j);
        }
    }
}


py::array MDT_connectivity(const py::array_t<double> &image, const py::array_t<double> &transformation,
                              std::string &connectivity_type, bool is_signed = false) {

    Image2d v_image(image.shape()[0], std::vector<double>(image.shape()[1]));
    std::vector<std::vector<double>> v_transformation(transformation.shape()[0],
                                                      std::vector<double>(transformation.shape()[1]));

    array2d_to_vector(image, v_image);
    array2d_to_vector(transformation, v_transformation);

    py::array ret = py::cast(make_transformation_2d_c(v_image, v_transformation, connectivity_type, is_signed));
    return ret;
}

py::array MDT_brute(const py::array_t<double>& image, const py::array_t<double>& transformation,
                    bool is_signed = false) {
    Image2d v_image(image.shape()[0], std::vector<double>(image.shape()[1]));
    std::vector<std::vector<double>> v_transformation(transformation.shape()[0],
                                                      std::vector<double>(transformation.shape()[1]));

    array2d_to_vector(image, v_image);
    array2d_to_vector(transformation, v_transformation);

    py::array ret = py::cast(make_transformation_2d_brute(v_image, v_transformation, is_signed));
    return ret;
}

py::array MDT_ellipse(const py::array_t<double>& image, double lambda1, double lambda2, double theta,
                      std::string &connectivity_type, bool is_signed) {
    Image2d v_image(image.shape()[0], std::vector<double>(image.shape()[1]));

    array2d_to_vector(image, v_image);

    py::array ret = py::cast(make_transformation_2d_ellipse(v_image, lambda1, lambda2, theta, connectivity_type, is_signed));
    return ret;
}

py::array MDT_window(const py::array_t<double>& image, const py::array_t<double>& transformation,
                    double distance) {
    Image2d v_image(image.shape()[0], std::vector<double>(image.shape()[1]));
    std::vector<std::vector<double>> v_transformation(transformation.shape()[0],
                                                      std::vector<double>(transformation.shape()[1]));

    array2d_to_vector(image, v_image);
    array2d_to_vector(transformation, v_transformation);

    py::array ret = py::cast(make_window_transformation_2d(v_image, v_transformation, distance));
    return ret;
}

PYBIND11_MODULE(mahalanobis_transformation, handle) {
    handle.doc() = "Description";
    handle.def("MDT_connectivity", &MDT_connectivity);
    handle.def("MDT_brute", &MDT_brute);
    handle.def("MDT_ellipse", &MDT_ellipse);
    handle.def("MDT_window", &MDT_window);
}