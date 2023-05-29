#include <cstdint>
#include <string>
#include <vector>

using Image2d = std::vector<std::vector<double>>;
using Image3d = std::vector<std::vector<std::vector<double>>>;
using TransformationMatrix = std::vector<std::vector<double>>;

bool check_size_2d(const Image2d &image);

bool check_size_3d(const Image3d &image);

bool check_transformation_matrix_2d(const TransformationMatrix &transformation);

bool check_transformation_matrix_3d(const TransformationMatrix &transformation);

int32_t get_vertex_id_2d(const Image2d &image, int32_t x, int32_t y);

int32_t get_vertex_id_3d(const Image3d &image, int32_t x, int32_t y, int32_t z);

int32_t get_x_in_image_2d(const Image2d &image, int32_t vertex_id);

int32_t get_y_in_image_2d(const Image2d &image, int32_t vertex_id);

int32_t get_x_in_image_3d(const Image3d &image, int32_t vertex_id);

int32_t get_y_in_image_3d(const Image3d &image, int32_t vertex_id);

int32_t get_z_in_image_3d(const Image3d &image, int32_t vertex_id);

double calculate_distance_2d(std::pair<int32_t, int32_t> v1, std::pair<int32_t, int32_t> v2,
                             const TransformationMatrix &transformation);

double calculate_distance_3d(std::vector<double> &v1, std::vector<double> &v2,
                             const TransformationMatrix &transformation);

void
get_neighbours_4_connectivity_2d(const Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, const TransformationMatrix &transformation);

void
get_neighbours_6_connectivity_3d(const Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, const TransformationMatrix &transformation);

void
get_neighbours_diagonal_connectivity_2d(const Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                        int32_t vertex, const TransformationMatrix &transformation);

void
get_neighbours_diagonal_connectivity_3d(const Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                        int32_t vertex, const TransformationMatrix &transformation);

void
get_neighbours_8_connectivity_2d(const Image2d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                 int32_t vertex, const TransformationMatrix &transformation);

void
get_neighbours_26_connectivity_3d(const Image3d &image, std::vector<std::pair<int32_t, double>> &neighbours,
                                  int32_t vertex, const TransformationMatrix &transformation);

bool
is_border_2d(const Image2d &image, const std::vector<std::vector<std::pair<int32_t, double>>> &image_graph, int32_t vertex,
             bool black);

bool
is_border_3d(const Image3d &image, const std::vector<std::vector<std::pair<int32_t, double>>> &image_graph, int32_t vertex,
             bool black);


void build_graph_2d(const Image2d &image,
                    std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                    const TransformationMatrix &transformation, const std::string &connectivity_type);

void build_graph_3d(Image3d &image,
                    std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                    TransformationMatrix &transformation, std::string &connectivity_type);

void update_distances(std::vector<int32_t> &border,
                      std::vector<std::vector<std::pair<int32_t, double>>> &image_graph,
                      std::vector<double> &distances);

Image2d make_transformation_2d(Image2d &image,
                               TransformationMatrix transformation = {{1.0, 0.0},
                                                                      {0.0, 1.0}},
                               std::string connectivity_type = "8-connectivity",
                               bool is_signed = false);

Image3d make_transformation_3d(Image3d &image,
                               TransformationMatrix transformation = {{1.0, 0.0, 0.0},
                                                                      {0.0, 1.0, 0.0},
                                                                      {0.0, 0.0, 1.0}},
                               std::string connectivity_type = "6-connectivity",
                               bool is_signed = false);

bool is_border_2d_no_graph(Image2d &image, int32_t x, int32_t y, bool black);

Image2d make_transformation_2d_brute(Image2d &image, const TransformationMatrix& transformation,
                                     bool is_signed);

Image2d make_transformation_2d_ellipse(Image2d &image, double lambda1, double lambda2, double theta,
                                       std::string &connectivity_type, bool is_signed);


bool is_2d_connectivity_type_ok(std::string &connectivity_type);

bool is_3d_connectivity_type_ok(std::string &connectivity_type);

Image2d make_window_transformation_2d(Image2d &image, TransformationMatrix &transformation, double border_distance);
