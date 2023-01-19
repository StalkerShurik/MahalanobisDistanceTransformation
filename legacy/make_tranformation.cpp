#include "fstream"
#include "iostream"
#include "../library.h"

int main() {
    std::ifstream fin("input.txt");
    std::ofstream fout("output.txt");
    std::string command;

    fin >> command;

    if (command == "make_transformation_2d" || command == "make_transformation_2d_brute") { //connectivity_type, is_signed, Tmatrix, height, width, picture

        std::string connectivity_type;
        bool is_signed;
        TransformationMatrix t_matrix = {{1, 0},
                                         {0, 1}};
        size_t height;
        size_t width;
        Image2d picture;


        fin >> connectivity_type;
        fin >> is_signed;
        fin >> t_matrix[0][0] >> t_matrix[0][1] >> t_matrix[1][0] >> t_matrix[1][1];
        fin >> height >> width;

        picture.resize(height, std::vector<double>(width));
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                fin >> picture[i][j];
            }
        }
        Image2d result;
        if (command == "") {
            result = make_transformation_2d(picture, t_matrix, connectivity_type, is_signed);
        } else {
            result = make_transformation_2d_brute(picture, t_matrix, is_signed);
        }
        for (size_t i = 0; i < height; ++i) {
            for (size_t j = 0; j < width; ++j) {
                fout << result[i][j] << '\n';
            }
        }
    } else {
        fout << "Error: incorrect command";
    }
}