#include <iostream>
#include <cstdint>
#include <fstream>

int main()
{
    std::string file_loc = "../../../data/sift/sift_base.fvecs";
    std::ifstream base_vectors(file_loc, std::ios::binary);
    
    // Reading the first 4 bytes.
    int32_t dim; // a variable to store the result
    base_vectors.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));

    std::cout<<dim<<"\n";

    return 0;

}