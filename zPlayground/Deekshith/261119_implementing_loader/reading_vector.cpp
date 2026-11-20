#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>

int main()
{
    int n=10; //for reading the nth vector
    std::vector<float> vector(128); //place where I ll save the vector
    int loc = 516*n+4; // Byte index of the vector

    std::string file_loc = "../../../data/sift/sift_base.fvecs";
    std::ifstream base_vectors(file_loc, std::ios::binary);
    base_vectors.seekg(loc, std::ios::beg); //move to the required byte index
    base_vectors.read(reinterpret_cast<char*>(vector.data()), 128*sizeof(float));

    // Print first five elements
    for(int i=0; i<5; i++)
    {
        std::cout<<vector[i]<<"\n";
    }
    return 0;
}
