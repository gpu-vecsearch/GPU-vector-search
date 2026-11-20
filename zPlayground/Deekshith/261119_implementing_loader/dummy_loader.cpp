#include <iostream>
#include <cstdint>
#include <fstream>
#include <vector>

int main()
{
    // initializing loading
    const int n_bases = 1000000, base_dim = 128;
    int32_t dim;
    std::vector<float> base_vecs(base_dim*n_bases);
    
    // opening the file
    std::string file_loc = "../../../data/sift/sift_base.fvecs";
    std::ifstream file(file_loc, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file!" << std::endl;
        return 1;
    }

    // reading the file
    for(int index=0; index<n_bases; index++)
    {
        file.read(reinterpret_cast<char*>(&dim),sizeof(int32_t));
        
        // The following warning acts as a check
        if (dim != base_dim)
        {
            std::cerr << "Error: Unexpected dimension at vector "<< index << ". Expected " << base_dim<< ", got " << dim << std::endl;
            return 1;
        }

        file.read(reinterpret_cast<char*>(&base_vecs[index*base_dim]),base_dim*sizeof(float));
        // Check on file reading
        if (file.fail())
        {
            std::cerr << "File reading failed at index "
                    << index << std::endl;
            return 1;
        }
    }

    // Print first five elements
    int indices[3] = {0, 1, 10};
    for(int j=0; j<3; j++)
    {
        std::cout<<"vector "<<indices[j]<<"\n";
        for(int i=0; i<5; i++)
        {
            std::cout<<base_vecs[indices[j]*base_dim + i]<<"\n";
        }
    }
    return 0;
}