#include "loader.h"

#include <iostream>
#include <fstream>

bool load_fvecs(const std::string& file_loc,std::vector<float>& vectors,int n_vectors,int dim)
{
    int32_t dim_read;
    // opening the file
    std::ifstream file(file_loc, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file!" << std::endl;
        return false;
    }

    // reading the file
    for(int index=0; index<n_vectors; index++)
    {
        file.read(reinterpret_cast<char*>(&dim_read),sizeof(int32_t));
        if (file.fail())
        {
            std::cerr << "Failed to read dimension at index "<< index << std::endl;
            return false;
        }
                
        // The following warning acts as a check
        if (dim_read != dim)
        {
            std::cerr << "Error: Unexpected dimension at vector "<< index << ". Expected " << dim<< ", got " << dim_read << std::endl;
            return false;
        }

        file.read(reinterpret_cast<char*>(&vectors[index*dim]),dim*sizeof(float));
        // Check on file reading
        if (file.fail())
        {
            std::cerr << "File reading failed at index "<< index << std::endl;
            return false;
        }
    }
    return true;
}

bool load_ivecs(const std::string& file_loc,std::vector<int32_t>& vectors,int n_vectors,int dim)
{
    int32_t dim_read;

    // opening the file
    std::ifstream file(file_loc, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: Could not open file!" << std::endl;
        return false;
    }

    // reading the file
    for(int index=0; index<n_vectors; index++)
    {
        file.read(reinterpret_cast<char*>(&dim_read),sizeof(int32_t));
        if (file.fail())
        {
            std::cerr << "Failed to read dimension at index "<< index << std::endl;
            return false;
        }
                
        // The following warning acts as a check
        if (dim_read != dim)
        {
            std::cerr << "Error: Unexpected dimension at vector "<< index << ". Expected " << dim<< ", got " << dim_read << std::endl;
            return false;
        }

        file.read(reinterpret_cast<char*>(&vectors[index*dim]),dim*sizeof(int32_t));
        // Check on file reading
        if (file.fail())
        {
            std::cerr << "File reading failed at index "<< index << std::endl;
            return false;
        }
    }
    return true;
}