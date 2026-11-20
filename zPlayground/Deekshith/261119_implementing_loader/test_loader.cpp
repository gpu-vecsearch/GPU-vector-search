#include <iostream>
#include <vector>
#include "loader.h"

int main()
{
    const int n_bases = 1000000;
    const int base_dim = 128;

    std::vector<float> base_vecs(n_bases * base_dim);

    if (!load_fvecs("../../../data/sift/sift_base.fvecs",base_vecs,n_bases,base_dim))
    {
        return 1;
    }

    std::cout << "Successfully loaded SIFT1M!" << std::endl;

    return 0;
}