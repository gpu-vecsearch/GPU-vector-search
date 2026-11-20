#include <iostream>
#include <vector>
#include <cstdint>
#include "loader.h"

int main()
{
    const int n_bases = 1000000;
    const int n_query = 10000;
    const int base_dim = 128;
    const int gt_dim = 100;

    std::vector<float> base_vecs(n_bases * base_dim);
    std::vector<int32_t> gt_indices(n_query * gt_dim);

    // loading the base vectors
    if (!load_fvecs("../../../data/sift/sift_base.fvecs",base_vecs,n_bases,base_dim))
    {
        return 1;
    }

    std::cout << "Successfully loaded base vectors!" << std::endl;

    // loding the ground truth indices
    if (!load_ivecs("../../../data/sift/sift_groundtruth.ivecs",gt_indices,n_query,gt_dim))
    {
        return 1;
    }

    std::cout << "Successfully loaded ground truth indices!" << std::endl;

    return 0;
}