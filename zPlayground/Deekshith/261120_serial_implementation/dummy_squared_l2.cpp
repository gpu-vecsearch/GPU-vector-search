#include "loader.h"
#include <iostream>

int main()
{
    // Defining the constants
    const int base_dim = 128;
    const int nbase = 1000000;
    const int query_dim = 128;
    const int nquery = 10000;

    // Reading the base and query vectors
    std::vector<float> base(base_dim * nbase);
    std::vector<float> query(query_dim * nquery);
    
    if(!load_fvecs("../../../data/sift/sift_base.fvecs", base, nbase, base_dim)) return 1;
    if(!load_fvecs("../../../data/sift/sift_query.fvecs", query, nquery, query_dim)) return 1;

    int q_id = 9;
    int b_id = 10;
    float squared_l2 = 0;
    for(int i=0; i<base_dim; i++)
    {
        float diff = (base[base_dim * b_id + i] - query[query_dim * q_id + i]);
        squared_l2 += diff * diff;
    }

    // print the output
    std::cout<<"Distance between query "<< q_id << " and base " << b_id << " is " << squared_l2 << std::endl; 
    return 0;
}