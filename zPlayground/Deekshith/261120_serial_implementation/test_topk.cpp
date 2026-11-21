#include "loader.h"
#include "topk.h"

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

    // Let us see how efficient is top k
    int qid = 0;
    int k = 10;
    auto topk_indices = topk_serial(query.data() + qid * query_dim, base.data(), base_dim, nbase, k);
    for( int i=0; i <k; i++) std::cout << "Index: " << topk_indices[i].id << ", Distance: " << topk_indices[i].dist << std::endl;
    return 0;
}