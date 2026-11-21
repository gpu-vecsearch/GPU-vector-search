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

    // Serialized topk on all the query vectors
    int k = 10;
    // Declaring to store the results.
    std::vector<int> neighbor_indices(nquery * k); 
    std::vector<float>  squared_l2_distances(nquery * k);

    for (int qid=0; qid<nquery; qid++)
    {
        auto topk_indices = topk_serial(query.data() + qid * query_dim, base.data(), base_dim, nbase, k);
        for(int j = 0; j<k; j++)
        {
            neighbor_indices[qid * k + j] = topk_indices[j].id;
            squared_l2_distances[qid * k + j] = topk_indices[j].dist;
        }
        if(qid%100==0) std::cout <<"query "<< qid << " done." << std::endl;
    }
    
    return 0;
}