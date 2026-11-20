#include "loader.h"
#include "squared_l2.h"

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

    int qid = 9;
    int bid = 10;
    float distance = squared_l2(base.data() + bid * base_dim, query.data() + qid*query_dim, base_dim);

    // print the output
    std::cout<<"Distance between query "<< qid << " and base " << bid << " is " << distance << std::endl; 
    return 0;
}