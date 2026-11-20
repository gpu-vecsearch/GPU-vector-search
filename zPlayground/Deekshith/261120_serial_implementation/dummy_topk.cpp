#include "loader.h"
#include "squared_l2.h"
#include <algorithm>

struct Neighbor
{
    int id;
    float dist;
};

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

    // Given one query let us evaluate all the distances sequentially
    int qid = 0;
    int k = 10;
    float dist;
    std::vector<Neighbor> topk(k); 
    topk.reserve(k+1); // We add +1 for sliding the output out during insertion

    // Initilize the top_k vector
    for(int bid=0; bid<k; bid++)
    {
        dist = squared_l2(base.data() + bid * base_dim, query.data() + qid*query_dim, base_dim);
        topk[bid] = {bid, dist};
    }

    // sort the intialized top k using distance in ascending order
    std::sort(topk.begin(), topk.end(), [](const Neighbor& a, const Neighbor& b){return a.dist < b.dist;});
    float tol_dist = topk.back().dist;
        
    // Later we compare and slide
    for(int bid=k; bid< nbase; bid++)
    {
        dist = squared_l2(base.data() + bid * base_dim, query.data() + qid*query_dim, base_dim);
        Neighbor new_neighbor = {bid, dist};
        if (dist < tol_dist)
        {
            auto it = std::lower_bound(topk.begin(), topk.end(), new_neighbor, 
            [](const Neighbor& a, const Neighbor& b){return a.dist < b.dist;});
            topk.insert(it, new_neighbor);
            topk.pop_back();
            tol_dist = topk.back().dist; // updating the tolerance distance
        }
    }
    for(const Neighbor& n : topk) std::cout << "Index: " << n.id << ", Distance: " << n.dist << std::endl;
    return 0;
}