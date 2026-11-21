#include "topk.h"
#include "squared_l2.h"

#include <algorithm>

std::vector<Neighbor> topk_serial(const float* query, const float* base, int dim, int nbase, int k)
{
    // Given one query let us evaluate all the distances sequentially
    float dist;
    std::vector<Neighbor> topk(k); 
    topk.reserve(k+1); // We add +1 for sliding the output out during insertion

    // Initilize the top_k vector
    for(int bid=0; bid<k; bid++)
    {
        dist = squared_l2(base + bid * dim, query, dim);
        topk[bid] = {bid, dist};
    }

    // sort the intialized top k using distance in ascending order
    std::sort(topk.begin(), topk.end(), [](const Neighbor& a, const Neighbor& b){return a.dist < b.dist;});
    float tol_dist = topk.back().dist;
        
    // Compare and slide
    for(int bid=k; bid< nbase; bid++)
    {
        dist = squared_l2(base + bid * dim, query, dim);
        Neighbor new_neighbor = {bid, dist};
        if (dist < tol_dist)
        {
            // Locate the position where the new neighbor fits
            auto it = std::lower_bound(topk.begin(), topk.end(), new_neighbor, 
            [](const Neighbor& a, const Neighbor& b){return a.dist < b.dist;});

            // Inser the new neighbor into the vecor which increases the vector size
            topk.insert(it, new_neighbor);
            topk.pop_back();
            tol_dist = topk.back().dist; // updating the tolerance distance
        }
    }
    return topk;
}