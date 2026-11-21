#pragma once

#include <vector>

struct Neighbor
{
    int id;
    float dist;
};
std::vector<Neighbor> topk_serial(const float* query, const float* base, int dim, int nbases, int k);