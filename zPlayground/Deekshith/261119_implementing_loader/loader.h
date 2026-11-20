#pragma once

#include <string>
#include <vector>
#include <cstdint>

bool load_fvecs(const std::string& file_loc,std::vector<float>& vectors,int n_vectors,int dim);
bool load_ivecs(const std::string& file_loc,std::vector<int32_t>& vectors,int n_vectors,int dim);