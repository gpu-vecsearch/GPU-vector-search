
float squared_l2(const float* base, const float* query, int dim)
{
    float distance=0;
    for(int i=0; i<dim; i++)
    {
        float diff = (*(base+i) - *(query+i));
        distance += diff * diff;
    }
    return distance;
}