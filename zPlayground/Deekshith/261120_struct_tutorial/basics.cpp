#include <iostream>

struct student
{
    int id;
    float score;
};

int main()
{
    student s1;
    s1.id = 27;
    s1.score = 98;
    std::cout<<"student id "<<s1.id<<std::endl;
    std::cout<<"Student score "<<s1.score<<std::endl; 
    return 0;
}