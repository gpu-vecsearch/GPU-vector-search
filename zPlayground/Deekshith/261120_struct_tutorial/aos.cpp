// We will explore the concept of array of structures (AoS)
#include <iostream>
#include <vector>

struct student
{
    int id;
    float score;
};

int main()
{
    std::vector<student> section;
    section.reserve(2); // assign a vector of capacity 100
    std::cout<<"Number of students assigned to the section is "<< section.size() << std::endl;
    std::cout<<"Capacity of the section is "<< section.capacity() << std::endl;
    
    student s1 ={1, 8.5f}; // This is called aggregate initilization.
    student s2 = {2, 10.0f};
    student s3 = {3, 7.0f};

    section.push_back(s1);
    section.push_back(s2);
    section.push_back(s3);
    std::cout<<"Number of students assigned to the section is "<< section.size() << std::endl;
    std::cout<<"Capacity of the section is "<< section.capacity();

    // Let us now print the marks of student 3
    std::cout<<"The score of student "<< section[2].id << " is " << section[2].score << std::endl;

    return 0;
}