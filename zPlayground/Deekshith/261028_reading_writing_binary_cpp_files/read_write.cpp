// I thoroughly understood reading and writing of binary files in cpp.
// The following is a small tutorial to consolidate my understanding.

#include<iostream>
#include<fstream> // required for file handling operations
#include<cstdint> // required to create an int of a uniform width across different cpp standard and architectures.

int main()
{
    // Firstly lets create an array of 10 elements.
    int32_t vector[10]= {11, 22, 33, 44, 55, 66, 77, 88, 99, 110};

    // Open the file in binary output mode
    std::ofstream file("output/data.bin", std::ios::binary);

    // Write method requires a character pointer.
    const char* c = reinterpret_cast<const char*>(vector); // Note here there is a hidden array to pointer decay
    // Appreciate why we are using a const here.

    file.write(c, sizeof(vector));
    file.close(); // This is important, as cpp doesn't write the files immediately.

    std::cout<<"Writing complete"<<"\n";

    // Having successfully created a file let us now try reading from the file.
    int32_t in_vec[10];

    // Open the file in binary input mode
    std::ifstream infile("output/data.bin", std::ios::binary);

    // Again read method requrires a character pointer
    char* inchar = reinterpret_cast<char*>(in_vec); // Note here there is a hidden array to pointer decay
    // Observe we are not using const here.

    infile.read(inchar, sizeof(in_vec));

    // Print the read vector
    for(int i=0; i<10; i++)
    {
        std::cout<<in_vec[i]<<"\n";
    }
    std::cout<<"Reading complete"<<"\n";
    return 0;  
}

//TODO: Revise this once again. next seekg() then loader.