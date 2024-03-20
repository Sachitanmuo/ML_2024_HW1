#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "Model.h"
using namespace std;


int main(int argc, char* argv[])
{
    if(argc < 3){
        cout << "Please provide values for M and Lamda." << endl;
        return 1;
    }

    int M = stoi(argv[1]);
    double Lamda = stod(argv[2]);
    Model model(M, Lamda);
    model.Train();
}