#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "Model.h"
using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 4){
        cout << "Please provide values for M, Lamda, and the file name." << endl;
        return 1;
    }
    int M = stoi(argv[1]);
    double Lamda = stod(argv[2]);
    string file = argv[3];
    Model model(M, Lamda, file);
    model.Train();
    model.Train_5fold();
    model.Test();
}
