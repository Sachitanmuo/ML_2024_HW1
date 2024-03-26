#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "Model.h"
using namespace std;

int main(int argc, char* argv[])
{
    if(argc < 5){
        cout << "Please provide values for M, Lamda, and the file name." << endl;
        return 1;
    }
    int M = stoi(argv[1]);
    double Lamda = stod(argv[2]);
    string file(argv[3]);
    string demo_file(argv[4]);
    Model model(M, Lamda, file);

    for(int i = 0; i < 6; i++){
        cout << "M = " << 5*(i+1) << endl;
        model.set_M(5*(i+1));
        model.Train();
        model.Test();
        model.demo(demo_file);
    }

    cout << "Regularization with lambda = 0.1:" << endl;
    for(int i = 0; i < 6; i++){
        cout << "M = " << 5*(i+1) << endl;
        model.set_M(5*(i+1));
        model.Train();
        model.Test();
        model.demo(demo_file);
    }
    //model.Train_5fold();
    //model.Test_5fold();
}

