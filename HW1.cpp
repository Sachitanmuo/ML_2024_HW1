#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "Model.h"
using namespace std;

int main(int argc, char* argv[])
{
    double M = 0;
    double Lamda = 0;
    string file = "HW1.csv";
    string demo_file = argv[1];
    Model model(M, Lamda, file);
    for(int i = 0; i < 6; i++){
        cout << "M = " << 5*(i+1) << endl;
        model.set_M(5*(i+1));
        model.Train();
        model.Test();
        model.demo(demo_file);
    }
    cout << "5 fold cross validation using M = 10:" << endl;
    model.set_M(10);
    model.Train_5fold();
    model.Test_5fold();

    cout << "Regularization with lambda = 0.1:" << endl;
    cout << "===========================================" << endl;
    model.set_lambda(0.1);
    for(int i = 0; i < 6; i++){
        cout << "M = " << 5*(i+1) << endl;
        model.set_M(5*(i+1));
        model.Train();
        model.Test();
        model.demo(demo_file);
    }
}

