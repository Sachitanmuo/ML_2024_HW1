#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "Model.h"
using namespace std;


int main()
{
    int M = 8;
    double Lamda =  0.1;
    Model model(M, Lamda);
    model.Train();
}