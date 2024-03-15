#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "Model.h"
using namespace std;


int main()
{
    int M = 5, Lamda = 1;
    Model model(M, Lamda);
    model.Train();
}