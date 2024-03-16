#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "eigen-3.3.7/Eigen/Dense"
using namespace std;


struct SongData {
    vector<double> input;
    double output;
};

class Model{
private:
    vector<vector<double>>* W_ML;
    vector<SongData> Training_set;
    vector<SongData> Testing_set;
    vector<SongData> Training_set_normalized;
    vector<SongData> Testing_set_normalized;
    vector<vector<vector<double>>>* Design_Matrix;
    int M;
    double Lamda;
public:
    Model(int m, double lamda);
    void read_file();
    void initialize_params();
    void Train();
    void Test();
    double sigmoid(double);
    double Error_func(vector<double> prediction, vector<double> ground_truth);
    double Acc(vector<double> prediction, vector<double> ground_truth);
    vector<SongData> Normalize(vector<SongData>&);
    vector<vector<vector<double>>>* generate_Design_Matrix();
    double phi(double x_k, int j);
    vector<vector<double>>* calculate_W_ML();
};