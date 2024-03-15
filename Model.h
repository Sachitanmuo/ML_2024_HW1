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
    vector<vector<double> > W_ML;
    vector<SongData> Training_set;
    vector<SongData> Testing_set;
    vector<SongData> Training_set_normalized;
    vector<SongData> Testing_set_normalized;
    vector<vector<vector<double>>>* Design_Matrix;
    int M;
    int Lamda;
public:
    Model(int m, int lamda);
    void read_file();
    void initialize_params();
    void Train();
    void Test();
    double sigmoid(double);
    double Error_func(SongData&);
    vector<SongData> Normalize(vector<SongData>&);
    vector<vector<double>>* Transpose(const vector<vector<double>>);
    vector<vector<double>>* Inverse(const vector<vector<double>>);
    vector<vector<vector<double>>>* generate_Design_Matrix();
    double phi(double x_k, int j);
    void calculate_W_ML();
};