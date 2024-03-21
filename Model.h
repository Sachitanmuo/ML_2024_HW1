#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
#include "eigen-3.3.7/Eigen/Eigen"
#include "eigen-3.3.7/Eigen/SVD"
using namespace std;


struct SongData {
    vector<double> input;
    double output;
};

class Model{
private:
    Eigen::MatrixXd* W_ML; //[M,11]
    Eigen::MatrixXd five_fold_W_ML;
    vector<SongData> Training_set;
    vector<SongData> Testing_set;
    vector<vector<SongData>> Training_set_5fold = vector<vector<SongData>>(5);
    vector<vector<SongData>>Testing_set_5fold = vector<vector<SongData>>(5);
    vector<vector<SongData>> Training_set_5fold_norm = vector<vector<SongData>>(5);
    vector<vector<SongData>> Testing_set_5fold_norm = vector<vector<SongData>>(5);
    vector<SongData> Training_set_normalized;
    vector<SongData> Testing_set_normalized;
    vector<vector<vector<double>>>* Design_Matrix;
    vector<Eigen::MatrixXd>* D_M; // a new format to store the design matrix
    int M;
    double Lamda;
    string File;
    vector<double> Mean;
    vector<double> Sd;
    vector<double>mean_[5];
    vector<double>sd_[5];
public:
    Model(int m, double lamda, string file);
    void read_file(int offset);
    void read_file();
    void initialize_params();
    void Train();
    void Train_5fold();
    void Test();
    void Test_5fold();
    double sigmoid(double);
    double Error_func(vector<double> prediction, vector<double> ground_truth, Eigen::MatrixXd m);
    double Acc(vector<double> prediction, vector<double> ground_truth);
    vector<SongData> Normalize(vector<SongData>&);
    double phi(double x_k, int j);
    Eigen::MatrixXd* calculate_W_ML_(vector<Eigen::MatrixXd>* D_M, vector<SongData> s);
    vector<Eigen::MatrixXd>* generate_D_M(vector<SongData>);
    void write_file_pred(vector<double> y_pred, vector<double> actual, vector<double> x3, string filename);
};