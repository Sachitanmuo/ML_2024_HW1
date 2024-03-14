
#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<cmath>
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
    int M;
public:
    Model(int m);
    void read_file();
    void initialize_params();
    vector<vector<double>> Design_Matrix(vector<double>&);
    void Train();
    void Test();
    double sigmoid(double);
    double Error_func(SongData&);
    vector<SongData> Normalize(vector<SongData>&);
    vector<vector<double>>* Transpose(const vector<vector<double>>);
    vector<vector<double>>* Inverse(const vector<vector<double>>);
    
};