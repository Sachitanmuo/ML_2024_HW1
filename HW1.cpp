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

class model{
private:
    vector<vector<double> > w;
    vector<SongData> Training_set;
    vector<SongData> Testing_set;
    int M;
public:
    model();
    void read_file();
    void initialize_params();
    vector<vector<double> > phi_func(vector<double>&);
    void train();
    void test();
    double sigmoid(double);
    double Error_func(SongData&);
};

model::model(){
    read_file();
}

double model::sigmoid(double x){
    return 1/(1 + exp(-x));
}

void model::read_file(){
    ifstream input_file("HW1.csv");
    string line;
    getline(input_file, line);

    int count = 1;

    while(getline(input_file, line)){
        istringstream iss(line);
        string token;
        SongData songdata;
        vector<double> InputVector;

        for(int i = 0; i < 12 ; i++){
            getline(iss, token, ',');
            InputVector.push_back(stod(token));
        }

        getline(iss, token, ',');
        songdata.input = InputVector;
        songdata.output = stod(token);

        count++ <= 10000 ? Training_set.push_back(songdata) : Testing_set.push_back(songdata);

    }

}

void model::initialize_params(){
    vector<vector<double> > weight (11, vector<double>(M, 1.));
}
vector<vector<double> > model::phi_func(vector<double>& input){
    vector<vector<double> > phi (11, vector<double>(M, 0));
    for(int i = 0; i < 11; i ++)
        for(int j = 0; j < M ; j++)
                phi[i][j] = j? sigmoid((3*j/M)/0.1) : 1;
    return phi;
}

double model::Error_func(SongData & data){
    vector<vector<double> > phi = phi_func(data.input);
    double y = 0;
    for(int i = 0; i < 11 ; i++)
        for(int j = 0; j < M ; j ++)
            y += w[i][j] * phi[j][i];
    return pow(y - data.output, 2);
}

void model::train(){

}

void model::test(){

}













int main()
{
}