#include"Model.h"

Model::Model(int m){
    M = m;
    read_file();
    Training_set_normalized = Normalize(Training_set);
    Testing_set_normalized = Normalize(Testing_set);
}

double Model::sigmoid(double x){
    return 1/(1 + exp(-x));
}

void Model::read_file(){
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

void Model::initialize_params(){
    vector<vector<double> > weight (11, vector<double>(M, 1.));
}
vector<vector<double> > Model::Design_Matrix(vector<double>& input){
    vector<vector<double> > phi (11, vector<double>(M, 0));
    for(int i = 0; i < 11; i ++)
        for(int j = 0; j < M ; j++)
                phi[i][j] = j? sigmoid((input[i] - (3*(-M+1+2*(j - 1)*((M-1)/(M-2)))/M))/0.1) : 1;
    return phi;
}

double Model::Error_func(SongData & data){
    vector<vector<double> > phi = Design_Matrix(data.input);
    double y = 0;
    for(int i = 0; i < 11 ; i++)
        for(int j = 0; j < M ; j ++)
            y += W_ML[i][j] * phi[j][i];
    return pow(y - data.output, 2);
}

void Model::Train(){
    Normalize(Training_set);
    Normalize(Testing_set);

    for(int i = 0; i < 5 ; i++){
        for(int j = 0 ; j < Training_set_normalized[i].input.size() ; j++){
            cout << Training_set_normalized[i].input[j]<<" ";
        }
        cout << endl;
    }
}

void Model::Test(){

}

vector<SongData> Model::Normalize(vector<SongData>& raw_data){
    vector<double> mean(raw_data[0].input.size(), 0);
    vector<double> std_dev(raw_data[0].input.size(), 0);
    vector<SongData> normalized(raw_data.size());
    //calculate the mean
    for(auto & data : raw_data){
        for(int i = 0; i < data.input.size(); i++){
            mean[i] += data.input[i];
        }
    }
    for(auto& m : mean) m/= 10000;

    //calculate the standard deviation
    for(auto & data : raw_data){
        for(int i = 0; i < data.input.size(); i++){
            std_dev[i] += pow(data.input[i] - mean[i], 2);
        }
    }
    for(auto& sd : std_dev) sd = sqrt(sd/(10000-1));

    for(int i = 0; i < raw_data.size() ; i++){
        for(int j = 0; j < Training_set[i].input.size(); j++){
            normalized[i].input.push_back((raw_data[i].input[j] - mean[j])/std_dev[j]);
        }
    }
    return normalized;
}

vector<vector<double>>* Transpose(const vector<vector<double>> matrix){

}

vector<vector<double>>* Inverse(const vector<vector<double>> matrix){

}