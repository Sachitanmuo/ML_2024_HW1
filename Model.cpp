#include"Model.h"

Model::Model(int m, int lamda){
    M = m;
    Lamda = lamda;
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

double Model::Error_func(SongData & data){
}

void Model::Train(){
    Normalize(Training_set);
    Normalize(Testing_set);
    Design_Matrix = generate_Design_Matrix();
    W_ML = calculate_W_ML();

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

vector<vector<double>>* Model::Transpose(const vector<vector<double>> matrix){
    vector<vector<double>>* transposed = new vector<vector<double>>(matrix[0].size(), vector<double>(matrix.size()));

    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[i].size(); j++) {
            (*transposed)[j][i] = matrix[i][j];
        }
    }

    return transposed;
}

vector<vector<double>>* Model::Inverse(const vector<vector<double>> matrix){

}

vector<vector<vector<double>>>* Model::generate_Design_Matrix(){
    vector<vector<vector<double>>>* DM = new vector<vector<vector<double>>>(Training_set_normalized.size(), vector<vector<double>>(M, vector<double>(11)));
    
    for(int i = 0; i < DM->size(); i++){
        for(int j = 0; j < (*DM)[i].size(); j++){
            for(int k = 0; k< (*DM)[i][j].size(); k++){
                (*DM)[i][j][k] = phi(Training_set_normalized[i].input[j], k);
            }
        }
    }
    return DM;
}

double Model::phi(double x_k, int j)
{
    return j?  sigmoid((x_k - (3*(-M+1+2*(j - 1)*((M-1)/(M-2)))/M))/0.1) : 1;
}

vector<vector<double>>* Model::calculate_W_ML(){
    /*Accroding to the slides:
    W_ML = (lamda*I + []^T[])^(-1) []^T t
    */

   //First calculate []^T[] it's shape is (M, M)
   int N = Design_Matrix->size();
   int K = Training_set_normalized[0].input.size();
   vector<vector<double>> A(M, vector<double>(M, 0));
    for(int i = 0; i < A.size(); i++){
        for(int j = 0; j < A[i].size(); j++){
            for(int k = 0; k < N; k++){
                for(int l = 0; l < K; l++){
                    A[i][j] += (*Design_Matrix)[k][i][l] * (*Design_Matrix)[k][j][l];
                }
            }
            if(i == j) A[i][j] += Lamda;
        }
    }
    
    
    for(int i = 0; i < M ; i++){
        for(int j = 0; j < M; j++){
            std::cout << A[i][j] << " ";
        }
        std::cout<<endl;
    }
    std::cout << "======================" << endl;
    //Inverse
    Eigen::MatrixXd matrix(M, M);
    for(int i=0; i < M; i++){
        for(int j = 0;j < M; j++){
            matrix(i,j) = A[i][j];
        }
    }
    Eigen::MatrixXd matrix_inversed = matrix.inverse();
    for(int i=0; i < M; i++){
        for(int j = 0;j < M; j++){
            std::cout << matrix_inversed(i,j) << " ";
        }
        std::cout << endl;
    }
    //Calculate ()^-1 DM^T
    vector<vector<vector<double>>> B(M, vector<vector<double>>(N, vector<double>(11, 0)));
    for(int i = 0; i < M; i++){ //Inverse_matrix row index
        for(int j=0; j < N; j++){ //DM^T column index = DM row index
            for(int k = 0;k < M; k++){ //each element in the row
                for(int l = 0;l < K; l++){ //each x_k in the x_vector
                    B[i][j][l] = matrix_inversed(i, k) * (*Design_Matrix)[j][k][l];
                }
            }
        }
    }
    vector<vector<double>>* W = new vector<vector<double>>(M, vector<double>(11, 0));


    //calculate W_ML = Bt
    for(int i = 0; i < M; i++){ //B row index
        for(int j=0; j < N; j++){ // t row index
            for(int k = 0;k < M; k++){ //each element in the row
                for(int l = 0;l < K; l++){ //each x_k in the x_vector
                    (*W)[i][l] = B[i][j][l] * Training_set_normalized[j].output;  
                }
            }
        }
    }
    return W;
}