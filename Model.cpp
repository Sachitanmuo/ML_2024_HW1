#include"Model.h"

Model::Model(int m, double lamda){
    M = m;
    Lamda = lamda;
    read_file();
    Training_set_normalized = Normalize(Training_set);
    Testing_set_normalized = Normalize(Testing_set);
}

double Model::sigmoid(double x){
    return 1./(1. + exp(-x));
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
        getline(iss, token, ',');
        songdata.output = stod(token);
        for(int i = 0; i < 11 ; i++){
            getline(iss, token, ',');
            InputVector.push_back(stod(token));
        }    
        songdata.input = InputVector;
        count++ <= 10000 ? Training_set.push_back(songdata) : Testing_set.push_back(songdata);

    }
}



void Model::initialize_params(){
    vector<vector<double> > weight (11, vector<double>(M, 1.));
}
double Model::Error_func(vector<double> prediction, vector<double> ground_truth){
    double sum = 0;
    for(int i = 0; i < prediction.size(); i++){
        sum += pow(prediction[i] - ground_truth[i], 2);
    }
    return sum / 2.;
}

void Model::Train(){
    Normalize(Training_set);
    Normalize(Testing_set);
    Design_Matrix = generate_Design_Matrix();
    
    cout << "Design_Matrix: " << endl;
    
    for(int i = 0; i < 5;i++){
        for(int j = 0; j < (*Design_Matrix)[0].size();j++){
            std::cout << (*Design_Matrix)[i][j][0] << " ";
        }
        std::cout << endl;
    }
    
   
    D_M = generate_D_M();
    
    cout << "D_M: " << endl;
    for(int i = 0; i < 5;i++){
        for(int j = 0; j < M;j++){
            std::cout << (*D_M)[0](i, j) << " ";
        }
        std::cout << endl;
    }
    /*
    W_ML = calculate_W_ML(); //(Mx11)
    
    cout << "W_ML: " << endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j< 11; j++){
            std::cout << (*W_ML)[i][j] << " ";
        }
        std::cout << endl;
    }
    */
    
    W_ML = calculate_W_ML_();

    cout << "W_ML_2: " << endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j< 11; j++){
            std::cout << (*W_ML)[i][j] << " ";
        }
        std::cout << endl;
    }
    
    int N = Training_set_normalized.size();
    int K = Training_set_normalized[0].input.size();
    vector<double> prediction(N, 0);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                prediction[i] += (*W_ML)[j][k] * phi(Training_set_normalized[i].input[k],j);
            }
        }
    }
    write_file(prediction, "prediction.csv");
    for(int i = 0 ; i < N; i++){
        std::cout << "Prediction: " << prediction[i] << "   |   Actual: " << Training_set_normalized[i].output << endl; 
    }


    vector<double> ground_truth;
    for(auto& data: Training_set_normalized){
        ground_truth.push_back(data.output);
    }

    double error = Error_func(prediction, ground_truth);
    double acc = Acc(prediction, ground_truth);
    std::cout << "Total Training Error = " << error <<endl;
    std::cout << "Accuracy = " << acc <<endl;
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
            normalized[i].output = raw_data[i].output;
        }
    }
    return normalized;
}





vector<vector<vector<double>>>* Model::generate_Design_Matrix(){
    int N = Training_set_normalized.size();
    int K = 11;
    vector<vector<vector<double>>>* DM = new vector<vector<vector<double>>>(N, vector<vector<double>>(M, vector<double>(K))); // [10000, M, 11]
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                (*DM)[i][j][k] = phi(Training_set_normalized[i].input[k], j);
            }
        }
    }
    return DM;

}

double Model::phi(double x_k, int j)
{
    return j > 0 ?  sigmoid((x_k - (3.*(-M+1+2*((double)j - 1.)*(((double)M-1.)/((double)M-2.)))/(double)M))/0.1) : 1;
}

vector<vector<double>>* Model::calculate_W_ML(){
    /*Accroding to the slides:
    W_ML = (lamda*I + []^T[])^(-1) []^T t
    */


   int N = Design_Matrix->size(); // N is the number of Trainin data
   int K = Training_set_normalized[0].input.size(); // K is the number of the features
   
   //First calculate []^T[], It's shape is (M, M)
   vector<vector<double>>* A = new vector<vector<double>>(M, vector<double>(M, 0));
    for(int i = 0; i < K; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < M; k++){
                for(int l = 0; l < N; l++){
                    (*A)[j][k] += (*Design_Matrix)[l][j][i] * (*Design_Matrix)[l][k][i];
                }
            }
        }
    }
    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            (*A)[i][j] += Lamda;
        }
    }
    //Inverse
    Eigen::MatrixXd *matrix = new Eigen::MatrixXd(M, M);
    for(int i=0; i < M; i++){
        for(int j = 0;j < M; j++){
            (*matrix)(i,j) = (*A)[i][j];
        }
    }
    //delete A;
    cout <<"Matrix: " << endl;
    cout << "Matrix A: " << endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            std::cout << (*A)[i][j] << " ";
        }
        std::cout << std::endl;
    }
    Eigen::MatrixXd matrix_inversed = matrix->inverse();
    delete matrix;

    cout << "Inversed Matrix: " << endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < M; j++){
            std::cout << matrix_inversed(i, j) << " ";
        }
        std::cout << std::endl;
    }
    //Calculate ()^-1 DM^T
    vector<vector<vector<double>>> *B = new vector<vector<vector<double>>>(M, vector<vector<double>>(N, vector<double>(K, 0)));
    for(int i = 0; i < M; i++){ //Inverse_matrix row index
        for(int j = 0; j < N; j++){ //DM^T column index = DM row index
            for(int k = 0;k < M; k++){ //each element in the row
                for(int l = 0;l < K; l++){ //each x_k in the x_vector
                    (*B)[i][j][l] += matrix_inversed(i, k) * (*Design_Matrix)[j][k][l];
                }
            }
        }
    }
    
    vector<vector<double>>* W = new vector<vector<double>>(M, vector<double>(11, 0));

    cout << "Pseudo inversed Matrix: " << endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j < 5; j++){
            for(int l = 0; l < K; l++){
                std::cout << (*B)[i][j][l] << " ";
            }
            std::cout << std::endl;
        }
    }

    //calculate W_ML = Bt
    for(int i = 0; i < M; i++){ //B row index
                                //ignore Y row index since the shape of Y is [N,1]
        for(int j=0; j < N; j++){ // each element in the row
            for(int l = 0;l < K; l++){ //each x_k in the x_vector
                (*W)[i][l] += (*B)[i][j][l] * Training_set_normalized[j].output;  
            }
            
        }
    }
    
    delete B;
    
    return W;
}
double Model::Acc(vector<double> predicted, vector<double> ground_truth){
    double Acc = 0;
    for(int i = 0; i < predicted.size(); i++){
        Acc += fabs((ground_truth[i] - predicted[i])/predicted[i]);
    }
    return (1 - Acc/predicted.size());
}

vector<Eigen::MatrixXd>* Model::generate_D_M(){
    vector<Eigen::MatrixXd>* DM = new vector<Eigen::MatrixXd>(11, Eigen::MatrixXd(Training_set_normalized.size(), M));
    //(Training_set_normalized.size(), vector<vector<double>>(M, vector<double>(11))); // [10000, M, 11]

    for(int i = 0; i < DM->size(); i++){
        for(int j = 0; j < Training_set_normalized.size(); j++){
            for(int k = 0; k < M; k++){
                (*DM)[i](j,k) = phi(Training_set_normalized[j].input[i], k);
            }
        }
    }
    return DM;
}

vector<vector<double>>* Model::calculate_W_ML_(){
    int N = Training_set_normalized.size();
    int K = Training_set_normalized[0].input.size();
    //reshape the [N,M,K] Matrix into [N,M*K] Matrix

    Eigen::MatrixXd Design_Matrix(N, M * K);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                Design_Matrix(i, j * K + k) = (*D_M)[k](i, j);
            }
        }
    }
    std::cout << "Flattering: " << endl;
    for(int i =0; i<5; i++){
        for(int j =0;j<K*M;j++){
            std::cout << Design_Matrix(i,j) << " ";
        }
        cout << endl;
    }
    Eigen::MatrixXd Pseudo_Inversed(M * K, N);
    Eigen::MatrixXd Pseudo_Inversed_(M * K, N);
    Pseudo_Inversed = Design_Matrix.completeOrthogonalDecomposition().pseudoInverse();
    

    cout << "Pseudo Inverse: " << endl;
    for(int i = 0; i < K*M ; i++){
        for(int j = 0 ; j < 5; j ++){
            cout << Pseudo_Inversed(i, j) << " ";
        }
        cout << endl;
    }
    vector<double> W_temp(M*K, 0);
    double temp = 0;
    for(int i =0; i<M*K;i++){
        temp = 0;
        for(int j = 0; j < N; j++){
            temp += Pseudo_Inversed(i,j) * Training_set_normalized[j].output;
        }
        W_temp[i] = temp;
    }
    vector<vector<double>>* W = new vector<vector<double>>(M, vector<double>(11, 0));

    for(int i = 0; i < M; i++){
        for(int j = 0; j < 11; j++){
            (*W)[i][j] = W_temp[i*11 + j];
        }
    }
 

    return W;
}


void Model::write_file(vector<double> y_pred, string filename){
    ofstream output_file(filename);
    for(int i = 0; i < y_pred.size(); i++){
        output_file << y_pred[i] << "," << Training_set[i].output << endl;
    }
    
    output_file.close();
}


/*
//==============Self Defined Pseudo Inverse=========================

    Eigen::MatrixXd transposed(N, M*K);
    transposed = Design_Matrix.transpose();
    Eigen::MatrixXd ATA(M*K, M*K);
    Eigen::MatrixXd Lambda(M*K, M*K);
    for(int i =0; i < M*K ; i++){
        for(int j = 0; j < M*K ; j++){
            if(i==j) Lambda(i, j) = Lamda;
            else Lambda(i, j) = 0;
        }
    }
    ATA = Design_Matrix.transpose() * Design_Matrix + Lambda;
    Eigen::MatrixXd Inversed(M*K, M*K);
    Inversed = ATA.inverse();
    Pseudo_Inversed_ = Inversed * transposed;
    
    for(int i = 0; i < M*K; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < M*K ; k++){
                Pseudo_Inversed_(i, j) += Inversed(i, k) * transposed(k, j);
            }
        }
    }
    //===================================================================
*/