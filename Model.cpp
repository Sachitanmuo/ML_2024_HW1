#include"Model.h"

Model::Model(int m, double lamda, string file){
    M = m;
    Lamda = lamda;
    read_file();
    File = file;
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
void Model::read_file(int offset){
    ifstream input_file("HW1.csv");
    string line;
    getline(input_file, line);

    int count = 1;
    vector<SongData> temp;
    while(getline(input_file, line)){
        istringstream iss(line);
        string token;
        SongData songdata;
        vector<double> InputVector;
        getline(iss, token, ',');
        songdata.output = stod(token);
        //cout << songdata.output << endl;
        for(int i = 0; i < 11 ; i++){
            getline(iss, token, ',');
            InputVector.push_back(stod(token));
        }    
        songdata.input = InputVector;
        temp.push_back(songdata);
        /*
        for(int i = 0; i< 11; i++){
            cout << temp[temp.size()-1].input[i] << " ";
        }
        cout << "\n" << temp[temp.size()-1].output << endl;*/
    }
    for(int i = 0; i < temp.size();i++){
        i < 10000 ? Training_set_5fold[offset].push_back(temp[(int)((i+ temp.size() * (offset/5.)))%temp.size()])
            : Testing_set_5fold[0].push_back(temp[(int)(i+ temp.size()* (offset/5.))%temp.size()]);
    }
}

void Model::Train_5fold(){
    ofstream info_file("5fold_info.txt");
    vector<Eigen::MatrixXd*> W(5);
    Eigen::MatrixXd final_weight = Eigen::MatrixXd::Zero(M, 11);
    vector<Eigen::MatrixXd>* D_M_[5];
    vector<SongData>* X;
    for(int i = 0; i < 5; i++) {
        W[i] = new Eigen::MatrixXd(M, 11);
    }
    for(int i = 0; i < 5; i++){
        read_file(i);
        Training_set_5fold[i] = Normalize(Training_set_5fold[i]);
        D_M_[i] = generate_D_M(Training_set_5fold[i]);
        W[i] = calculate_W_ML_(D_M_[i], Training_set_5fold[i]);
        vector<double> y_pred(Training_set_5fold[i].size(), 0);
        vector<double> actual(Training_set_5fold[i].size(), 0);
        vector<double> x3(Training_set_5fold[i].size(), 0);
        for(int x = 0; x < Training_set_5fold[i].size(); x++){
            for(int it = 0; it < M; it++){
                for(int j = 0; j < 11; j++){
                y_pred[x] += (*W[i])(it, j) *phi(Training_set_5fold[i][x].input[j], it);
                }
            }
            actual[x] = Training_set_5fold[i][x].output;
            x3.push_back(Training_set_normalized[i].input[2]);
            //cout << y_pred[x] << "|" << actual[x] << endl;
        }
        info_file << "Fold " << i+1<<":"<<endl;
        info_file << "Error: " << Error_func(y_pred, actual, (*W[i])) << endl;
        info_file << "Accuracy: " << Acc(y_pred, actual) << endl;
        string o = "fivefold_" + to_string(i) + ".csv";
        write_file_pred(y_pred, actual, x3, o);
    }
    for(int z = 0; z < 5; z++){
        for(int j = 0; j < M;j++){
            for(int k = 0; k < 11; k++){
                final_weight(j, k) = final_weight(j, k) + (*W[z])(j, k);
            }
        }
    }
    five_fold_W_ML = final_weight * 0.2;

}


void Model::initialize_params(){
    vector<vector<double> > weight (11, vector<double>(M, 1.));
}
double Model::Error_func(vector<double> prediction, vector<double> ground_truth, Eigen::MatrixXd m){
    Eigen::MatrixXd XTX = m * m.transpose();
    double sum = 0;
    for(int i = 0; i < M;i++){
        for(int j =0; j< M;j++){
            sum += fabs(Lamda * XTX(i, j));
        }
    }
    for(int i = 0; i < prediction.size(); i++){
        sum += pow(prediction[i] - ground_truth[i], 2);
    }
    return sum / prediction.size();
}

void Model::Train(){
    Training_set_normalized = Normalize(Training_set);
  
    D_M = generate_D_M(Training_set_normalized);
    
    cout << "D_M: " << endl;
    for(int i = 0; i < 5;i++){
        for(int j = 0; j < M;j++){
            std::cout << (*D_M)[0](i, j) << " ";
        }
        std::cout << endl;
    }
    W_ML = calculate_W_ML_(D_M, Training_set_normalized);

    cout << "W_ML_2: " << endl;
    for(int i = 0; i < M; i++){
        for(int j = 0; j< 11; j++){
            std::cout << (*W_ML)(i,j) << " ";
        }
        std::cout << endl;
    }
    
    int N = Training_set_normalized.size();
    int K = Training_set_normalized[0].input.size();
    vector<double> prediction(N, 0);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                prediction[i] += (*W_ML)(j, k) * phi(Training_set_normalized[i].input[k],j);
            }
        }
    }
    vector<double> actual, x3;
    for(int i = 0; i < Training_set_normalized.size(); i++){
        actual.push_back(Training_set_normalized[i].output);
        x3.push_back(Training_set[i].input[2]);
    }
    write_file_pred(prediction, actual, x3, "prediction_train.csv");
    for(int i = 0 ; i < N; i++){
        //std::cout << "Prediction: " << prediction[i] << "   |   Actual: " << Training_set_normalized[i].output << endl; 
    }

    double error = Error_func(prediction, actual, *W_ML);
    double acc = Acc(prediction, actual);
    std::cout << "Total Training Error = " << error <<endl;
    std::cout << "Accuracy = " << acc <<endl;
}

void Model::Test(){
    Testing_set_normalized = Normalize(Testing_set);
    int N = Testing_set_normalized.size();
    int K = Testing_set_normalized[0].input.size();
    vector<double> prediction(N, 0);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                prediction[i] += (*W_ML)(j, k) * phi(Training_set_normalized[i].input[k],j);
            }
        }
    }
    vector<double> actual, x3;
    for(int i = 0; i < Training_set_normalized.size(); i++){
        actual.push_back(Training_set_normalized[i].output);
        x3.push_back(Training_set_normalized[i].input[2]);
    }
    write_file_pred(prediction, actual, x3, "prediction_test.csv");
    for(int i = 0 ; i < N; i++){
        //std::cout << "Prediction: " << prediction[i] << "   |   Actual: " << Training_set_normalized[i].output << endl; 
    }
    double error = Error_func(prediction, actual, *W_ML);
    double acc = Acc(prediction, actual);
    std::cout << "Testing Error = " << error <<endl;
    std::cout << "Accuracy = " << acc <<endl;
}

void Model::Test_5fold(){

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
        for(int j = 0; j < raw_data[i].input.size(); j++){
            normalized[i].input.push_back((raw_data[i].input[j] - mean[j])/std_dev[j]);
            normalized[i].output = raw_data[i].output;
        }
    }
    Mean = mean;
    Sd = std_dev;
    return normalized;
}

double Model::Acc(vector<double> predicted, vector<double> ground_truth){
    double Acc = 0;
    for(int i = 0; i < predicted.size(); i++){
        Acc += fabs((ground_truth[i] - predicted[i])/predicted[i]);
    }
    return (1 - Acc/predicted.size());
}

vector<Eigen::MatrixXd>* Model::generate_D_M(vector<SongData> x){
    vector<Eigen::MatrixXd>* DM = new vector<Eigen::MatrixXd>(11, Eigen::MatrixXd(Training_set_normalized.size(), M));
    //(Training_set_normalized.size(), vector<vector<double>>(M, vector<double>(11))); // [10000, M, 11]

    for(int i = 0; i < DM->size(); i++){
        for(int j = 0; j < x.size(); j++){
            for(int k = 0; k < M; k++){
                (*DM)[i](j,k) = phi(x[j].input[i], k);
            }
        }
    }
    return DM;
}

Eigen::MatrixXd* Model::calculate_W_ML_(vector<Eigen::MatrixXd>* D_M_, vector<SongData> s){
    int N = Training_set_normalized.size();
    int K = Training_set_normalized[0].input.size();
    //reshape the [N,M,K] Matrix into [N,M*K] Matrix

    Eigen::MatrixXd Design_Matrix(N, M * K);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                Design_Matrix(i, j * K + k) = (*D_M_)[k](i, j);
            }
        }
    }
    
    Eigen::MatrixXd y(N, 1);
    for(int i = 0; i < N; i++){
        y(i, 0) = s[i].output;
    }
    //==============Self Defined Pseudo Inverse=========================

    Eigen::MatrixXd ATA = Design_Matrix.transpose() * Design_Matrix;
    ATA += Lamda * Eigen::MatrixXd::Identity(M*K, M*K);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd ATA_inversed = svd.solve(Eigen::MatrixXd::Identity(M*K, M*K));
    //ATA.diagonal().array() += Lamda;
    Eigen::MatrixXd W_ = ATA_inversed * Design_Matrix.transpose() * y;
    Eigen::MatrixXd *W = new Eigen::MatrixXd(M, K);
    //===================================================================
    cout << "Shape of W_: " << W_.rows() << " x " << W_.cols() << endl;
    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            (*W)(i, j) = W_(i*K + j, 0);
        }
    }
    cout << " *W: " << endl;
    cout << *W << endl;
    return W;
}


void Model::write_file_pred(vector<double> y_pred, vector<double> actual, vector<double> x3, string filename){
    ofstream output_file(filename);
    for(int i = 0; i < y_pred.size(); i++){
        output_file << y_pred[i] << "," << actual[i] << "," << x3[i] << endl;
    }
    
    output_file.close();
}

double Model::phi(double x_k, int j)
{
    return j > 0 ?  sigmoid((x_k - (3.*(-M+1+2*((double)j - 1.)*(((double)M-1.)/((double)M-2.)))/(double)M))/0.1) : 1;
}
