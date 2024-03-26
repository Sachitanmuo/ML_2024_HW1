#include"Model.h"

Model::Model(int m, double lamda, string file){
    M = m;
    Lamda = lamda;
    File = file;
    read_file();
}

double Model::sigmoid(double x){
    return 1./(1. + exp(-x));
}
void Model::read_file(){
    ifstream input_file(File);
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
            //getline(iss, token, ',');
            //InputVector.push_back(stod(token));
        }    
        songdata.input = InputVector;
        count++ <= 10000 ? Training_set.push_back(songdata) : Testing_set.push_back(songdata);
    }
}
void Model::read_file(int offset){
    ifstream input_file(File);
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
    }
    for(int i = 0; i < temp.size();i++){
        i < 10000 ? Training_set_5fold[offset].push_back(temp[(int)((i+ temp.size() * (offset/5.)))%temp.size()])
            : Testing_set_5fold[offset].push_back(temp[(int)(i+ temp.size()* (offset/5.))%temp.size()]);
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
        Training_set_5fold_norm[i] = Normalize(Training_set_5fold[i], mean_[i], sd_[i]);
        D_M_[i] = generate_D_M(Training_set_5fold_norm[i]);
        W[i] = calculate_W_ML_(D_M_[i], Training_set_5fold_norm[i]);
        vector<double> y_pred(Training_set_5fold_norm[i].size(), 0);
        vector<double> actual(Training_set_5fold_norm[i].size(), 0);
        vector<double> x3(Training_set_5fold_norm[i].size(), 0);
        for(int x = 0; x < Training_set_5fold_norm[i].size(); x++){
            for(int it = 0; it < M; it++){
                for(int j = 0; j < 11; j++){
                y_pred[x] += (*W[i])(it, j) * phi(Training_set_5fold_norm[i][x].input[j], it);
                }
            }
            actual[x] = Training_set_5fold_norm[i][x].output;
            x3[x] = (Training_set_5fold[i][x].input[2]);
            //cout << y_pred[x] << "|" << actual[x] << endl;
        }
        info_file << "Fold " << i+1<<":"<<endl;
        info_file << "Error: " << Error_func(y_pred, actual) << endl;
        info_file << "Accuracy: " << Acc(y_pred, actual) << endl;
        string o = "fivefold_" + to_string(i+1) + ".csv";
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
double Model::Error_func(vector<double> prediction, vector<double> ground_truth){
    double sum = 0;
    for(int i = 0; i < prediction.size(); i++){
        sum += pow(prediction[i] - ground_truth[i], 2);
    }
    return sum / prediction.size();
}

void Model::Train(){
    Training_set_normalized = Normalize(Training_set, Mean, Sd);
    ofstream out("train_info.txt");
    D_M = generate_D_M(Training_set_normalized);
    
    W_ML = calculate_W_ML_(D_M, Training_set_normalized);    
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

    double error = Error_func(prediction, actual);
    double acc = Acc(prediction, actual);
    std::cout << "Training Error = " << error <<endl;
    std::cout << "Training Accuracy = " << acc <<endl;
    std::cout << "-------------------------------------------" << endl;
    out << "Training Error = " << error <<endl;
    out << "Accuracy = " << acc <<endl;
}

void Model::Test(){
    Testing_set_normalized = Normalize_test(Testing_set, Mean, Sd);//use Training set's mean and standard deviation
    int N = Testing_set_normalized.size();
    int K = Testing_set_normalized[0].input.size();
    ofstream out("Test_Info.txt");
    vector<double> prediction(N, 0);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                prediction[i] += (*W_ML)(j, k) * phi(Testing_set_normalized[i].input[k],j);
            }
        }
    }
    vector<double> actual, x3;
    for(int i = 0; i < Testing_set_normalized.size(); i++){
        actual.push_back(Testing_set_normalized[i].output);
        x3.push_back(Testing_set[i].input[2]);
    }
    write_file_pred(prediction, actual, x3, "prediction_test.csv");
    for(int i = 0 ; i < N; i++){
        //std::cout << "Prediction: " << prediction[i] << "   |   Actual: " << Training_set_normalized[i].output << endl; 
    }
    double error = Error_func(prediction, actual);
    double acc = Acc(prediction, actual);
    std::cout << "Testing Error = " << error <<endl;
    std::cout << "Testing Accuracy = " << acc <<endl;
    std::cout << "===========================================" << endl;
    out << "Testing Error = " << error <<endl;
    out << "Accuracy = " << acc <<endl;
}

void Model::Test_5fold(){
    ofstream info_file("5fold_test_info.txt");
    for(int i = 0; i < 5; i++){
        Testing_set_5fold_norm[i] = Normalize_test(Testing_set_5fold[i], mean_[i], sd_[i]);
        vector<double> y_pred(Testing_set_5fold_norm[i].size(), 0);
        vector<double> actual(Testing_set_5fold_norm[i].size(), 0);
        vector<double> x3(Testing_set_5fold[i].size(), 0);
        for(int x = 0; x < Testing_set_5fold_norm[i].size(); x++){
            for(int it = 0; it < M; it++){
                for(int j = 0; j < 11; j++){
                y_pred[x] += five_fold_W_ML(it, j) *phi(Testing_set_5fold_norm[i][x].input[j], it);
                }
            }
            actual[x] = Testing_set_5fold[i][x].output;
            x3[x] = (Testing_set_5fold[i][x].input[2]);
        }
        if(i == 0){
            std::cout << "5-fold Testing Error: " << Error_func(y_pred, actual) << endl;
            std::cout << "5-fold Testing Accuracy: " << Acc(y_pred, actual) << endl;
            std::cout << "===========================================" << endl;

        }
        info_file << "Fold " << i+1<<":"<<endl;
        info_file << "Error: " << Error_func(y_pred, actual) << endl;
        info_file << "Accuracy: " << Acc(y_pred, actual) << endl;
        string o = "fivefold_test_" + to_string(i+1) + ".csv";
        write_file_pred(y_pred, actual, x3, o);
    }

}

vector<SongData> Model::Normalize_test(vector<SongData>& raw_data, vector<double>& m, vector<double>& std_dev){
    vector<SongData> normalized(raw_data.size());
    for(int i = 0; i < raw_data.size() ; i++){
        for(int j = 0; j < raw_data[i].input.size(); j++){
            normalized[i].input.push_back((raw_data[i].input[j] - m[j])/std_dev[j]);
            normalized[i].output = raw_data[i].output;
        }
    }

    return normalized;
}

void Model::set_M(int m){
    M=m;
}

vector<SongData> Model::Normalize(vector<SongData>& raw_data, vector<double>& m, vector<double>& std){
    vector<double> mean(raw_data[0].input.size(), 0);
    vector<double> std_dev(raw_data[0].input.size(), 0);
    vector<SongData> normalized(raw_data.size());
    //calculate the mean
    for(auto & data : raw_data){
        for(int i = 0; i < data.input.size(); i++){
            mean[i] += data.input[i];
        }
    }
    for(auto& m_ : mean) m_/= 10000;

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
    m = mean;
    std = std_dev;
    return normalized;
}
double Model::Acc(vector<double> predicted, vector<double> ground_truth){
    double Acc = 0;
    for(int i = 0; i < predicted.size(); i++){
        if(predicted[i] == 0){
            Acc += fabs((ground_truth[i] - predicted[i]));
        }
        else{
            Acc += fabs((ground_truth[i] - predicted[i])/predicted[i]);
        }
    }
    return (1 - Acc/predicted.size());
}

vector<Eigen::MatrixXd>* Model::generate_D_M(vector<SongData> x){
    int N = x[0].input.size();
    vector<Eigen::MatrixXd>* DM = new vector<Eigen::MatrixXd>(N, Eigen::MatrixXd(Training_set_normalized.size(), M));
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
    Eigen::MatrixXd W_ = ATA_inversed * Design_Matrix.transpose() * y;
    Eigen::MatrixXd *W = new Eigen::MatrixXd(M, K);
    //===================================================================
    
    for(int i = 0; i < M; i++){
        for(int j = 0; j < K; j++){
            (*W)(i, j) = W_(i*K + j, 0);
        }
    }
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


void Model::demo(string demo_file){
    ifstream input_file(demo_file);
    string line;
    getline(input_file, line);
    int count = 1;
    vector<SongData> demo_set;
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
        demo_set.push_back(songdata);
    }
    std::cout << "Number of datas in the Demo set: " << demo_set.size() << endl;
    //Normalize
    vector<SongData>demo_set_norm;
    demo_set_norm = Normalize_test(demo_set, Mean, Sd);
    
    int N = demo_set_norm.size();
    int K = demo_set_norm[0].input.size();
    ofstream out("Demo_Info.txt");
    vector<double> prediction(N, 0);
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            for(int k = 0; k < K; k++){
                prediction[i] += (*W_ML)(j, k) * phi(demo_set_norm[i].input[k],j);
            }
        }
    }
    vector<double> actual, x3;
    for(int i = 0; i < demo_set_norm.size(); i++){
        actual.push_back(demo_set_norm[i].output);
        x3.push_back(demo_set[i].input[2]);
    }
    write_file_pred(prediction, actual, x3, "prediction_demo.csv");
    double error = Error_func(prediction, actual);
    double acc = Acc(prediction, actual);
    std::cout << "Demo Error = " << error <<endl;
    std::cout << "Demo Accuracy = " << acc <<endl;
    std::cout << "===========================================" << endl;
    out << "Demo Error = " << error <<endl;
    out << "Demo Accuracy = " << acc <<endl;
}

void Model::set_lambda(double l){
    Lamda = l;
}
