#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include "eigen-3.3.7/Eigen/Dense"

using namespace std;
using namespace Eigen;

const double s = 0.1;
const int K = 11;
double M = 30;
int N = 10000;
vector<double> x[12], phi[12], mu, mean, standard_devation;
vector<vector<double>> d_matrix(N *K, vector<double>(M, 1));
ofstream output;

void read_csv() // input data
{
	ifstream input;
	string line;
	string dis;

	input.open("HW1.csv");

	getline(input, dis);
	while (getline(input, line))
	{
		stringstream ss(line);
		string word;
		for (int i = 0; i < 12; i++)
		{
			getline(ss, word, ',');
			double num = stod(word);
			x[i].push_back(num);
		}
	}
	input.close();
}

void basis_function(double M) // make basis function
{
	double temp, a, ans;
	int k = 1, l = 0;

	// cal mu
	mu.push_back(0);
	for (int j = 1; j < M; j++)
	{
		temp = 3 * (-M + 1 + 2 * (j - 1) * (M - 1) / (M - 2)) / M;
		mu.push_back(temp);
	}

	// cal phi
	for (int i = 0; i < N * K; i++)
	{
		for (int j = 1; j < M; j++)
		{
			a = (x[k][l] - mu[j]) / s;
			ans = 1 / (1 + exp(-a));
			d_matrix[i][j] = ans;
		}
		k++;
		if (k == 12)
		{
			k = 1;
			l++;
		}
	}

	// cal pseudo inverse
	MatrixXd d_temp(110000, 30);

	for (double i = 0; i < N*K; i++)
	{
		for (double j = 0; j < M; j++)
		{
			d_temp(i, j) = d_matrix[i][j];
		}
	}

	MatrixXd d_pseudo_inverse = d_temp.completeOrthogonalDecomposition().pseudoInverse();

	// cal WML
	/*for (int i = 0; i < M; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			double sum = 0;
			for (int k = 0; k < N; k++)
			{
				sum += d_pseudo_inverse(i)(k) * x[k][j];
			}
			d_t_d[i][j] += sum;
		}
	}*/
	
	output.open("output.txt");
	
	for(int i=0;i<N*K;i++)
	{
		for(int j=0;j<M;j++)
		{
			output << d_temp(i, j) << " ";
		}
		output << endl;
	}

}

int main(int argc, char **argv)
{
	read_csv();
	double temp;
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			temp = temp + x[i][j];
		}
		temp = temp / 10000;
		mean.push_back(temp);
		temp = 0;
	}
	// standard devation
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			temp = temp + pow((x[i][j] - mean[i]), 2);
		}
		temp = temp / 9999;
		temp = pow(temp, 0.5);
		standard_devation.push_back(temp);
		temp = 0;
	}
	// normalize
	for (int i = 0; i < 12; i++)
	{
		for (int j = 0; j < 10000; j++)
		{
			x[i][j] = (x[i][j] - mean[i]) / standard_devation[i];
		}
	}

	basis_function(M);


	return 0;
}
