#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include<vector>
using namespace std;

void readfiles(vector<double>&column, string filename) // Function to read the csv file for the source input
{
	
	const int COLUMNS = 1;
	vector< vector <double> > data;
	ifstream ifile(filename.c_str());

	if (ifile.is_open()) 
    {
        double num;
	vector <double> numbers_in_line;
	while (ifile >> num) 
	{
            numbers_in_line.push_back(num);
	    if (numbers_in_line.size() == COLUMNS) 
	    {
                data.push_back(numbers_in_line);
                numbers_in_line.clear();
            }
        }
    }
    	else 
	{
        	cout << "There was an error opening the input file!\n";
        	exit(1);
   	}

    //now get the column from the 2d vector:
    //example: the 2nd column
    int col = 1;

    for (int i = 0; i < data.size(); ++i) 
    {
        column.push_back(data[i][col - 1]);
    }

    ifile.close();
}
double calc_square_error ( double *W,  double *Wold,  int No,  int Ni)
{
	int i, j;
	double error = 0;
	for (i = 0; i < No; i++) {
		for (j = 0; j < Ni; j++) error += pow (W[Ni * i + j] - Wold[Ni * i + j], 2);
	}
	return error / No / Ni;
}

int save_data ( char *filename,  double *data,  int M,  int N)
{
	int i, j;
	FILE *fp;
	if ((fp = fopen (filename, "w")) == NULL) return 1;
	fprintf (fp, "num");
	for (j = 0; j < N; j++) fprintf (fp, ", x%d", j);
	fprintf (fp, "\n");
	for (i = 0; i < M; i++) {
		fprintf (fp, "%d", i);
		for (j = 0; j < N; j++) fprintf (fp, ", %f", data[N * i + j]);
		fprintf (fp, "\n");
	}
	fclose (fp);
	return 0;
}

#define H_DIM 40
#define H_ELE(hist, a, b, c, d) hist[H_DIM * (H_DIM * (H_DIM * a + b) + c) + d]

int hist_add (double *hist,  double *u)
{
	 double scale = H_DIM / 2.0 / 6.0;
	 int offset = H_DIM / 2;
	double a[4] = {scale * u[0] + offset, scale * u[1] + offset, scale * u[2] + offset, scale * u[3] + offset};
	if (   0 <= a[0] && a[0] < H_DIM && 0 <= a[1] && a[1] < H_DIM
		&& 0 <= a[2] && a[2] < H_DIM && 0 <= a[3] && a[3] < H_DIM) {
		int db=int(H_DIM*(H_DIM*(H_DIM*a[0]+a[1])+a[2])+a[3]);
		hist[db]+=1;
	}
		
		//H_ELE(hist, a[0], a[1], a[2], a[3])++;
	return 0;
}

int hist_normalize (double *hist)
{
	int i, j, k, l;
	double sum = 0;
	for (i = 0; i < H_DIM; i++) {
		for (j = 0; j < H_DIM; j++) {
			for (k = 0; k < H_DIM; k++) {
				for (l = 0; l < H_DIM; l++) sum += H_ELE(hist, i, j, k, l);
			}
		}
	}
	for (i = 0; i < H_DIM; i++) {
		for (j = 0; j < H_DIM; j++) {
			for (k = 0; k < H_DIM; k++) {
				for (l = 0; l < H_DIM; l++) H_ELE(hist, i, j, k, l) /= sum;
			}
		}
	}
	return 0;
}

int calc_corr (double *corr,  double *source,  double *output,  int Ns,  int No,  int M)
{
	int i, j, t;
	double mean_s[Ns];    memset (mean_s, 0, sizeof (mean_s));
	double mean_u[No];    memset (mean_u, 0, sizeof (mean_u));
	double stdv_s[Ns];    memset (stdv_s, 0, sizeof (stdv_s));
	double stdv_u[No];    memset (stdv_u, 0, sizeof (stdv_u));
	double cov[No * Ns];  memset (cov,    0, sizeof (cov));
	for (t = 0; t < M; t++) {
		for (i = 0; i < No; i++) {
			mean_u[i] += output[No * t + i];
			stdv_u[i] += output[No * t + i] * output[No * t + i];
		}
		for (j = 0; j < Ns; j++) {
			mean_s[j] += source[Ns * t + j];
			stdv_s[j] += source[Ns * t + j] * source[Ns * t + j];
		}
		for (i = 0; i < No; i++) {
			for (j = 0; j < Ns; j++) cov[Ns * i + j] += output[No * t + i] * source[Ns * t + j];
		}
	}
	for (i = 0; i < No; i++) mean_u[i] /= M;
	for (j = 0; j < Ns; j++) mean_s[j] /= M;
	for (i = 0; i < No; i++) stdv_u[i] = sqrt (stdv_u[i] / M - mean_u[i] * mean_u[i]);
	for (j = 0; j < Ns; j++) stdv_s[j] = sqrt (stdv_s[j] / M - mean_s[j] * mean_s[j]);
	for (i = 0; i < No; i++) {
		for (j = 0; j < Ns; j++) cov[Ns * i + j] = cov[Ns * i + j] / M - mean_u[i] * mean_s[j];
	}
	for (i = 0; i < No; i++) {
		for (j = 0; j < Ns; j++) corr[Ns * i + j] = cov[Ns * i + j] / stdv_u[i] / stdv_s[j];
	}
	return 0;
}



int rescale_output (double *u,  double *scale,  int No)
{
	int i;
	for (i = 0; i < No; i++) u[i] *= scale[i];
	return 0;
}
