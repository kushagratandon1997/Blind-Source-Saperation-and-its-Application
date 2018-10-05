#include <bits/stdc++.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "scirep_mathematics.h"
#include "scirep_mathematics2.h"
#include "scirep_functions.h"
#include "scirep_cost.h"
using namespace std;

#define COMPARISON 1 // 0; For Fig. 2AB, 1; For Fig. 2CD

int main (int argc, char **argv)
{
	if (argc != 3) return 1;
	int i, j, k, t;
	const int Ns = 3;      // Number of sources
	const int Ni = Ns;      // Number of inputs
	const int No = 3;       // Number of outputs
	const int T = 20000000; // Simulation time
        const int z = 120000; // number of input values for each dataset
	const double sqrt2 = sqrt (2.0), sqrt3 = sqrt (3.0), sqrt12 = 2.0 * sqrt (3.0);
	
	const int seed = atoi (argv[2]);
	srandom (1000000 + seed);
	for (i = 0; i < 100000; i++) random ();
	
	//--------------------
	// Generative process
	vector<double> array;
	double *s = (double *) malloc (sizeof(double)* Ns * z); 
	double *x = (double *) malloc (sizeof(double)* Ni * z);
    double A[Ni * Ns], R[Ns * Ns], L[Ns * Ns];

	mat_id (A,Ns);
	
	for (i = 0; i < Ns; i++) 
	{
		for (j = 0; j < Ns; j++) 
		{
			if (i != j) mat_rotation (A, A, Ns, i, j, 2.0 * M_PI * random () / RAND_MAX);
		}
	}
	printf ("A\n"); 
	mat_print (A, Ni, Ns);
	
	for(int mt=0; mt<z; mt++) mat_mul (&x[Ni*mt], A, &s[Ns*mt], Ns, 1, Ns);
	
	double AT[Ns * Ni], ATA[Ns * Ns];
	mat_trans (AT, A, Ni, Ns);
	mat_mul (ATA, AT, A, Ns, Ns, Ni);
	printf ("ATA\n"); mat_print (ATA, Ns, Ns);
	
	//--------------------
	// Neural networks
	const double eta = 0.000008;
#if COMPARISON == 0
	double beta = 0;   // For EGHR
	if (atof (argv[1]) > 0) beta = pow (10, (atof (argv[1]) - 1.0) / 10.0) / 1000.0;
	printf ("beta = %f\n", beta);
#elif COMPARISON == 1
	const double beta  = 0;   // For EGHR
	const double beta2 = 0.8; // For EGHR2
#endif
	
	// For EGHR
	double u[No * z], g[No], E;
	double W[No * Ni], K[No * Ns];
	double EGX[No * Ni]; memset (EGX, 0, sizeof (EGX));
	double x2_mean = 0, u2_mean = 0, Ex, Egi, Eu, E_mean = 0;
	double Wold[No * Ni]; memcpy (Wold, W, sizeof (double) * No * Ni);
	// Initial state of W
	for (i = 0; i < No; i++) 
	{
		for (j = 0; j < Ni; j++) W[Ni * i + j] = 0.5 * nrandom ();     // Random
//		for (j = 0; j < Ni; j++) W[Ni * i + j] = (i == j) ? (1) : (0); // Identical
//		for (j = 0; j < Ni; j++) W[Ni * i + j] = A[Ni * j + i];        // Inverse
	}

#if COMPARISON == 1
	// For EGHR2
	double W2[No * Ni], K2[No * Ns];
	double EGX2[No * Ni]; memset (EGX2, 0, sizeof (EGX2));
	double u2_mean2 = 0, E_mean2 = 0;
	// For Oja
	double y[Ni], Wt[Ni * No];
	double Woja[No * Ni]; mat_id2 (Woja, No, Ni, 0.2);
	double GX_GY[No * Ni]; memset (GX_GY, 0, sizeof (GX_GY));
	double Koja[No * Ns];
	// For Amari
	double Wamari[No * Ni]; mat_id2 (Wamari, No, Ni, 0.2);
	double GY[No * Ni]; memset (GY, 0, sizeof (GY));
	double Kamari[No * Ns];
	// For Oja-->Amari
	double uoja[No];
	double Wamari2[No * No]; mat_id2 (Wamari2, No, No, 0.2);
	double GY2[No * No]; memset (GY2, 0, sizeof (GY2));
	double Kamari2[No * Ns];
	// For random
	double Wrnd[No * Ni]; mat_id2 (Wrnd, No, Ni, 0.2);
	double Krnd[No * Ns];
	// Initial state of W
	memcpy (W2,     W, sizeof (double) * No * Ni);
	memcpy (Woja,   W, sizeof (double) * No * Ni);
	memcpy (Wamari, W, sizeof (double) * No * Ni);
	memcpy (Wrnd,   W, sizeof (double) * No * Ni);
	mat_id2 (Wamari2, No, No, 1.0);
#endif
	
	//--------------------
	double u_var[No];        memset (u_var,        0, sizeof (u_var));
#if COMPARISON == 1
	double u2_var[No];       memset (u2_var,       0, sizeof (u2_var));
	double uoja_var[No];     memset (uoja_var,     0, sizeof (uoja_var));
	double uamari_var[No];   memset (uamari_var,   0, sizeof (uamari_var));
	double uamari2_var[No];  memset (uamari2_var,  0, sizeof (uamari2_var));
	double urnd_var[No];     memset (urnd_var,     0, sizeof (urnd_var));
#endif
	double u_kurt[No];       memset (u_kurt,       0, sizeof (u_kurt));
#if COMPARISON == 1
	double u2_kurt[No];      memset (u2_kurt,      0, sizeof (u2_kurt));
	double uoja_kurt[No];    memset (uoja_kurt,    0, sizeof (uoja_kurt));
	double uamari_kurt[No];  memset (uamari_kurt,  0, sizeof (uamari_kurt));
	double uamari2_kurt[No]; memset (uamari2_kurt, 0, sizeof (uamari2_kurt));
	double urnd_kurt[No];    memset (urnd_kurt,    0, sizeof (urnd_kurt));
#endif
	
	//--------------------
	//Source reading

	readfiles(array,"natural_img.csv"); 
	readfiles(array,"high_var.csv"); 
	readfiles(array,"low_var.csv");

	for(long long int itr=0;itr<360000;itr++)
      	{
		s[itr]=array[itr];
	}
		
	//Input

	mat_mul (x, A, s, Ni, 1, Ns);
	mat_mul (K, W, A, No, Ns, Ni);
	printf ("W\n"); mat_print (W, No, Ni);
	printf ("K\n"); mat_print (K, No, Ns);
	printf ("\n");

	for (i = 0, Ex = 0; i < Ni; i++) Ex += x[i] * x[i] / 2.0;
		x2_mean += eta * 10 * (-x2_mean + Ex);   Ex = Ex - x2_mean;
		
		//--------------------
		// EGHR
		mat_mul (u, W, x, No, 1, Ni);
		for (i = 0, E = 0, Eu = 0; i < No; i++) 
		{
//			E += fabs (u[i]) * sqrt2;           g[i] = tanh (u[i] * 100) * sqrt2;     // For Laplace
//			E += u[i] * u[i] / 2.0;             g[i] = u[i];                          // For Gaussian
			E += double_softmax(u[i]) / sqrt12; g[i] = double_sigmoid(u[i]) / sqrt12; // For Uniform
			Eu += u[i] * u[i] / 2.0;
		}
		E_mean  += eta * 10 * (-E_mean  + E );   E  = E  - E_mean;
		u2_mean += eta * 10 * (-u2_mean + Eu);   Eu = Eu - u2_mean;
		for (i = 0, k = 0; i < No; i++) 
		{
			Egi = (1.0 - E) * g[i] * (1.0 - beta) + (Ex - Eu) * u[i] * beta;
			for (j = 0; j < Ni; j++, k++) 
			{
				EGX[k] += eta * 10 * (-EGX[k] + Egi  * x[j] );
				W[k]   += eta *      ( EGX[k]);
			}
		}
		for (i = 0; i < No; i++) u_var[i]  += eta * (-u_var[i]  + pow (u[i], 2.0));
		for (i = 0; i < No; i++) u_kurt[i] += eta * (-u_kurt[i] + pow (u[i], 4.0));
			printf ("u_var        %0.9f, %0.9f, %0.9f\n", u_var[0],        u_var[1],        u_var[2] );
			printf ("u_kurt       %0.9f, %0.9f, %0.9f\n", u_kurt[0],       u_kurt[1],       u_kurt[2] );

	
	// Run the simulation

	printf ("\n");
	for (t = 0; t < T; t++) 
	{
		if (t % 100000 == 0) 
		{
			printf ("\x1b[1At = %d, %f, %f, %f\n", t, x2_mean, u2_mean, sqrt (calc_square_error (W, Wold, No, Ni)));
			memcpy (Wold, W, sizeof (double) * No * Ni);
		}
		
#if COMPARISON == 1
		//--------------------
		// EGHR2
		mat_mul (u, W2, x, No, 1, Ni);
		for (i = 0, E = 0, Eu = 0; i < No; i++) {
			E += double_softmax(u[i]) / sqrt12; g[i] = double_sigmoid(u[i]) / sqrt12; // For Uniform
			Eu += u[i] * u[i] / 2.0;
		}
		E_mean2  += eta * 10 * (-E_mean2  + E );   E  = E  - E_mean2;
		u2_mean2 += eta * 10 * (-u2_mean2 + Eu);   Eu = Eu - u2_mean2;
		for (i = 0, k = 0; i < No; i++) {
			Egi = (1.0 - E) * g[i] * (1.0 - beta2) + (Ex - Eu) * u[i] * beta2;
			for (j = 0; j < Ni; j++, k++) {
				EGX2[k] += eta * 10 * (-EGX2[k] + Egi  * x[j]  );
				W2[k]   += eta *      ( EGX2[k]);
			}
		}
		for (i = 0; i < No; i++) u2_var[i]  += eta * (-u2_var[i]  + pow (u[i], 2.0));
		for (i = 0; i < No; i++) u2_kurt[i] += eta * (-u2_kurt[i] + pow (u[i], 4.0));
		
		//--------------------
		// Oja
		mat_mul (u, Woja, x, No, 1, Ni);
		mat_trans (Wt, Woja, No, Ni);
		mat_mul (y, Wt, u, Ni, 1, No);
		for (i = 0, k = 0; i < No; i++) 
		{
			for (j = 0; j < Ni; j++, k++) 
			{
				GX_GY[k] += eta * 100 * (-GX_GY[k] + u[i] * x[j] - u[i] * y[j]);
				Woja[k]  += eta *       ( GX_GY[k]                            );
			}
		}
		memcpy (uoja, u, sizeof (double) * No);
		for (i = 0; i < No; i++) uoja_var[i]  += eta * (-uoja_var[i]  + pow (u[i], 2.0));
		for (i = 0; i < No; i++) uoja_kurt[i] += eta * (-uoja_kurt[i] + pow (u[i], 4.0));
		
		//--------------------
		// Amari
		mat_mul (u, Wamari, x, No, 1, Ni);
		mat_trans (Wt, Wamari, No, Ni);
		mat_mul (y, Wt, u, Ni, 1, No);
		for (i = 0; i < No; i++) g[i] = double_sigmoid(u[i]) / sqrt12; // For Uniform
		for (i = 0, k = 0; i < No; i++) {
			for (j = 0; j < Ni; j++, k++) {
				GY[k]     += eta * 10 * (-GY[k]     + g[i] * y[j]);
				Wamari[k] += eta *      ( Wamari[k] - GY[k]      );
			}
		}
		for (i = 0; i < No; i++) uamari_var[i]  += eta * (-uamari_var[i]  + pow (u[i], 2.0));
		for (i = 0; i < No; i++) uamari_kurt[i] += eta * (-uamari_kurt[i] + pow (u[i], 4.0));
		
		//--------------------
		// Oja-->Amari
		mat_mul (u, Wamari2, uoja, No, 1, No);
		mat_trans (Wt, Wamari2, No, No);
		mat_mul (y, Wt, u, No, 1, No);
		for (i = 0; i < No; i++) g[i] = double_sigmoid(u[i]) / sqrt12; // For Uniform
		for (i = 0, k = 0; i < No; i++) {
			for (j = 0; j < No; j++, k++) {
				GY2[k]     += eta * 10 * (-GY2[k]     + g[i] * y[j]);
				Wamari2[k] += eta *      ( Wamari2[k] - GY2[k]     );
			}
		}
		for (i = 0; i < No; i++) uamari2_var[i]  += eta * (-uamari2_var[i]  + pow (u[i], 2.0));
		for (i = 0; i < No; i++) uamari2_kurt[i] += eta * (-uamari2_kurt[i] + pow (u[i], 4.0));
		
		//--------------------
		// Random
		mat_mul (u, Wrnd, x, No, 1, Ni);
		for (i = 0; i < No; i++) urnd_var[i]  += eta * (-urnd_var[i]  + pow (u[i], 2.0));
		for (i = 0; i < No; i++) urnd_kurt[i] += eta * (-urnd_kurt[i] + pow (u[i], 4.0));
#endif
	}
	
	
#if COMPARISON == 1
	mat_mul (K2, W2, A, No, Ns, Ni);
	printf ("W2\n"); mat_print (W2, No, Ni);
	printf ("K2\n"); mat_print (K2, No, Ns);
	printf ("\n");
	
	mat_mul (Koja, Woja, A, No, Ns, Ni);
	printf ("Woja\n"); mat_print (Woja, No, Ni);
	printf ("Koja\n"); mat_print (Koja, No, Ns);
	printf ("\n");
	
	mat_mul (Kamari, Wamari, A, No, Ns, Ni);
	printf ("Wamari\n"); mat_print (Wamari, No, Ni);
	printf ("Kamari\n"); mat_print (Kamari, No, Ns);
	printf ("\n");
	
	double Wamarioja[No * Ni]; mat_mul (Wamarioja, Wamari2, Woja, No, Ni, No);
	double Kamarioja[No * Ns];
	mat_mul (Kamarioja, Wamarioja, A, No, Ns, Ni);
	printf ("Wamarioja\n"); mat_print (Wamarioja, No, Ni);
	printf ("Kamarioja\n"); mat_print (Kamarioja, No, Ns);
	printf ("\n");
	
	mat_mul (Krnd, Wrnd, A, No, Ns, Ni);
#endif
	
	// Variance and kurtosis
	for (i = 0; i < No; i++) 
	{
		u_kurt[i]       = u_kurt[i]       / pow (u_var[i],       2.0) - 3.0;
#if COMPARISON == 1
		u2_kurt[i]      = u2_kurt[i]      / pow (u2_var[i],      2.0) - 3.0;
		uoja_kurt[i]    = uoja_kurt[i]    / pow (uoja_var[i],    2.0) - 3.0;
		uamari_kurt[i]  = uamari_kurt[i]  / pow (uamari_var[i],  2.0) - 3.0;
		uamari2_kurt[i] = uamari2_kurt[i] / pow (uamari2_var[i], 2.0) - 3.0;
		urnd_kurt[i]    = urnd_kurt[i]    / pow (urnd_var[i],    2.0) - 3.0;
#endif
	}
#if COMPARISON == 1
	printf ("u2_var       %6.3f, %6.3f, %6.3f\n", u2_var[0],       u2_var[1],       u2_var[2]            );
	printf ("uoja_var     %6.3f, %6.3f, %6.3f\n", uoja_var[0],     uoja_var[1],     uoja_var[2]          );
	printf ("uamari_var   %6.3f, %6.3f, %6.3f\n", uamari_var[0],   uamari_var[1],   uamari_var[2]        );
	printf ("uamari2_var  %6.3f, %6.3f, %6.3f\n", uamari2_var[0],  uamari2_var[1],  uamari2_var[2]       );
	printf ("urnd_var     %6.3f, %6.3f, %6.3f\n", urnd_var[0],     urnd_var[1],     urnd_var[2]          );
#endif
	
#if COMPARISON == 1
	printf ("u2_kurt      %6.3f, %6.3f, %6.3f\n", u2_kurt[0],      u2_kurt[1],      u2_kurt[2]           );
	printf ("uoja_kurt    %6.3f, %6.3f, %6.3f\n", uoja_kurt[0],    uoja_kurt[1],    uoja_kurt[2]         );
	printf ("uamari_kurt  %6.3f, %6.3f, %6.3f\n", uamari_kurt[0],  uamari_kurt[1],  uamari_kurt[2]       );
	printf ("uamari2_kurt %6.3f, %6.3f, %6.3f\n", uamari2_kurt[0], uamari2_kurt[1], uamari2_kurt[2]      );
	printf ("urnd_kurt    %6.3f, %6.3f, %6.3f\n", urnd_kurt[0],    urnd_kurt[1],    urnd_kurt[2]         );
#endif
	
	// Evaluation
	const int M = 20000000;
	double *source = (double *) malloc (sizeof (double) * Ns * M);
	double *input  = (double *) malloc (sizeof (double) * Ni * M);
	double *output = (double *) malloc (sizeof (double) * No * M);
	char filename[1000];
#if COMPARISON == 1
	double *hist       = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double *hist2      = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double *histoja    = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double *histamari  = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double *histamari2 = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double *histrnd    = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double *histopt    = (double *) malloc (sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (hist,       0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (hist2,      0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (histoja,    0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (histamari,  0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (histamari2, 0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (histrnd,    0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	memset (histopt,    0, sizeof (double) * H_DIM * H_DIM * H_DIM * H_DIM);
	double scale[No], scale2[No], scaleoja[No], scaleamari[No], scaleamari2[No], scalernd[No];
	for (i = 0; i < No; i++) {
		scale[i]       = 1.0 / sqrt (u_var[i]      );
		scale2[i]      = 1.0 / sqrt (u2_var[i]     );
		scaleoja[i]    = 1.0 / sqrt (uoja_var[i]   );
		scaleamari[i]  = 1.0 / sqrt (uamari_var[i] );
		scaleamari2[i] = 1.0 / sqrt (uamari2_var[i]);
		scalernd[i]    = 1.0 / sqrt (urnd_var[i]   );
	}
#endif
	
	printf ("\n");
	for (t = 0; t < M; t++) {
		if (t % 100000 == 0) printf ("\x1b[1At = %d\n", t);
		for (i = 0; i < Ns; i += 2) s[i] = nrandom ();         // Gaussian
		for (i = 1; i < Ns; i += 2) s[i] = urandom () * sqrt3; // Uniform
		mat_mul (x, A, s, Ni, 1, Ns);
		mat_mul (u, W, x, No, 1, Ni);
		memcpy (&source[Ns * t], s, sizeof (double) * Ns);
		memcpy (&input[Ni * t],  x, sizeof (double) * Ni);
		memcpy (&output[No * t], u, sizeof (double) * No);
#if COMPARISON == 1
		rescale_output (u, scale, No); hist_add (hist, u); // For EGHR
		mat_mul (u, W2,        x, No, 1, Ni); rescale_output (u, scale2,      No); hist_add (hist2,      u); // For EGHR2
		mat_mul (u, Woja,      x, No, 1, Ni); rescale_output (u, scaleoja,    No); hist_add (histoja,    u); // For Oja
		mat_mul (u, Wamari,    x, No, 1, Ni); rescale_output (u, scaleamari,  No); hist_add (histamari,  u); // For Amari
		mat_mul (u, Wamarioja, x, No, 1, Ni); rescale_output (u, scaleamari2, No); hist_add (histamari2, u); // For Oja+Amari
		mat_mul (u, Wrnd,      x, No, 1, Ni); rescale_output (u, scalernd,    No); hist_add (histrnd,    u); // For Random
		u[0] = s[1]; u[1] = s[3]; u[2] = s[5]; u[3] = s[7];                        hist_add (histopt,    u); // For Optimal
#endif
	}
	
	double corr[No * Ns]; memset (corr, 0, sizeof (corr));
	calc_corr (corr, source, output, Ns, No, M);
	sprintf (filename, "eghr_corr_%d_%d_%d.csv", COMPARISON, atoi (argv[1]), seed); save_data (filename, corr, No, Ns);
	sprintf (filename, "eghr_K_%d_%d_%d.csv",    COMPARISON, atoi (argv[1]), seed); save_data (filename, K,    No, Ns);
	if (seed == 0 && (atoi (argv[1]) == 0 || atoi (argv[1]) == 30)) {
		sprintf (filename, "data_source_%d_%d_%d.csv", COMPARISON, atoi (argv[1]), seed); save_data (filename, source, 2000, Ns);
		sprintf (filename, "data_input_%d_%d_%d.csv",  COMPARISON, atoi (argv[1]), seed); save_data (filename, input,  2000, Ni);
		sprintf (filename, "data_output_%d_%d_%d.csv", COMPARISON, atoi (argv[1]), seed); save_data (filename, output, 2000, No);
	}
	
#if COMPARISON == 1
	double PCAcost[7], ICAcost[7]; // Cost functions
	double Wopt[No * Ns], Kopt[No * Ns], Ainv[Ni * Ns];
	mat_id2 (Kopt, No, Ns, 1.0); mat_inv (Ainv, A, Ni); mat_mul (Wopt, Kopt, Ainv, No, Ni, Ni);
	hist_normalize (hist      ); ICAcost[0] = calc_ICAcost (hist      ); PCAcost[0] = calc_PCAcost (A, W,         Ni, No);
	hist_normalize (hist2     ); ICAcost[1] = calc_ICAcost (hist2     ); PCAcost[1] = calc_PCAcost (A, W2,        Ni, No);
	hist_normalize (histoja   ); ICAcost[2] = calc_ICAcost (histoja   ); PCAcost[2] = calc_PCAcost (A, Woja,      Ni, No);
	hist_normalize (histamari ); ICAcost[3] = calc_ICAcost (histamari ); PCAcost[3] = calc_PCAcost (A, Wamari,    Ni, No);
	hist_normalize (histamari2); ICAcost[4] = calc_ICAcost (histamari2); PCAcost[4] = calc_PCAcost (A, Wamarioja, Ni, No);
	hist_normalize (histrnd   ); ICAcost[5] = calc_ICAcost (histrnd   ); PCAcost[5] = calc_PCAcost (A, Wrnd,      Ni, No);
	hist_normalize (histopt   ); ICAcost[6] = calc_ICAcost (histopt   ); PCAcost[6] = calc_PCAcost (A, Wopt,      Ni, No);
	free (hist); free (hist2); free (histoja); free (histamari); free (histamari2); free (histrnd); free (histopt);
	printf ("PCAcost %f, %f, %f, %f, %f, %f, %f\n", PCAcost[0], PCAcost[1], PCAcost[2], PCAcost[3], PCAcost[4], PCAcost[5], PCAcost[6]);
	printf ("ICAcost %f, %f, %f, %f, %f, %f, %f\n", ICAcost[0], ICAcost[1], ICAcost[2], ICAcost[3], ICAcost[4], ICAcost[5], ICAcost[6]);
	sprintf (filename, "PCAcost_%d_%d_%d.csv", COMPARISON, atoi (argv[1]), seed); save_data (filename, PCAcost, 1, 7);
	sprintf (filename, "ICAcost_%d_%d_%d.csv", COMPARISON, atoi (argv[1]), seed); save_data (filename, ICAcost, 1, 7);
#endif
	
	free (source); free (input); free (output);
	return 0;
}
