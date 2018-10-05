#include <bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <vector>
using namespace std;


#define double_sigmoid(x) (-2.5 * 5.0 / (1.0 + exp (5.0 * ((x) + sqrt3))) + 2.5 * 5.0 / (1.0 + exp (-5.0 * ((x) - sqrt3))))
#define double_softmax(x) (2.5 * log (exp (-5.0 * ((x) + sqrt3)) + 1.0) + 2.5 * log (exp (5.0 * ((x) - sqrt3)) + 1.0))
#define sigmoid(x) (1.0 / (1.0 + exp (-(x))))

#define NR(x, a) ((log (exp ((a) * (x)) + exp (a)) - log (exp (-(a) * (x)) + exp (a)))/(a))
#define Softmax(x, a) ((log (exp ((a) * (x)) + 1)) / (a))
#define Ramp(x) ((x > 0) ? (x) : (0))

double nrandom ();
double erandom ();
double urandom ();
int mat_trans (double *B,  double *A,  int m,  int n);
int mat_mul (double *C, double *A,double *B ,  int m,  int n,  int l);
int mat_muls (double *B,  double *A,  double b,  int m,  int n);
int mat_inv (double *B,  double *A,  int n);
double mat_logdet ( double *A,  int n);
int mat_id (double *A,  int n);
int mat_id2 (double *A,  int m,  int n,  double scale);
int mat_rotation (double *B,  double *A,  int n,  int i,  int j,  double th);
int mat_print ( double *A,  int m,  int n);
int mat_copy (double *B,  double *A,  int m,  int n);
int mat_eigen (double *L, double *P,  double *C,  int n,  int max_repeat);



double nrandom ()
{
	double n1 = 1.0 * (random () + 1) / RAND_MAX;
	double n2 = 1.0 * random () / RAND_MAX;
	return sqrt (-2 * log (n1)) * sin (2 * M_PI * n2);
}

double erandom ()
{
	return -log (1.0 * (random () + 1) / RAND_MAX) * ((random () < RAND_MAX / 2) ? (1) : (-1));
}

double urandom ()
{
	return (2.0 * random () / RAND_MAX - 1.0);
}
//Matrix transpose
int mat_trans (double *B,  double *A,  int m,  int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) B[m * j + i] = A[n * i + j];
	}
	return 0;
}
//Matrix multiplication
int mat_mul (double *C, double *A,double *B ,  int m,  int n,  int l)
{
	int i, j, k;
	memset (C, 0, sizeof (double) * m * n);
	double c;
	double* a = NULL;
	if (n == 1) {
		for (i = 0; i < m; i++) {
			c = 0;
			a = &A[l * i];
			for (k = 0; k < l; k++) c += a[k] * B[k];
			C[i] = c;
		}
	}
	else {
		for (i = 0; i < m; i++) {
			for (j = 0; j < n; j++) {
				c = 0;
				for (k = 0; k < l; k++) c += A[l * i + k] * B[n * k + j];
				C[n * i + j] = c;
			}
		}
	}
	return 0;
}

int mat_muls (double *B,  double *A,  double b,  int m,  int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) B[n * i + j] = A[n * i + j] * b;
	}
	return 0;
}

int mat_inv (double *B,  double *A,  int n)
{
	double *C = (double *) malloc (sizeof (double) * n * n);
	double buf;
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			B[n * i + j] = (i == j) ? (1) : (0);
			C[n * i + j] = A[n * i + j];
		}
	}
	for (i = 0; i < n; i++) {
		buf = 1.0 / C[n * i + i];
        if (C[n * i + i] == 0) printf ("error: in mat_inv divided by 0 (row %d)\n", i);
		for (j = 0; j < n; j++) {
			C[n * i + j] *= buf;
			B[n * i + j] *= buf;
		}
		for (j = 0; j < n; j++) {
			if (i != j) {
				buf = C[n * j + i];
				for (k = 0; k < n; k++) {
					C[n * j + k] -= C[n * i + k] * buf;
					B[n * j + k] -= B[n * i + k] * buf;
				}
			}
		}
	}
	free (C);
	return 0;
}

double mat_logdet ( double *A,  int n)
{
	double *B = (double *) malloc (sizeof (double) * n * n);
	double buf;
	double logdet = 0;
	int i, j, k;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) B[n * i + j] = A[n * i + j];
	}
	for (i = 0; i < n; i++) {
		if (B[n * i + i] == 0) {
			printf ("mat_logdet B(%d,%d)=0\n", i, i);
			for (j = i + 1; j < n && B[n * j + i] != 0; j++) {
			}
			if (j == n) {
				logdet += log (0);
				return logdet;
			}
			else {
				for (k = i; k < n; k++) {
					buf = B[n * i + k];
					B[n * i + k] = B[n * j + k];
					B[n * j + k] = buf;
				}
			}
		}
		logdet += log (fabs (B[n * i + i]));
		for (j = i + 1; j < n; j++) {
			buf = B[n * j + i] / B[n * i + i];
			for (k = i; k < n; k++) B[n * j + k] -= buf * B[n * i + k];
		}
	}
	free (B);
	return logdet;
}

int mat_id (double *A,  int n)
{
	int i, j;
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) A[n * i + j] = (i == j) ? (1) : (0);
	}
	return 0;
}

int mat_id2 (double *A,  int m,  int n,  double scale)
{
	int i;
	memset (A, 0, sizeof (double) * m * n);
	if (m > n) {
		for (i = 0; i < n; i++) A[n * i + i] = scale;
	}
	else {
		for (i = 0; i < m; i++) A[n * i + i] = scale;
	}
	return 0;
}

int mat_rotation (double *B,  double *A,  int n,  int i,  int j,  double th)
{
	int k, l;
	double C[n * n]; 
	mat_copy (C, A, n, n);
	double R[n * n]; mat_id (R, n);
	R[n * i + i] =  cos (th); R[n * i + j] = -sin (th);
	R[n * j + i] =  sin (th); R[n * j + j] =  cos (th);
	mat_mul (B, R, C, n, n, n);
	return 0;
}

int mat_print ( double *A,  int m,  int n)
{
	int i, j, k;
	for (i = 0; i < m; i++) {
		for (j = 0, k = 0; j < n; j++) k = (fabs (A[n * i + j]) > fabs (A[n * i + k])) ? (j) : (k);
		for (j = 0; j < n; j++) {
			if (j == k)                          printf ("\x1b[1m\x1b[32m%6.3f\x1b[0m\x1b[39m,", A[n * i + j]);
			else if (fabs (A[n * i + j]) < 0.01) printf ("\x1b[90m%6.3f\x1b[39m,", A[n * i + j]);
			else                                 printf ("%6.3f,", A[n * i + j]);
		}
		printf ("\n");
	}
	return 0;
}

int mat_copy (double *B,  double *A,  int m,  int n)
{
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) B[n * i + j] = A[n * i + j];
	}
	return 0;
}

int mat_eigen (double *L, double *P,  double *C,  int n,  int max_repeat)
{
	// L : eigenvalues
	// P : eigenvectors
	// C should be a symmetrical matrix
	int i, j;
	double *A = (double *) malloc (sizeof (double) * n * n);
	memcpy (A, C, sizeof (double) * n * n);
	mat_id (P, n);
	int t;
	int p, q;
	double max;
	double app, apq, aqp, aqq;
	double alpha, beta, gamma, s, c, temp;
	 double min_error = 0.0001;
	double *vec_p = (double *) malloc (sizeof (double) * n);
	double *vec_q = (double *) malloc (sizeof (double) * n);
	int *j_argmax = (int *) malloc (sizeof (int) * n);
	for (i = 0; i < n; i++) {
		j_argmax[i] = (i + 1) % n;
		for (j = 0; j < n; j++) {
			if (i != j && fabs (A[n * i + j]) > fabs (A[n * i + j_argmax[i]])) j_argmax[i] = j;
		}
	}
	
	for (t = 0; t < max_repeat; t++) {
		max = 0;
		for (i = 0; i < n; i++) {
			if (fabs (A[n * i + j_argmax[i]]) > max) {
				max = fabs (A[n * i + j_argmax[i]]);
				p = i;
				q = j_argmax[i];
			}
		}
		if (t % 1000 == 0) printf ("calc eigen values t = %d, max = %f\n", t, max);
		if (max < min_error) break;
		
		app = A[n * p + p]; apq = A[n * p + q];
		aqp = A[n * q + p]; aqq = A[n * q + q];
		alpha = (app - aqq) / 2;
		beta  = -apq;
		gamma = fabs (alpha) / sqrt (alpha * alpha + beta * beta);
		s = sqrt ((1 - gamma) / 2) * ((alpha * beta > 0) ? (1) : (-1));
		c = sqrt ((1 + gamma) / 2);
		memcpy (vec_p, &A[n * p], sizeof (double) * n);
		memcpy (vec_q, &A[n * q], sizeof (double) * n);
		for (i = 0; i < n; i++) {
			A[n * p + i] = c * vec_p[i] - s * vec_q[i];
			A[n * q + i] = s * vec_p[i] + c * vec_q[i];
		}
		for (i = 0; i < n; i++) {
			A[n * i + p] = A[n * p + i];
			A[n * i + q] = A[n * q + i];
		}
		A[n * p + p] = c * c * app - c * s * apq - s * c * aqp + s * s * aqq;
		A[n * p + q] = c * s * app + c * c * apq - s * s * aqp - s * c * aqq;
		A[n * q + p] = s * c * app - s * s * apq + c * c * aqp - c * s * aqq;
		A[n * q + q] = s * s * app + s * c * apq + c * s * aqp + c * c * aqq;
		
		for (i = 0; i < n; i++) {
			if (p != i && fabs (A[n * i + p]) > fabs (A[n * i + j_argmax[i]])) j_argmax[i] = p;
			if (q != i && fabs (A[n * i + q]) > fabs (A[n * i + j_argmax[i]])) j_argmax[i] = q;
		}
		for (j = 0; j < n; j++) {
			if (p != j && fabs (A[n * p + j]) > fabs (A[n * p + j_argmax[p]])) j_argmax[p] = j;
			if (q != j && fabs (A[n * q + j]) > fabs (A[n * q + j_argmax[q]])) j_argmax[q] = j;
		}
		
		memcpy (vec_p, &P[n * p], sizeof (double) * n);
		memcpy (vec_q, &P[n * q], sizeof (double) * n);
		for (i = 0; i < n; i++) {
			P[n * p + i] = c * vec_p[i] - s * vec_q[i];
			P[n * q + i] = s * vec_p[i] + c * vec_q[i];
		}
	}
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) L[n * i + j] = (i == j) ? (A[n * i + j]) : (0);
	}
	for (i = 0; i < n; i++) {
		for (j = i + 1; j < n; j++) {
			temp = P[n * i + j];
			P[n * i + j] = P[n * j + i];
			P[n * j + i] = temp;
		}
	}
	free (A); free (vec_p); free (vec_q); free (j_argmax);
	return 0;
}
