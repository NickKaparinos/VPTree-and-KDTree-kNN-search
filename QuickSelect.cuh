#ifndef QUICKSELECT_CUH
#define QUICKSELECT_CUH

int partition(double *A, int left, int right);

double quickselect(double *A, double left, double right, int k);

double ksmallest(double *A, double n, int k);

double median(double *A, int n);

void quickSelect2Arrays(int k, double* A, int* B, int left, int right);
#endif
