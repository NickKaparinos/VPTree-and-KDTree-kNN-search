#include <algorithm>

int partition(double *A, int left, int right){

    double pivot = A[right];
    int i = left, x;

    for (x = left; x < right; x++){
        if (A[x] <= pivot){
            std::swap(A[i], A[x]);
            i++;
        }
    }
    std::swap(A[i], A[right]);
    return i;
}


double quickselect(double *A, double left, double right, int k){
    //p is position of pivot in the partitioned array
    int p = partition(A, left, right);

    //k equals pivot got lucky
    if (p == k-1){
        return A[p];
    }
    //k less than pivot
    else if (k - 1 < p){
        return quickselect(A, left, p - 1, k);
    }
    //k greater than pivot
    else{
        return quickselect(A, p + 1, right, k);
    }
}

double ksmallest(double *A, double n, int k){

    int left = 0;
    int right = n - 1;

    return quickselect(A, left, right, k);
}

double median(double *A, int n){
    if(n%2==1){
        return (double)quickselect(A, 0, n-1, (int)n/2+1);
    }else{
        return (double)(quickselect(A, 0, n-1, n/2) + quickselect(A, 0, n-1, (int)n/2+1))/2 ;
    }
}

void quickSelect2Arrays(int k, double* A, int* B, int left, int right){
    // Performs quickselect on two arrays
    double pivot = A[right];
    int i = left, x;
    for (x = left; x <= right; x++){
        if (A[x] <= pivot){
            std::swap(A[x], A[i]);
            std::swap(B[x], B[i]);
            i++;
        }
    }

    if (i == k + 1) {
        return;
    }else if (i < k + 1) {
        quickSelect2Arrays(k, A, B, i, right);
    }
    else {
        quickSelect2Arrays(k, A, B, left, i - 2);
    }
}