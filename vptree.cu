#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vptree.cuh"
#include "QuickSelect.cuh"
#include <cuda_runtime.h>

vptree * buildvp(double *X, int n, int d){
    vptree* T;
    T = (vptree*)malloc(sizeof(vptree));
    T->D = d;
    T->N = n;
    T->A = X;
    T->parent = NULL;
    T->VPCords = (double*) calloc(T->D,sizeof(double));
    T->ivp = T->N-1;
    T->VPCords = (double*) malloc(T->D*sizeof(double));
    T->members = (int*)malloc(n * sizeof(int));
    T->numOfMembers = T->N;

    for(int i=0; i<T->D; i++){
        T->VPCords[i] = *((T->A)+ T->ivp*T->D + i);
    }

    
    for(int i=0; i<T->N; i++){
        T->members[i] = i;
    }

    // Calculate distances between the root and every other point
    double* distancesRoot = calculateDistances(T);

    // distancesCopy is used to calculate the median
    // distancesRoot cannot be used because the median function suffles the rows of the array
    double* distancesCopy = (double*)malloc((T->numOfMembers - 1) * sizeof(double));
    for (int i = 0; i < T->numOfMembers - 1; i++) {
        distancesCopy[i] = distancesRoot[i];
    }

    T->median = median(distancesCopy, T->numOfMembers - 1);
    free(distancesCopy);


    T->inner = buildInner(T, distancesRoot);
    T->outer = buildOuter(T, distancesRoot);
    return T;
}

vptree * getInner(vptree * T){
    return T->inner;
}

vptree * getOuter(vptree * T){
    return T->outer;
}

double getMD(vptree * T){
    return T->median;
}

double * getVP(vptree * T){
    return T->VPCords;
}

int getIDX(vptree * T){
    return T->ivp;
}

vptree* buildInner(vptree* T, double* distances) {
    cudaError_t cudaStatus;
    vptree* inner;
    bool* dev_isInside;
    double* dev_distances;
    vptree* dev_T;
    int* dev_membersInner;
    int* dev_membersT;

    cudaMallocManaged(&inner, sizeof(vptree));
    inner->VPCords = (double*)malloc(T->D * sizeof(double));
    cudaMallocManaged(&dev_membersInner, (T->numOfMembers - 1) * sizeof(int));

    cudaMalloc(&dev_isInside, (T->numOfMembers - 1) * sizeof(bool));
    cudaMalloc(&dev_T, sizeof(vptree));
    cudaMalloc(&dev_distances, (T->numOfMembers - 1) * sizeof(double));
    cudaMalloc(&dev_membersT, (T->numOfMembers - 1) * sizeof(int));

    cudaMemcpy(dev_distances, distances, (T->numOfMembers - 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_membersT, T->members, (T->numOfMembers - 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_T, T, sizeof(vptree), cudaMemcpyHostToDevice);

    // Kernel
    buildInnerKernel <<<1, 1 >>> (dev_T, inner, dev_isInside, dev_distances, dev_membersInner, dev_membersT);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Copy members of the inner subtree calculated by the kernel function
    inner->members = (int*)malloc(inner->numOfMembers * sizeof(int));
    for (int i = 0; i < inner->numOfMembers; i++) {
        inner->members[i] = dev_membersInner[i];
    }

    // Store the cooordinates of the vantage point
    for (int i = 0; i < inner->D; i++) {
        inner->VPCords[i] = *((inner->A) + inner->ivp * T->D + i);
    }


    if (inner->numOfMembers > 2) {
        // Calculate distances and build subtrees
        double* distancesInner = calculateDistances(inner);

        double* distancesCopy = (double*)malloc((inner->numOfMembers - 1) * sizeof(double));
        for (int i = 0; i < inner->numOfMembers - 1; i++) {
            distancesCopy[i] = distancesInner[i];
        }

        inner->median = median(distancesCopy, inner->numOfMembers - 1);
        free(distancesCopy);

        inner->inner = buildInner(inner, distancesInner);
        inner->outer = buildOuter(inner, distancesInner);
    }else if (inner->numOfMembers == 2) {
        // Build leaf node
        vptree* leaf;
        leaf = (vptree*)malloc(sizeof(vptree));
        leaf->parent = inner;
        leaf->A = inner->A;
        leaf->D = inner->D;
        leaf->N = inner->N;
        leaf->inner = NULL;
        leaf->outer = NULL;
        leaf->ivp = inner->members[0];
        leaf->VPCords = (double*)malloc(leaf->D * sizeof(double));

        for (int i = 0; i < leaf->D; i++) {
            leaf->VPCords[i] = *((leaf->A) + leaf->ivp * T->D + i);
        }

        leaf->members = (int*)malloc(sizeof(int));
        leaf->members[0] = inner->members[0];
        leaf->numOfMembers = 1;

        inner->inner = leaf;
        inner->outer = NULL;

        double distance = 0.0;
        for(int j=0; j<T->D; j++){
            distance+=pow(inner->VPCords[j]-*((T->A)+inner->members[0]*T->D+j),2.0);
        }
        distance = sqrt(distance);
        inner->median = distance;
    }else if (inner->numOfMembers == 1) {
        // inner is a leaf node
        inner->inner = NULL;
        inner->outer = NULL;
        inner->median = 0;
    }
    return inner;
}

__global__ void buildInnerKernel(vptree* T, vptree* inner,bool* isInside, double* distances, int* membersInner, int* membersT) {
    inner->parent = T;
    inner->A = T->A;
    inner->D = T->D;
    inner->N = T->N;

    int numPointsInside = 0;

    for (int i = 0; i < T->numOfMembers - 1; i++) { // find points that are inside the radius
        isInside[i] = distances[i] <= T->median;    //check if inside radious
        if (isInside[i]) {
            numPointsInside++;
        }
    }

    if (numPointsInside == T->numOfMembers - 1) {                 // if all points are inside the radius, flag half of them to be outside
        int nfloor = numPointsInside / 2;                         // only happens on the rare case that all the points are equidistant from the vantage point
        for (int i = nfloor; i < numPointsInside; i++) {
            isInside[i] = false;
        }
        inner->numOfMembers = nfloor;
    }
    else {
        inner->numOfMembers = numPointsInside;
    }

    int counter = 0;
    for (int i = 0; i < T->numOfMembers - 1; i++) {       //add members to array
        if (isInside[i]) {
            membersInner[counter] = membersT[i];
            counter++;
        }
    }

    inner->ivp = membersInner[inner->numOfMembers - 1];
}


vptree* buildOuter(vptree* T, double* distances) {
    cudaError_t cudaStatus;
    vptree* outer;
 
    bool* dev_isInside;
    double* dev_distances;
    vptree* dev_T;
    int* dev_membersOuter;
    int* dev_membersT;

    cudaMallocManaged(&outer, sizeof(vptree));
    outer->VPCords = (double*)malloc(T->D * sizeof(double));
    cudaMallocManaged(&dev_membersOuter, (T->numOfMembers - 1) * sizeof(int));

    cudaMalloc(&dev_isInside, (T->numOfMembers - 1) * sizeof(bool));
    cudaMalloc(&dev_T, sizeof(vptree));
    cudaMalloc(&dev_distances, (T->numOfMembers - 1) * sizeof(double));
    cudaMalloc(&dev_membersT, (T->numOfMembers - 1) * sizeof(int));

    cudaMemcpy(dev_distances, distances, (T->numOfMembers - 1) * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_membersT, T->members, (T->numOfMembers - 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_T, T, sizeof(vptree), cudaMemcpyHostToDevice);

    // Kernel
    buildOuterKernel <<<1, 1 >>> (dev_T, outer, dev_isInside, dev_distances, dev_membersOuter, dev_membersT);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Copy members of the outer subtree calculated by the kernel function
    outer->members = (int*)malloc(outer->numOfMembers * sizeof(int));
    for (int i = 0; i < outer->numOfMembers; i++) {
        outer->members[i] = dev_membersOuter[i];
    }

    // Store the cooordinates of the vantage point
    for (int i = 0; i < outer->D; i++) {
        outer->VPCords[i] = *((outer->A) + outer->ivp * T->D + i);
    }

    if (outer->numOfMembers > 2) {
        // Calculate distances and build subtrees
        double* distancesOuter = calculateDistances(outer);

        double* distancesCopy = (double*)malloc((outer->numOfMembers - 1) * sizeof(double));
        for (int i = 0; i < outer->numOfMembers - 1; i++) {
            distancesCopy[i] = distancesOuter[i];
        }

        outer->median = median(distancesCopy, outer->numOfMembers - 1);
        free(distancesCopy);

        outer->inner = buildInner(outer, distancesOuter);
        outer->outer = buildOuter(outer, distancesOuter);
    }
    else if (outer->numOfMembers == 2) {
        // Build leaf node
        vptree* leaf;
        leaf = (vptree*)malloc(sizeof(vptree));
        leaf->parent = outer;
        leaf->A = outer->A;
        leaf->D = outer->D;
        leaf->N = outer->N;
        leaf->inner = NULL;
        leaf->outer = NULL;
        leaf->ivp = outer->members[0];
        leaf->VPCords = (double*)malloc(leaf->D * sizeof(double));
        for (int i = 0; i < leaf->D; i++) {  //Get VPcords
            leaf->VPCords[i] = *((leaf->A) + leaf->ivp * T->D + i);
        }
        leaf->members = (int*)malloc(sizeof(int));
        leaf->members[0] = outer->members[0];
        leaf->numOfMembers = 1;

        outer->inner = leaf;
        outer->outer = NULL;
        double distance = 0.0;
        for (int j = 0; j < T->D; j++) {
            distance += pow(outer->VPCords[j] - *((T->A) + outer->members[0] * T->D + j), 2.0);
        }
        distance = sqrt(distance);
        outer->median = distance;
    }
    else if (outer->numOfMembers == 1) {
        // outer is a leaf node
        outer->inner = NULL;
        outer->outer = NULL;
    }
    return outer;
}

__global__ void buildOuterKernel(vptree* T, vptree* outer, bool* isInside, double* distances, int* membersOuter, int* membersT) {
    outer->parent = T;
    outer->A = T->A;
    outer->D = T->D;
    outer->N = T->N;

    int numPointsOutside = 0;

    for (int i = 0; i < T->numOfMembers - 1; i++) { // find points that are inside the radius
        isInside[i] = distances[i] <= T->median;    //check if inside radious
        if (!isInside[i]) {
            numPointsOutside++;
        }
    }

    if (numPointsOutside == 0) {                                   // if all points are inside the radius, flag half of them to be outside
        numPointsOutside = T->numOfMembers - 1;                    // only happens when all the points are equidistant from the vantage point
        int nfloor = numPointsOutside / 2;
        for (int i = nfloor; i < numPointsOutside; i++) {
            isInside[i] = false;
        }
        outer->numOfMembers = numPointsOutside - nfloor;
    }
    else {
        outer->numOfMembers = numPointsOutside;
    }

    int counter = 0;
    for (int i = 0; i < T->numOfMembers - 1; i++) {       //add members to array
        if (!isInside[i]) {
            membersOuter[counter] = membersT[i];
            counter++;
        }
    }

    outer->ivp = membersOuter[outer->numOfMembers - 1];
}

double* calculateDistances(vptree* T) {
    // Calculate the distances between of the vantage point of the vptree T and every other point
    // The calculation is done using the gpu
    cudaError_t cudaStatus;
    double* distances;
    double* vp;

    cudaMallocManaged(&distances, (T->numOfMembers - 1) * sizeof(double));
    cudaMallocManaged(&vp, T->D * sizeof(double));

    for (int i = 0; i < T->numOfMembers - 1; i++) {
        distances[i] = 0;
    }
    
    for (int i = 0; i < T->D; i++) {
        vp[i] = T->VPCords[i];
    }

    int* dev_D;
    double* dev_A;
    int* dev_members;
    int* dev_numOfMembers;
    int* dev_threads;

    const int sizeA = T->N * T->D * sizeof(double);
    const int sizeMembers = T->numOfMembers * sizeof(int);
    const int threads = T->numOfMembers - 1;

    cudaMalloc(&dev_D, sizeof(int));
    cudaMalloc(&dev_A, sizeA);
    cudaMalloc(&dev_members, sizeMembers);
    cudaMalloc(&dev_numOfMembers, sizeof(int));
    cudaMalloc(&dev_threads, sizeof(int));

    cudaMemcpy(dev_D, &T->D, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A, T->A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_members, T->members, sizeMembers, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_numOfMembers, &T->numOfMembers, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_threads, &threads, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate kernel launch parameters
    int threadsPerBlock;
    int numBlocks;

    // threadsPerBlock cannot exceed 1024
    if (threads > 1024) {
        threadsPerBlock = 1024;
        numBlocks = ceil(float(threads) / 1024);
    }
    else {
        threadsPerBlock = threads;
        numBlocks = 1;
    }

    //Kernel
    distKernel <<<numBlocks, threadsPerBlock >>> (dev_D, dev_A, dev_members, dev_numOfMembers, distances, vp, dev_threads);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_D);
    cudaFree(dev_A);
    cudaFree(dev_members);

    double* result = (double*) malloc((T->numOfMembers - 1) * sizeof(double) );
    for (int i = 0; i < T->numOfMembers - 1; i++) {
        result[i] = distances[i];
    }

    return result;
}

__global__ void distKernel(int* D, double* A, int* members, int* numOfMembers, double* distances, double* vp, int* threads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;;
    if (i < *threads) {
        for (int j = 0; j < *D; j++) {
            distances[i] += pow(vp[j] - *(A + members[i] * (*D) + j), 2.0);
        }
        distances[i] = sqrt(distances[i]);
    }
}


double calculateMedian(vptree* T) {
    double* distances;
    cudaError_t cudaStatus;
    cudaMallocManaged(&distances, (T->parent->numOfMembers - 1) * sizeof(double));

    for (int i = 0; i < T->parent->numOfMembers - 1; i++) {
        distances[i] = 0;
    }

    double* vp;
    cudaMallocManaged(&vp, T->D * sizeof(double));

    for (int i = 0; i < T->D; i++) {
        vp[i] = T->parent->VPCords[i];
    }

    int* dev_D;
    double* dev_A;
    int* dev_members;
    int* dev_numOfMembers;
    int* dev_threads;

    const int sizeA = T->N * T->D * sizeof(double);
    const int sizeMembers = T->parent->numOfMembers * sizeof(int);
    const int threads = T->parent->numOfMembers - 1;

    cudaMalloc(&dev_D, sizeof(int));
    cudaMalloc(&dev_A, sizeA);
    cudaMalloc(&dev_members, sizeMembers);
    cudaMalloc(&dev_numOfMembers, sizeof(int));
    cudaMalloc(&dev_threads, sizeof(int));

    cudaMemcpy(dev_D, &T->D, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A, T->A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_members, T->parent->members, sizeMembers, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_numOfMembers, &T->parent->numOfMembers, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_threads, &threads, sizeof(int), cudaMemcpyHostToDevice);
    int threadsPerBlock;
    int numBlocks;

    if (threads > 1024) {
        threadsPerBlock = 1024;
        numBlocks = ceil(float(threads) / 1024);
    }
    else {
        threadsPerBlock = threads;
        numBlocks = 1;
    }

    //Kernel
    distKernel <<<numBlocks, threadsPerBlock >>> (dev_D, dev_A, dev_members, dev_numOfMembers, distances, vp, dev_threads);
    cudaDeviceSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

    cudaFree(dev_D);
    cudaFree(dev_A);
    cudaFree(dev_members);

    double Median = median(distances, T->parent->numOfMembers - 1);
    return Median;
}