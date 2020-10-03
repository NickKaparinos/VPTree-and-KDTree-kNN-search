/*!
  \file   main.cu
  \brief  Construction of VP and KD trees and knn search using them

  \author Nikos Kaparinos
  \date   2020
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <chrono>
#include "vptree.cuh"
#include "knnSearch.h"
#include "readMatrix.c"
//#include "readMatrix.h"

using namespace std::chrono;

int main() {
    // Flags
    const bool validateVPTreeKnn = false;  
    const bool validateKDTreeKnn = false;
    const bool verbose = false;                            // If true, knn results of point indexToDisplayVerboseKnnResults will be displayed
    const int indexToDisplayVerboseKnnResults = 0;         // Knn results of this point with be displayed, if verbose is true
    const bool readDataFromFile = false;

    // Constants
    const int n = 100;
    const int d = 2;
    const int k = 5;
    const int numThreads = n;


    // Populate data array
    double* dataArr = (double*)malloc(n * d * sizeof(double));

    srand(100);
    if (readDataFromFile) {
        readMatrix(n, d, dataArr, "data.txt");
    }else {
        for (int i = 0; i < n * d; i++) {
            dataArr[i] = rand() % 10000;
        }
    }


    // --------- VP --------- //


    // Timed Construction of KDTree
    high_resolution_clock::time_point buildVPStart = high_resolution_clock::now();
    vptree* rootVP = buildvp(dataArr, n, d);
    high_resolution_clock::time_point buildVPEnd = high_resolution_clock::now();
    duration<double> timeSpanBuildVP = duration_cast<duration<double>>(buildVPEnd - buildVPStart);
    std::cout << "VPTree construnction time: " << timeSpanBuildVP.count() << " seconds.\n";


    // VPTree parallel knn search using omp
    bool* VPTreeknnCorrect = (bool*)malloc(numThreads * sizeof(bool));
    double* VPKnnTime = (double*)malloc(numThreads * sizeof(double));
    int* numNodesVP = (int*)calloc(numThreads, sizeof(int));

    omp_set_num_threads(numThreads);
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();

        std::priority_queue<HeapItem> VPTreeNeighborHeap;
        double tau = std::numeric_limits<double>::max();

        // Timed VPTree knn search
        high_resolution_clock::time_point VPknnStart = high_resolution_clock::now();
        VPTreeknnSearch(tid, rootVP, k, &VPTreeNeighborHeap, tau, tid, numNodesVP);
        high_resolution_clock::time_point VPknnEnd = high_resolution_clock::now();
        duration<double> timeSpanVPKnn = duration_cast<duration<double>>(VPknnEnd - VPknnStart);
        VPKnnTime[tid] = timeSpanVPKnn.count();

        // Verbose VPTree knn results
        if (verbose && tid == indexToDisplayVerboseKnnResults) {
            std::priority_queue<HeapItem> heapCopy = VPTreeNeighborHeap;
            printf("Verbose knn Results for point %d using VPTree, k = %d\n", indexToDisplayVerboseKnnResults, k);
            for (int i = 0; i < k; i++) {
                printf("%d nearest neighbor index = %d, dist = %f\n", k - i, heapCopy.top().index, heapCopy.top().dist);
                heapCopy.pop();
            }
            printf("\n");
        }

        // Validate thread knn results
        if (validateVPTreeKnn) {
            // Executre brute knn
            std::priority_queue<HeapItem> heapBrute;
            knnBrute(dataArr, tid, n, d, &heapBrute, k);

            // Verbose brute force knn results
            if (verbose && tid == indexToDisplayVerboseKnnResults) {
                std::priority_queue<HeapItem> heapBruteCpy = heapBrute;
                printf("Verbose knn Results for point %d using brute force, k = %d\n", indexToDisplayVerboseKnnResults, k);
                for (int i = 0; i < k; i++) {
                    printf("%d nearest neighbor index = %d, dist = %f\n", k - i, heapBruteCpy.top().index, heapBruteCpy.top().dist);
                    heapBruteCpy.pop();
                }
                printf("\n");
            }

            bool correct = 1;
            for (int i = 0; i < k; i++) {
                double temp1 = VPTreeNeighborHeap.top().dist;       // Validation is done using distance, instead of index
                double temp2 = heapBrute.top().dist;                // Because some points may be equidistant
                if (fabs(temp1 - temp2) > 1e-2) {
                    correct = 0;
                    break;
                }
                VPTreeNeighborHeap.pop();
                heapBrute.pop();
            }
            VPTreeknnCorrect[tid] = correct;
        }
    }

    if (validateVPTreeKnn) {
        // Check the validation result of each thread
        bool VPTreeknnCorrectFinal = 1;
        for (int i = 0; i < numThreads; i++) {
            VPTreeknnCorrectFinal = VPTreeknnCorrectFinal & VPTreeknnCorrect[i];
        }
        // Display message
        if (VPTreeknnCorrectFinal) {
            printf("\nVPTree knn Result: Correct!\n\n");
        }
        else {
            printf("\nVPTree knn Result: INCORRECT! ERROR!\n\n");
        }
    }

    // Calculate average number of Vantage point tree nodes visited searching for knn
    // Calculate average knn search time using the VP tree
    double avgNodesVP = 0;
    double avgVPknnTime = 0;
    for (int i = 0; i < numThreads; i++) {
        avgNodesVP += numNodesVP[i];
        avgVPknnTime += VPKnnTime[i];
    }
    avgNodesVP /= numThreads;
    avgVPknnTime /= numThreads;
    printf("Average number of nodes visited using the VPTree: %f\n", avgNodesVP);
    printf("Average VPTree knn search time: %f seconds.\n\n", avgVPknnTime);

    free(VPTreeknnCorrect);
    free(numNodesVP);
    free(VPKnnTime);
    free(rootVP);


    // --------- KD --------- //


    // Timed KDTree construction
    kdtree* rootKD = (kdtree*)malloc(sizeof(kdtree));
    high_resolution_clock::time_point buildKDStart = high_resolution_clock::now();
    KDTreeBuild(dataArr, rootKD, n, d);
    high_resolution_clock::time_point buildKDEnd = high_resolution_clock::now();
    duration<double> timeSpanBuildKD = duration_cast<duration<double>>(buildKDEnd - buildKDStart);
    std::cout << "KDTree construnction time: " << timeSpanBuildKD.count() << " seconds.\n";

    // KDTree parallel knn search using omp
    bool* KDTreeknnCorrect = (bool*)malloc(numThreads * sizeof(bool));
    double* KDKnnTime = (double*)malloc(numThreads * sizeof(double));
    int* numNodesKD = (int*)calloc(numThreads, sizeof(int));

    //for (int w = 0; w < numThreads; w++) {
    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();

        // Timed KDTree knn search 
        double* query = (double*)malloc(d * sizeof(double));
        for (int i = 0; i < d; i++) {
            query[i] = dataArr[tid * d + i];
        }

        std::priority_queue<HeapItem> KDTreeNeighborHeap;
        high_resolution_clock::time_point KDknnStart = high_resolution_clock::now();
        KDTreeknnSearch(rootKD, KDTreeNeighborHeap, query, d, k, tid, numNodesKD);
        high_resolution_clock::time_point KDknnEnd = high_resolution_clock::now();
        duration<double> timeSpanKDKnn = duration_cast<duration<double>>(KDknnEnd - KDknnStart);
        KDKnnTime[tid] = timeSpanKDKnn.count();

        free(query);

        // Verbose KDTree knn results
        if (verbose && tid == indexToDisplayVerboseKnnResults) {
            std::priority_queue<HeapItem> heapCopy = KDTreeNeighborHeap;
            printf("Verbose knn Results for point %d using KDTree, k = %d\n", indexToDisplayVerboseKnnResults, k);
            for (int i = 0; i < k; i++) {
                printf("%d nearest neighbor index = %d, dist = %f\n", k - i, heapCopy.top().index, heapCopy.top().dist);
                heapCopy.pop();
            }
            printf("\n");
        }

        // Validate thread knn results
        if (validateKDTreeKnn) {
            // Execute brute knn
            std::priority_queue<HeapItem> heapBrute;
            knnBrute(dataArr, tid, n, d, &heapBrute, k);

            bool correct = 1;
            for (int i = 0; i < k; i++) {
                double temp1 = KDTreeNeighborHeap.top().dist;       // Validation is done using distance, instead of index
                double temp2 = heapBrute.top().dist;                // Because some points may be equidistant
                if (fabs(temp1 - temp2) > 1e-2) {
                    correct = 0;
                    break;
                }
                KDTreeNeighborHeap.pop();
                heapBrute.pop();
            }
            KDTreeknnCorrect[tid] = correct;
        }
    }

    if (validateKDTreeKnn) {
        // Check the validation result of each thread
        bool KDTreeknnCorrectFinal = 1;
        for (int i = 0; i < numThreads; i++) {
            KDTreeknnCorrectFinal = KDTreeknnCorrectFinal & KDTreeknnCorrect[i];
        }

        // Display message
        if (KDTreeknnCorrectFinal) {
            printf("\nKDTree knn Result: Correct!\n\n");
        }
        else {
            printf("\nKDTree knn Result: INCORRECT! ERROR!\n\n");
        }
    }
    
    // Calculate average number of KD tree nodes visited searching for knn
    // Calculate average knn search time using the KD tree
    double avgNodesKD = 0;
    double avgKDknnTime = 0;
    for (int i = 0; i < numThreads; i++) {
        avgNodesKD += numNodesKD[i];
        avgKDknnTime += KDKnnTime[i];
    }
    avgNodesKD /= numThreads;
    avgKDknnTime /= numThreads;
    printf("Average number of nodes visited using the KDTree: %f\n", avgNodesKD);
    printf("Average KDTree knn search time: %f seconds.\n\n", avgKDknnTime);

    return 0;
}
