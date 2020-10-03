#ifndef KNNSEARCH_H
#define KNNSEARCH_H
#include "vptree.cuh"
#include <queue>
#include <vector>
#include <iostream>


struct HeapItem {
    HeapItem(int index, double dist) :
        index(index), dist(dist) {}
    int index;
    double dist;
    bool operator<(const HeapItem& o) const {
        return dist < o.dist;
    }
};

typedef struct kdtree {
    kdtree* leftNode;
    kdtree* rightNode;
    double* coordinates;
    double centerValue;
    int index;
    int axis;
} kdtree;


double dist(double* X, double* Y, int dim);

void knnBrute(double* A, int query, int n, int d, std::priority_queue<HeapItem>* heap, int k);

void VPTreeknnSearch(int query, vptree* node, int k, std::priority_queue<HeapItem>* heap, double t, int threadID, int* numNodesVP);

void KDTreeBuild(double* X, kdtree* root, int n, int d);

void KDTreeBuildNode(double* X, kdtree* tree, double* temp, int* indexArray, int n, int d, int start, int end, int depth);

void KDTreeknnSearch(kdtree* node, std::priority_queue<HeapItem>& neighborHeap, double* query, int d, int k, int threadID, int* numNodesKD);





#endif