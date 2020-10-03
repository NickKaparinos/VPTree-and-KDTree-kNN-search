#include <stdio.h>
#include <queue>
#include <vector>
#include "knnSearch.h"
#include "vptree.cuh"
#include "knnSearch.h"
#include "QuickSelect.cuh"


double dist(double* X, double* Y, int dim)
{
	double dist = 0.0;
	for (int i = 0; i < dim; i++)
		dist += (X[i] - Y[i]) * (X[i] - Y[i]);
	return sqrt(dist);
}

void knnBrute(double* A, int query, int n, int d, std::priority_queue<HeapItem>* heap, int k) {
	for (int i = 0; i < n; i++) {
		double distance = dist(&A[query * d], &A[i * d], d);

		heap->push(HeapItem(i, distance));
		if (heap->size() == k + 1) {
			heap->pop();
		}
	}
}


void VPTreeknnSearch(int query, vptree* node, int k, std::priority_queue<HeapItem>* heap, double t, int threadID, int* numNodesVP) {
	if (node == NULL) return;

	// numNodes contains the number of number of nodes each thread has visited
	numNodesVP[threadID]++;

	int d = node->D;
	int n = node->N;
	double tau = t;

	// Find the distance of query point and vantage point
	double dist = 0.0;
	int vantagePointIdx = node->ivp;
	double* queryCoords = (double*)malloc(d * sizeof(double));

	for (int i = 0; i < d; i++) {
		queryCoords[i] = node->A[query * d + i];

		dist += pow(queryCoords[i] - node->VPCords[i], 2);
	}
	dist = sqrt(dist);

	free(queryCoords);

	if (dist < tau) {
		heap->push(HeapItem(vantagePointIdx, dist));
		if (heap->size() == k + 1) {
			heap->pop();
		}
		if (heap->size() == k) {
			tau = heap->top().dist;
		}
	}

	if (node->inner == NULL && node->outer == NULL) {
		return;
	}

	if (dist < node->median) {
		// Search inner subtree first
		VPTreeknnSearch(query, node->inner, k, heap, tau, threadID, numNodesVP);
		if (dist + tau >= node->median) {
			VPTreeknnSearch(query, node->outer, k, heap, tau, threadID, numNodesVP);
		}

	}
	else {
		// Search outer subtree first
		VPTreeknnSearch(query, node->outer, k, heap, tau, threadID, numNodesVP);
		if (dist - tau <= node->median) {
			VPTreeknnSearch(query, node->inner, k, heap, tau, threadID, numNodesVP);
		}

	}

	/*if (dist - tau <= node->median) {
		VPTreeknnSearch(query, node->inner, k, heap, tau, thread);
	}

	if (dist + tau >= node->median) {
		VPTreeknnSearch(query, node->outer, k, heap, tau, thread);
	}*/


}

void KDTreeBuild(double* X, kdtree* root, int n, int d) {
	double* temp = (double*)malloc(n * sizeof(double));
	int* indexArray = (int*)malloc(n * sizeof(int));

	for (int i = 0; i < n; i++) {
		indexArray[i] = i;
	}

	KDTreeBuildNode(X, root, temp, indexArray, n, d, 0, n - 1, 0);
}


void KDTreeBuildNode(double* X, kdtree* node, double* temp, int* indexArray, int n, int d, int start, int end, int depth) {

	double** dataArr = (double**)malloc(n * sizeof(double*));
	for (int i = 0; i < n; i++) {
		dataArr[i] = (double*)malloc(d * sizeof(double));
		for (int j = 0; j < d; j++) {
			dataArr[i][j] = X[i * d + j];
		}
	}
	//double(*dataArr)[d] = (double(*)[d])X;

	node->axis = depth % d;

	if (start == end) {
		// Leaf Node
		node->index = indexArray[start];
		node->coordinates = dataArr[indexArray[start]];
		node->centerValue = 0.0;
		node->leftNode = NULL;
		node->rightNode = NULL;
		return;
	}

	for (int i = start; i <= end; i++) {
		temp[i] = dataArr[indexArray[i]][node->axis];
	}

	int newDepth = depth + 1;
	int center = (start + end) / 2;

	quickSelect2Arrays(center, temp, indexArray, start, end);

	node->index = indexArray[center];
	node->coordinates = dataArr[indexArray[center]];
	node->centerValue = temp[center];
	node->rightNode = (kdtree*)malloc(sizeof(kdtree));

	free(dataArr);

	KDTreeBuildNode(X, node->rightNode, temp, indexArray, n, d, center + 1, end, newDepth);

	if (center <= start) {
		node->leftNode = NULL;
	}
	else {
		node->leftNode = (kdtree*)malloc(sizeof(kdtree));
		KDTreeBuildNode(X, node->leftNode, temp, indexArray, n, d, start, center - 1, newDepth);
	}
}


void KDTreeknnSearch(kdtree* node, std::priority_queue<HeapItem>& neighborHeap, double* query, int d, int k, int threadID, int* numNodesKD) {
	if (node == NULL) {
		return;
	}

	numNodesKD[threadID]++;

	HeapItem point = { node->index, dist(query, node->coordinates, d) };

	// If heap is not full, add point
	if (neighborHeap.size() < k) {
		neighborHeap.push(point);
	}
	else if (point.dist < neighborHeap.top().dist) {	// If heap is full and point should be in the heap, make room and add it
		neighborHeap.pop();
		neighborHeap.push(point);
	}

	int axis = node->axis;
	if (query[axis] <= node->centerValue) {		// Check if the left subtree should be searched		// = was added
		KDTreeknnSearch(node->leftNode, neighborHeap, query, d, k, threadID, numNodesKD);

		if (node->centerValue <= query[axis] + neighborHeap.top().dist) {	// Check if the right subtree should be searched too
			KDTreeknnSearch(node->rightNode, neighborHeap, query, d, k, threadID, numNodesKD);
		}
	}
	else {
		KDTreeknnSearch(node->rightNode, neighborHeap, query, d, k, threadID, numNodesKD);

		if (node->centerValue >= query[axis] - neighborHeap.top().dist) {	// Check if the left subtree should be searched too
			KDTreeknnSearch(node->leftNode, neighborHeap, query, d, k, threadID, numNodesKD);
		}
	}
}


