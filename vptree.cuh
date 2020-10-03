#ifndef VPTREE_CUH
#define VPTREE_CUH
#include <cuda.h>
#include <device_launch_parameters.h>

// type definition of vptree
typedef struct vptree{
    struct vptree* parent;
    struct vptree* inner;
    struct vptree* outer;
    int ivp;
    double* VPCords;
    double median;
    int* members;
    int numOfMembers;
    double* A;
    int D;
    int N;
}vptree;


// ========== LIST OF ACCESSORS
//! Build vantage-point tree given input dataset X
/*!
\param X Input data points, stored as [n-by-d] array
\param n Number of data points (rows of X)
\param d Number of dimensions (columns of X)
\return The vantage-point tree
*/
vptree * buildvp(double *X, int n, int d);
//! Return vantage-point subtree with points inside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
vptree * getInner(vptree * T);
//! Return vantage-point subtree with points outside radius
/*!
\param node A vantage-point tree
\return The vantage-point subtree
*/
vptree * getOuter(vptree * T);
//! Return median of distances to vantage point
/*!
\param node A vantage-point tree
\return The median distance
*/double getMD(vptree * T);
//! Return the coordinates of the vantage point
/*!
\param node A vantage-point tree
\return The coordinates [d-dimensional vector]
*/
double * getVP(vptree * T);
//! Return the index of the vantage point
/*!
\param node A vantage-point tree
\return The index to the input vector of data points
*/
int getIDX(vptree * T);

vptree* buildInner(vptree *T, double* distances);

vptree* buildOuter(vptree *T, double* distances);

double* calculateDistances(vptree* T);

__global__ void buildInnerKernel(vptree* T, vptree* inner, bool* isInside, double* distances, int* membersInner, int* membersT);

__global__ void buildOuterKernel(vptree* T, vptree* outer, bool* isInside, double* distances, int* membersOuter, int* membersT);

__global__ void distKernel(int* D, double* A, int* members, int* numOfMembers, double* distances, double* vp, int* threads);


double calculateMedian(vptree *T);
#endif
