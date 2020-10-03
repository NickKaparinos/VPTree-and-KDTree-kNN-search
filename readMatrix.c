#include "readMatrix.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#pragma warning(disable:4996)

void readMatrix(int rows, int cols, double* a, const char* filename){
    FILE* pf;
    char str[40];
    char** temp;
    temp = (char**)malloc(cols * sizeof(char*));
    for (int i = 0; i < cols; i++) {
        temp[i] = (char*)malloc(40 * sizeof(char));
    }

    pf = fopen(filename, "r");
    if (pf == NULL)
        return;

    for (int i = 0; i < rows; ++i) {
        fgets(str, 40, pf);
        sscanf(str, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s", temp[0], temp[1], temp[2], temp[3], temp[4], temp[5], temp[6], temp[7], temp[8], temp[9], temp[10], temp[11], temp[12], temp[13], temp[14], temp[15], temp[16], temp[17]);
        for (int j = 0; j < cols; j++) {
            a[i * cols + j] = atof(temp[j]);
        }
        //puts(str);
    }

    fclose(pf);
}
