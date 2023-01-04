#ifndef CSRGraph_H
#define CSRGraph_H

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <assert.h>

struct CSRGraph{
    unsigned int vertexNum; // Number of Vertices
    unsigned int edgeNum; // Number of Edges
    unsigned int* dst;
    unsigned int* srcPtr;
    int* degree;
    
    void create(unsigned int xn,unsigned int xm); // Initializes the graph rep
    void copy(CSRGraph graph);
    void deleteVertex(unsigned int v);
    unsigned int findMaxDegree();
    void printGraph();
    void del();
};
#endif