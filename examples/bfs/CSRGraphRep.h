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

int comp(const void *elem1, const void *elem2)
{
	int f = *((int *)elem1);
	int s = *((int *)elem2);
	if (f > s)
		return 1;
	if (f < s)
		return -1;
	return 0;
}

CSRGraph createCSRGraphFromFile(const char *filename)
{

	CSRGraph graph;
	unsigned int vertexNum;
	unsigned int edgeNum;

	FILE *fp;
	fp = fopen(filename, "r");

	fscanf(fp, "%u%u", &vertexNum, &edgeNum);

	graph.create(vertexNum, edgeNum);

	unsigned int **edgeList = (unsigned int **)malloc(sizeof(unsigned int *) * 2);
	edgeList[0] = (unsigned int *)malloc(sizeof(unsigned int) * edgeNum);
	edgeList[1] = (unsigned int *)malloc(sizeof(unsigned int) * edgeNum);

	for (unsigned int i = 0; i < edgeNum; i++)
	{
		unsigned int v0, v1;
		fscanf(fp, "%u%u", &v0, &v1);
		edgeList[0][i] = v0;
		edgeList[1][i] = v1;
	}

	fclose(fp);

	// Gets the degrees of vertices
	for (unsigned int i = 0; i < edgeNum; i++)
	{
		assert(edgeList[0][i] < vertexNum);
		graph.degree[edgeList[0][i]]++;
		if (edgeList[1][i] >= vertexNum)
		{
			printf("\n%d\n", edgeList[1][i]);
		}
		assert(edgeList[1][i] < vertexNum);
		graph.degree[edgeList[1][i]]++;
	}
	// Fill srcPtration array
	unsigned int nextIndex = 0;
	unsigned int *srcPtr2 = (unsigned int *)malloc(sizeof(unsigned int) * vertexNum);
	for (int i = 0; i < vertexNum; i++)
	{
		graph.srcPtr[i] = nextIndex;
		srcPtr2[i] = nextIndex;
		nextIndex += graph.degree[i];
	}
	graph.srcPtr[vertexNum] = edgeNum * 2;
	// fill Graph Array
	for (unsigned int i = 0; i < edgeNum; i++)
	{
		assert(edgeList[0][i] < vertexNum);
		assert(srcPtr2[edgeList[0][i]] < 2 * edgeNum);
		graph.dst[srcPtr2[edgeList[0][i]]] = edgeList[1][i];
		srcPtr2[edgeList[0][i]]++;
		assert(edgeList[1][i] < vertexNum);
		assert(srcPtr2[edgeList[1][i]] < 2 * edgeNum);
		graph.dst[srcPtr2[edgeList[1][i]]] = edgeList[0][i];
		srcPtr2[edgeList[1][i]]++;
	}

	free(srcPtr2);
	free(edgeList[0]);
	edgeList[0] = NULL;
	free(edgeList[1]);
	edgeList[1] = NULL;
	free(edgeList);
	edgeList = NULL;

	for (unsigned int vertex = 0; vertex < graph.vertexNum; ++vertex)
	{
		qsort(&graph.dst[graph.srcPtr[vertex]], graph.degree[vertex], sizeof(int), comp);
	}

	return graph;
}
#endif