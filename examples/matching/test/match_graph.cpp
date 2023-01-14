//=======================================================================
// Copyright (c) 2005 Aaron Windsor
//
// Distributed under the Boost Software License, Version 1.0. 
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
//=======================================================================
#include <string>
#include <iostream>
#include <boost/graph/adjacency_list.hpp>
#include <cassert>

#include <boost/graph/max_cardinality_matching.hpp>
#include "config.h"


using namespace boost;

typedef adjacency_list<vecS, vecS, undirectedS> my_graph; 

int main(int argc, char *argv[]) {

    // Create the following graph: (it'll look better when output
    // to the terminal in a fixed width font...)
    Config config = parseArgs(argc,argv);
    printf("\nGraph file: %s",config.graphFileName);
    printf("\nUUID: %s\n",config.outputFilePrefix);

	unsigned int vertexNum;
	unsigned int edgeNum;

	FILE *fp;
	fp = fopen(config.graphFileName, "r");

	fscanf(fp, "%u%u", &vertexNum, &edgeNum);

    const int n_vertices = vertexNum;

    // It has a perfect matching of size 8. There are two isolated
    // vertices that we'll use later...

    my_graph g(n_vertices);

    // our vertices are stored in a vector, so we can refer to vertices
    // by integers in the range 0..15

	for (unsigned int i = 0; i < edgeNum; i++)
	{
		unsigned int v0, v1;
		fscanf(fp, "%u%u", &v0, &v1);
        add_edge(v0,v1,g);
	}

	fclose(fp);

    std::vector<graph_traits<my_graph>::vertex_descriptor> mate(n_vertices);

    // find the maximum cardinality matching. we'll use a checked version
    // of the algorithm, which takes a little longer than the unchecked
    // version, but has the advantage that it will return "false" if the
    // matching returned is not actually a maximum cardinality matching
    // in the graph.
    chrono::time_point<std::chrono::system_clock> mcm_begin, mcm_end;
    augment_begin = std::chrono::system_clock::now(); 
    bool success = checked_edmonds_maximum_cardinality_matching(g, &mate[0]);
    assert(success);
    mcm_end = std::chrono::system_clock::now(); 
    elapsed_seconds_max = mcm_end - mcm_begin; 
    printf("\nElapsed Time for Boost MCM: %f\n",elapsed_seconds_max.count());
    printf("\nBoost MCM size: %d\n",matching_size(g, &mate[0]));
    std::cout << std::endl << "Found a matching of size " << matching_size(g, &mate[0]) << std::endl;

    return 0;
}