/**
 * SYCL FOR CUDA : BFS Example
 *
 * Copyright 2020 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 * @File: vector_addition.cpp
 */

#include <algorithm>
#include <iostream>
#include <vector>
#include "config.h"
#include "CSRGraphRep.h"
#include "auxFunctions.h"
#include "alternatingBFSTree.h"
#include "match.h"
#include "edmondsSerial.h"
#include "augment.h"

#include <CL/sycl.hpp>
// For min(T a, T b)

int main(int argc, char *argv[]) {

  Config config = parseArgs(argc,argv);
  printf("\nGraph file: %s",config.graphFileName);
  printf("\nUUID: %s\n",config.outputFilePrefix);


  CSRGraph graph = createCSRGraphFromFile(config.graphFileName);
  performChecks(graph, config);
  printf("finished checking\n");

  constexpr const size_t SingletonSz = 1;
  constexpr const size_t DoubletonSz = 2;

  const sycl::range RowSize{graph.vertexNum+1};
  const sycl::range ColSize{graph.edgeNum*2};
  const sycl::range VertexSize{graph.vertexNum};


  const sycl::range Singleton{SingletonSz};
  const sycl::range Doubleton{DoubletonSz};


  // Device input vectors
  sycl::buffer<unsigned int> rows{graph.srcPtr, RowSize};
  sycl::buffer<unsigned int> cols{graph.dst, ColSize};
  sycl::buffer<int> degree{graph.degree, VertexSize};


  // Device output vector
  sycl::buffer<int> match{VertexSize};
  // Intermediate vector
  sycl::buffer<int> requests{VertexSize};

  auto CUDASelector = [](sycl::device const &dev) {
    if (dev.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    } else {
      return -1;
    }
  };
  //sycl::queue myQueue{CUDASelector};
  sycl::queue myQueue{};
  std::cout << "Running on "
      << myQueue.get_device().get_info<sycl::info::device::name>()
      << "\n";
  //sycl::queue myQueue{cl::sycl::host_selector()};
  std::cout << "Selected device : " <<
  myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
  const int numBlocks = graph.vertexNum;
  const int nrVertices = graph.vertexNum;

  const size_t threadsPerBlock = 32;
  const size_t totalThreads = numBlocks * threadsPerBlock;

  const sycl::range NumWorkItems{totalThreads};
  const sycl::range WorkGroupSize{threadsPerBlock};
  sycl::nd_range<1> test{NumWorkItems, WorkGroupSize};
  printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);


  chrono::time_point<std::chrono::system_clock> begin, end;
	std::chrono::duration<double> elapsed_seconds_max, elapsed_seconds_edge, elapsed_seconds_mvc;

  begin = std::chrono::system_clock::now(); 
  //maximalMatching();
  // This kernel is fine as is for use in a search tree. 
  // Each subkernel should be logically kernelized by available
  // resources.
  chrono::time_point<std::chrono::system_clock> initial_match_begin, initial_match_end;
  initial_match_begin = std::chrono::system_clock::now(); 
  int syclinitmatchc;
  maximalMatching(myQueue, 
                syclinitmatchc,
                rows, 
                cols, 
                requests,
                match,
                graph.vertexNum,
                config.barrier);
  initial_match_end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = initial_match_end - initial_match_begin; 

  printf("\nElapsed Time for SYCL Initial Max Match: %f\n",elapsed_seconds_max.count());
  printf("SYCL initial match count is: %u\n", syclinitmatchc);

  // These arrays are uncompromising.
  sycl::buffer<int> dist{VertexSize};
  sycl::buffer<int> blossom{VertexSize};

  // For bt'ing. This array is uncompromising.
  // it is neccessary to get the starting vertex
  // which is required for detecting blossoms.
  sycl::buffer<int> pred{VertexSize};

  // To identify the source of the disjoint alt-tree
  // in the forest.  This is necessary to ensure, two
  // augmenting paths don't share a start -> blossom
  // This isn't strictly neccessary since we can bt.
  sycl::buffer<int> start{VertexSize};

  sycl::buffer<int> depth{Singleton};

  // This kernel is fine as is for use in a search tree. 
  // Each subkernel should be logically kernelized by available
  // resources.
  alternatingBFSTree(myQueue, 
                    rows, 
                    cols, 
                    dist,
                    pred,
                    start,
                    depth,
                    degree,
                    match,
                    graph.vertexNum);

  
  // To identify one and only one Augmenting path 
  // to use the starting v.
  // Both of these are only neccessary in the data-parallel
  // version.  If a single work-group performs the matching,
  // these can be eliminated by work item synchronization.
  sycl::buffer<int> winningAugmentingPath{VertexSize};
  sycl::buffer<int> auxMatch{VertexSize};

  // Match inside even levels > 0 to avoid race conditions
  // in blossoms/augmenting paths.  For example consider a 
  // cycle represented by a circular linked list of odd length n.
  // If each vertex in the level tried creating an augmenting path 
  // with its rightside vertex, the entire level would augment, 
  // but at (2/3)n can be in an forest of disjoint augmenting paths.

  // This kernel should be modified for use in a search tree.
  maximalMatching(myQueue, 
                  rows, 
                  cols, 
                  requests,
                  match,
                  dist,
                  depth,
                  auxMatch,
                  graph.vertexNum,
                  config.barrier);

  int syclmatchc = 0;
  augment_a(myQueue, 
            syclmatchc,
            rows, 
            cols, 
            pred,
            dist,
            start,
            depth,
            match,
            auxMatch,
            winningAugmentingPath,
            graph.vertexNum);
  
  /*
  int matchc =  edmonds(myQueue, 
                      rows, 
                      cols, 
                      vertexNum);
  */

  // detect and shrink blossoms
  // If there is no stem, s.t. r = start
  // then no matching is of this blossom
  // is possible even after contraction.

  // After contraction, reperform bfs
  // If (B,k) is matched in an augmenting path
  // replace (B,k) with (r,k)
  // Replace the unmatched edge (B,j) with the
  // edge (i,j) from which (B,j) originated,
  // followed by the even alt path in B from
  // i to r. If i was an S-vertex when B was formed
  // we use the labels to backtrack from i to r.
  // Otherwise, we use the labels in reverse order
  // around the blossom.

  // Storing B as a doubly linked list with a 
  // marked base makes this easy.

  // I have two options
  //  1)Perform another BFS after blossom contraction
  //  - This will mean I'll need an int array for blossom membership
  //    It may be important for recursive blossom creation/destruction
  //    to differentiate Blossoms with an int key.
  //    If the next BFS iteration passes through a blossom, special conditions
  //    should be used for passing through the blossom.
  //    No edges where both (i,j) are both in the blossom should
  //    be traversed unless (i,j) were within the original blossom.
  //    Therefore, I'll need the doubly linked list structure (rooted at r?) to
  //    subset the edges amongst the blossom.
  //    Alternatively, if the edges are given values, the non-eligible
  //    edges can be deactivated.
  //  2)Immediately process all blossom paths without contraction.
  //    This means considering all eligible outgoing blossom edges for alt-M
  //    augmenting from the vertex which happened upon a blossom.
  //    Such a lazy blossom contraction loses the guaruntees of upper
  //    bounds on a cycle length, and thus should be avoided.
  //    Unless you can prove that an augmenting path will only
  //    pass through the first blossom stumbled upon.



  //augment_b
  

  end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 

  printf("\nElapsed Time for SYCL Max Match: %f\n",elapsed_seconds_max.count());
  printf("SYCL match count is: %u\n", syclmatchc);
  fflush(stdout);
  // Test matching against serial edmonds
  begin = std::chrono::system_clock::now(); 
  EdmondsSerial e(graph);
  int matchc = e.edmonds();
  end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 
  // matchc is number of matched edges
  printf("\nElapsed Time for Serial edmonds Max Match: %f\n",elapsed_seconds_max.count());
  printf("Serial max matching count : %d\n", 2*matchc);
  return 0;
}
