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
  sycl::queue myQueue{CUDASelector};
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

  //maximalMatching();
  maximalMatching(myQueue, 
                rows, 
                cols, 
                requests,
                match,
                graph.vertexNum,
                config.barrier);
  // Initialize input data
  sycl::buffer<int> dist{VertexSize};
  // For bt'ing.
  sycl::buffer<int> pred{VertexSize};
  // To identify the source of the disjoint alt-tree
  // in the forest.  This is necessary to ensure, two
  // augmenting paths/blossoms don't share a start.
  sycl::buffer<int> start{VertexSize};

  sycl::buffer<int> depth{Singleton};

  /*
  alternatingBFSTree(myQueue,
                    rows, 
                    cols, 
                    dist,
                    degree,
                    match,
                    graph.vertexNum);
  */
  
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
  sycl::buffer<int> winningAP{VertexSize};
  sycl::buffer<int> auxMatch{VertexSize};

  maximalMatching(myQueue, 
                  rows, 
                  cols, 
                  requests,
                  auxMatch,
                  dist,
                  depth,
                  graph.vertexNum,
                  config.barrier);
  return 0;
}
