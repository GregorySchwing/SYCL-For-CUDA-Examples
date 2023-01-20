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
//#include "atomicAugment.h"
//#include "lockFreeAugment.h"

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
  sycl::buffer<uint32_t> rows{graph.srcPtr, RowSize};
  sycl::buffer<uint32_t> cols{graph.dst, ColSize};
  sycl::buffer<int> degree{graph.degree, VertexSize};
  sycl::buffer<int> depth{Singleton};


  // Device output vector
  sycl::buffer<int> match{VertexSize};
  // Intermediate vector
  sycl::buffer<int> requests{VertexSize};
  // Two uint32's are packed into a space.  This way we avoid atomics.
  sycl::buffer<uint64_t> bridgeVertex{VertexSize};

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
  //printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);


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

  {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;


    auto m = match.get_access<read_t>();
    bool bad = false;
    for (int i = 0; i < graph.vertexNum; i++) {
      if (m[i] >= 4 && !graph.has((m[i]-4), (m[(m[i]-4)]-4))){
        printf("Matching between vertices over non-exisiting edge!!! %d %d\n",(m[i]-4),(m[(m[i]-4)]-4));
        bad = true;
      }
    }
    if (bad){
      fflush(stdout);
      exit(1);
    }
  }

  printf("\nElapsed Time for SYCL Initial Max Match: %f\n",elapsed_seconds_max.count());
  printf("SYCL initial match count is: %u\n", syclinitmatchc/2);

  chrono::time_point<std::chrono::system_clock> nd_item_initial_match_begin, nd_item_initial_match_end;
  nd_item_initial_match_begin = std::chrono::system_clock::now(); 
  int nditem_syclinitmatchc;
  maximalMatchingNDItem(myQueue, 
                nditem_syclinitmatchc,
                rows, 
                cols, 
                requests,
                match,
                graph.vertexNum,
                config.barrier);
  nd_item_initial_match_end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = nd_item_initial_match_end - nd_item_initial_match_begin; 



  {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;


    auto m = match.get_access<read_t>();
    bool bad = false;
    for (int i = 0; i < graph.vertexNum; i++) {
      if (m[i] >= 4 && !graph.has((m[i]-4), (m[(m[i]-4)]-4))){
        printf("Matching between vertices over non-exisiting edge!!! %d %d\n",(m[i]-4),(m[(m[i]-4)]-4));
        bad = true;
      }
    }
    if (bad){
      fflush(stdout);
      exit(1);
    }
  }

  printf("\nElapsed Time for SYCL NDItem Initial Max Match: %f\n",elapsed_seconds_max.count());
  printf("SYCL initial match count is: %u\n", nditem_syclinitmatchc/2);

  // These arrays are uncompromising.
  sycl::buffer<int> dist{VertexSize};
  // when a BFS round comes upon a blossom
  // the blossom starts at the head and advances the
  // frontier to all outgoing edges of each node in the blossom
  //sycl::buffer<int> blossom{VertexSize};
  //sycl::buffer<int> fll{VertexSize};
  //sycl::buffer<int> bll{VertexSize};

  // For bt'ing. This array is uncompromising.
  // it is neccessary to get the starting vertex
  // which is required for detecting blossoms.
  sycl::buffer<int> pred{VertexSize};

  // To identify the source of the disjoint alt-tree
  // in the forest.  This is necessary to ensure, two
  // augmenting paths don't share a start -> blossom
  // This isn't strictly neccessary since we can bt.
  sycl::buffer<int> start{VertexSize};

  // For the augmenting methods
  sycl::buffer<bool> matchable{VertexSize};

  sycl::buffer<int> base{VertexSize};
  sycl::buffer<int> forward{VertexSize};
  sycl::buffer<int> backward{VertexSize};

  {
    const auto dwrite_t = sycl::access::mode::discard_write;
    auto base_i = base.get_access<dwrite_t>();
    auto forward_i = forward.get_access<dwrite_t>();
    auto backward_i = backward.get_access<dwrite_t>();
    for (int i = 0; i < graph.vertexNum; i++) {
      base_i[i] = -1;
      forward_i[i] = -1;
      backward_i[i] = -1;
    }
  }

  int currentMatchc = 0, prevMatchc = 0, iteration = 0;
  currentMatchc = nditem_syclinitmatchc;
  do {
    prevMatchc = currentMatchc;
    // This kernel is fine as is for use in a search tree. 
    // Each subkernel should be logically kernelized by available
    // resources.
    chrono::time_point<std::chrono::system_clock> BFS_begin, BFS_end;
    BFS_begin = std::chrono::system_clock::now(); 
    /*
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
    */

    alternatingBFSTree(myQueue,
                      graph,
                      currentMatchc, 
                      rows, 
                      cols, 
                      dist,
                      pred,
                      start,
                      degree,
                      match,
                      requests,
                      matchable,
                      bridgeVertex,
                      base,
                      forward,
                      backward,
                      graph.vertexNum);

    BFS_end = std::chrono::system_clock::now(); 
    elapsed_seconds_max = BFS_end - BFS_begin; 
    printf("\nElapsed Time for SYCL BFS: %f\n",elapsed_seconds_max.count());
    
    /*

    chrono::time_point<std::chrono::system_clock> augment_begin, augment_end;
    augment_begin = std::chrono::system_clock::now(); 
    //int syclmatchc = 0;

    atomicAugment_a(myQueue, 
              currentMatchc,
              rows, 
              cols, 
              pred,
              dist,
              start,
              depth,
              match,
              requests,
              bridgeVertex,
              graph.vertexNum);
    
    augment_end = std::chrono::system_clock::now(); 
    elapsed_seconds_max = augment_end - augment_begin; 
    printf("\nElapsed Time for SYCL augment_a: %f\n",elapsed_seconds_max.count());
    printf("\nSYCL augment size: %d\n",currentMatchc/2);

    chrono::time_point<std::chrono::system_clock> augment_b_begin, augment_b_end;
    augment_b_begin = std::chrono::system_clock::now(); 

    atomicAugment_b(myQueue, 
                currentMatchc,
                rows, 
                cols, 
                start,
                pred,
                dist,
                match,
                bridgeVertex,
                graph.vertexNum);

    augment_b_end = std::chrono::system_clock::now(); 
    elapsed_seconds_max = augment_b_end - augment_b_begin; 
    printf("\nElapsed Time for SYCL augment_b: %f\n",elapsed_seconds_max.count());
    printf("\nSYCL augment size: %d\n",currentMatchc/2);
    */

    /*
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
    chrono::time_point<std::chrono::system_clock> aux_matching_begin, aux_matching_end;
    aux_matching_begin = std::chrono::system_clock::now(); 
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

    aux_matching_end = std::chrono::system_clock::now(); 
    elapsed_seconds_max = aux_matching_end - aux_matching_begin; 
    printf("\nElapsed Time for SYCL Aux matching: %f\n",elapsed_seconds_max.count());
    
    chrono::time_point<std::chrono::system_clock> augment_begin, augment_end;
    augment_begin = std::chrono::system_clock::now(); 
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
    
    augment_end = std::chrono::system_clock::now(); 
    elapsed_seconds_max = augment_end - augment_begin; 
    printf("\nElapsed Time for SYCL augment: %f\n",elapsed_seconds_max.count());
    */
    printf("\nOuter Iteration %d\n",iteration++);
  } while (prevMatchc != currentMatchc); 

    


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
  


  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;


    auto m = match.get_access<read_t>();

    for (int i = 0; i < graph.vertexNum; i++) {
      if (m[i] >= 4 && !graph.has((m[i]-4), (m[(m[i]-4)]-4))){
        printf("Matching between vertices over non-exisiting edge!!! %d %d\n",(m[i]-4),(m[(m[i]-4)]-4));
      }
    }
  }

  end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 

  printf("\nElapsed Time for SYCL Max Match: %f\n",elapsed_seconds_max.count());
  printf("SYCL match count is: %u\n", currentMatchc/2);
  fflush(stdout);
  // Test matching against serial edmonds
  begin = std::chrono::system_clock::now(); 
  EdmondsSerial e(graph);
  int matchc = e.edmonds();
  end = std::chrono::system_clock::now(); 
	elapsed_seconds_max = end - begin; 
  // matchc is number of matched edges
  printf("\nElapsed Time for Serial edmonds Max Match: %f\n",elapsed_seconds_max.count());
  printf("Serial max matching count : %d\n", matchc);
  return 0;
}
