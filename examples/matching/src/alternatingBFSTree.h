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

#ifndef ALTERNATINGBFSTREE_H
#define ALTERNATINGBFSTREE_H

#include <algorithm>
#include <iostream>
#include <vector>
#include <CL/sycl.hpp>

// Frontier level synchronization w pred
void alternatingBFSTree(sycl::queue &q, 
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<int> &dist,
                sycl::buffer<int> &pred,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &degree,
                sycl::buffer<int> &match,
                const int vertexNum){

  constexpr const size_t SingletonSz = 1;

  const sycl::range Singleton{SingletonSz};

  // Expanded
  sycl::buffer<bool> expanded{Singleton};
  sycl::buffer<int> dzs{Singleton};



  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;
    auto deg = degree.get_access<read_t>();
    auto dzs_i = dzs.get_access<read_write_t>();


    auto m = match.get_access<read_t>();
    auto d = dist.get_access<dwrite_t>();    
    auto p = pred.get_access<dwrite_t>();
    auto s = start.get_access<dwrite_t>();

    auto dep = depth.get_access<dwrite_t>();
    auto exp = expanded.get_access<dwrite_t>();
    dzs_i[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
      if (!deg[i])
        ++dzs_i[0];
      if (m[i] < 4){
        d[i] = 0;
      } else {
        d[i] = -1;
      }
      s[i] = i;
      p[i] = i;
    }
    dep[0] = -1;
    exp[0] = 0;
    printf("Num degree zero vertices %d\n", dzs_i[0]);
  }

  const size_t numBlocks = vertexNum;
  const sycl::range VertexSize{numBlocks};

  const size_t threadsPerBlock = 32;
  const size_t totalThreads = numBlocks * threadsPerBlock;

  const sycl::range NumWorkItems{totalThreads};
  const sycl::range WorkGroupSize{threadsPerBlock};
  sycl::nd_range<1> test{NumWorkItems, WorkGroupSize};


  //printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);

  bool flag = false;
  do{

    {
      const auto write_t = sycl::access::mode::write;
      auto exp = expanded.get_access<write_t>();
      exp[0] = false;
    }

    // Command Group creation
    auto cg = [&](sycl::handler &h) {    
      const auto read_write_t = sycl::access::mode::read_write;
      auto dep = depth.get_access<read_write_t>(h);
      h.parallel_for(Singleton,
                    [=](sycl::id<1> i) { dep[0] = dep[0]+1; });
    };
    q.submit(cg);


    // Necessary to avoid atomics in setting pred/dist/start
    // Conflicts could come from setting pred and start non-atomically.
    // (Push phase) As dist is race-proof, only set pred in the frontier expansion 
    // (Pull phase) pull start into new frontier.
    // Command Group creation
    auto cg3 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      auto depth_i = depth.get_access<read_t>(h);
      auto dist_i = dist.get_access<read_t>(h);
      auto pred_i = pred.get_access<read_t>(h);
      auto start_i = start.get_access<write_t>(h);
      h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        if(depth_i[0] == dist_i[i]) 
          start_i[i] = start_i[pred_i[i]];
      });
    };
    q.submit(cg3);

    // Command Group creation
    auto cg2 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto rows_i = rows.get_access<read_t>(h);
      auto cols_i = cols.get_access<read_t>(h);
      auto depth_i = depth.get_access<read_t>(h);
      auto match_i = match.get_access<read_t>(h);

      auto expanded_i = expanded.get_access<write_t>(h);
      auto dist_i = dist.get_access<read_write_t>(h);
      auto pred_i = pred.get_access<write_t>(h);

      h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        //printf("hellow from item %lu thread %lu gr %lu w range %lu \n", item.get_global_linear_id(), threadIdx, src, r[0]);
                        
                        // Not a frontier vertex
                        if (dist_i[src] != depth_i[0]) return;
                        if (depth_i[0] % 2 == 0){
                          for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                            // atomic isn't neccessary since I don't set predecessor.
                            // even if I set predecessor, all races remain in the universe of
                            // valid solutions.
                              if (dist_i[col] == -1){
                                dist_i[col] = dist_i[src] + 1;
                                pred_i[col] = src;
                                expanded_i[0] = 1;
                              }
                          }
                        } else {
                          auto col = match_i[src];  
                          // If this edge is matched, I know the next vertex already
                          if (dist_i[col] == -1 && 4 <= col){
                              //if (dist_i[col] == -1 && (match_i[src] == match_i[col])){
                              dist_i[col] = dist_i[src] + 1;
                              pred_i[col] = src;
                              expanded_i[0] = 1;
                          }
                        }       
      });
    };
    q.submit(cg2);

    {
      const auto read_t = sycl::access::mode::read;
      auto exp = expanded.get_access<read_t>();
      flag = exp[0];
    }
  } while(flag);

  //#ifdef NDEBUG
  {
    const auto read_t = sycl::access::mode::read;
    auto d = dist.get_access<read_t>();
    auto s = start.get_access<read_t>();
    auto dep = depth.get_access<read_t>();
    #ifdef NDEBUG
    std::cout << "Distance from start is : " << std::endl;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      for (int i = 0; i < vertexNum; i++) {
        if (d[i] == depth_to_print) printf("vertex %d dist %d start %d\n",i, d[i], s[i]);
      }
    }
    #endif
    const auto read_write_t = sycl::access::mode::read_write;
    auto dzs_i = dzs.get_access<read_write_t>();

    dzs_i[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
      if (d[i] > dep[0]) printf("Error: vertex %d dist %d start %d\n",i, d[i], s[i]);
      if (!d[i])++dzs_i[0];
    }
    std::cout << std::endl;
    std::cout << "Number zero dist vertices is : " << dzs_i[0] << std::endl;
  }
  //#endif

  return;
}


// Only continue a a src's frontier while said frontier hasnt reached a blossom/augpath.
// This avoids wasted effort, since MS-BFS matches regenerate the BFS trees every iteration.
// The problem is atomically handling whether a src has found an augpath.  Blossoms share a src
// so there is no difficulty.
void alternatingBFSTree(sycl::queue &q, 
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<int> &dist,
                sycl::buffer<int> &pred,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &degree,
                sycl::buffer<int> &match,
                sycl::buffer<uint64_t> &bridgeVertex,
                const int vertexNum){

  constexpr const size_t SingletonSz = 1;

  const sycl::range Singleton{SingletonSz};

  // Expanded
  sycl::buffer<bool> expanded{Singleton};
  sycl::buffer<int> dzs{Singleton};


  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;
    auto deg = degree.get_access<read_t>();
    auto dzs_i = dzs.get_access<read_write_t>();


    auto m = match.get_access<read_t>();
    auto d = dist.get_access<dwrite_t>();    
    auto p = pred.get_access<dwrite_t>();
    auto s = start.get_access<dwrite_t>();

    auto dep = depth.get_access<dwrite_t>();
    auto exp = expanded.get_access<dwrite_t>();
    dzs_i[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
      if (!deg[i])
        ++dzs_i[0];
      if (m[i] < 4){
        d[i] = 0;
      } else {
        d[i] = -1;
      }
      s[i] = i;
      p[i] = i;
    }
    dep[0] = -1;
    exp[0] = 0;
    printf("Num degree zero vertices %d\n", dzs_i[0]);
  }

  const size_t numBlocks = vertexNum;
  const sycl::range VertexSize{numBlocks};

  const size_t threadsPerBlock = 32;
  const size_t totalThreads = numBlocks * threadsPerBlock;

  const sycl::range NumWorkItems{totalThreads};
  const sycl::range WorkGroupSize{threadsPerBlock};
  sycl::nd_range<1> test{NumWorkItems, WorkGroupSize};


  //printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);

  bool flag = false;
  do{

    {
      const auto write_t = sycl::access::mode::write;
      auto exp = expanded.get_access<write_t>();
      exp[0] = false;
    }

    // Command Group creation
    auto cg = [&](sycl::handler &h) {    
      const auto read_write_t = sycl::access::mode::read_write;
      auto dep = depth.get_access<read_write_t>(h);
      h.parallel_for(Singleton,
                    [=](sycl::id<1> i) { dep[0] = dep[0]+1; });
    };
    q.submit(cg);


    // Necessary to avoid atomics in setting pred/dist/start
    // Conflicts could come from setting pred and start non-atomically.
    // (Push phase) As dist is race-proof, only set pred in the frontier expansion 
    // (Pull phase) pull start into new frontier.
    // Command Group creation
    auto cg3 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      auto depth_i = depth.get_access<read_t>(h);
      auto dist_i = dist.get_access<read_t>(h);
      auto pred_i = pred.get_access<read_t>(h);
      auto start_i = start.get_access<write_t>(h);
      h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        if(depth_i[0] == dist_i[i]) 
          start_i[i] = start_i[pred_i[i]];
      });
    };
    q.submit(cg3);

    // Command Group creation
    auto cg2 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto rows_i = rows.get_access<read_t>(h);
      auto cols_i = cols.get_access<read_t>(h);
      auto depth_i = depth.get_access<read_t>(h);
      auto match_i = match.get_access<read_t>(h);

      auto expanded_i = expanded.get_access<write_t>(h);
      auto dist_i = dist.get_access<read_write_t>(h);
      auto pred_i = pred.get_access<write_t>(h);

      h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        //printf("hellow from item %lu thread %lu gr %lu w range %lu \n", item.get_global_linear_id(), threadIdx, src, r[0]);
                        
                        // Not a frontier vertex
                        if (dist_i[src] != depth_i[0]) return;
                        if (depth_i[0] % 2 == 0){
                          for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                            // atomic isn't neccessary since I don't set predecessor.
                            // even if I set predecessor, all races remain in the universe of
                            // valid solutions.
                              if (dist_i[col] == -1){
                                dist_i[col] = dist_i[src] + 1;
                                pred_i[col] = src;
                                expanded_i[0] = 1;
                              }
                          }
                        } else {
                          auto col = match_i[src];  
                          // If this edge is matched, I know the next vertex already
                          if (dist_i[col] == -1 && 4 <= col){
                              //if (dist_i[col] == -1 && (match_i[src] == match_i[col])){
                              dist_i[col] = dist_i[src] + 1;
                              pred_i[col] = src;
                              expanded_i[0] = 1;
                          }
                        }       
      });
    };
    q.submit(cg2);

    // check for bridges.  Terminate a frontier prematurely if one is found.
    // A bridge is an unmatched edge between two even levels
    // or a matched edge between two odd levels.

    // Command Group creation
    auto cg3 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto rows_i = rows.get_access<read_t>(h);
      auto cols_i = cols.get_access<read_t>(h);
      auto depth_i = depth.get_access<read_t>(h);
      auto match_i = match.get_access<read_t>(h);

      auto dist_i = dist.get_access<read_t>(h);
      auto pred_i = pred.get_access<read_t>(h);

      h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        //printf("hellow from item %lu thread %lu gr %lu w range %lu \n", item.get_global_linear_id(), threadIdx, src, r[0]);
                        
                        // Not a new frontier vertex
                        if (dist_i[src] != depth_i[0]+1) return;

                        // A bridge is an unmatched edge between two even levels
                        if ((depth_i[0]+1) % 2 == 0){
                          for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                              // An edge to a vertex in my even level.
                              if (dist_i[col] == dist_i[src]){
                                
                              }
                          }
                        } else {
                          for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                              // A matched edge to a vertex in my odd level.
                              if (match_i[col] == match_i[src] &&
                                  dist_i[col] == dist_i[src]){
                                
                              }
                          }
                        }       
      });
    };
    q.submit(cg3);


    {
      const auto read_t = sycl::access::mode::read;
      auto exp = expanded.get_access<read_t>();
      flag = exp[0];
    }
  } while(flag);

  //#ifdef NDEBUG
  {
    const auto read_t = sycl::access::mode::read;
    auto d = dist.get_access<read_t>();
    auto s = start.get_access<read_t>();
    auto dep = depth.get_access<read_t>();
    #ifdef NDEBUG
    std::cout << "Distance from start is : " << std::endl;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      for (int i = 0; i < vertexNum; i++) {
        if (d[i] == depth_to_print) printf("vertex %d dist %d start %d\n",i, d[i], s[i]);
      }
    }
    #endif
    const auto read_write_t = sycl::access::mode::read_write;
    auto dzs_i = dzs.get_access<read_write_t>();

    dzs_i[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
      if (d[i] > dep[0]) printf("Error: vertex %d dist %d start %d\n",i, d[i], s[i]);
      if (!d[i])++dzs_i[0];
    }
    std::cout << std::endl;
    std::cout << "Number zero dist vertices is : " << dzs_i[0] << std::endl;
  }
  //#endif

  return;
}


#endif