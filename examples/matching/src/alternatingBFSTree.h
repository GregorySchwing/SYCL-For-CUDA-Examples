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

#include "lockFreeAugment.h"

// Only continue a a src's frontier while said frontier hasnt reached a blossom/augpath.
// This avoids wasted effort, since MS-BFS matches regenerate the BFS trees every iteration.
// The problem is atomically handling whether a src has found an augpath.  Blossoms share a src
// so there is no difficulty.
void alternatingBFSTree(sycl::queue &q, 
                CSRGraph & g,
                int & matchCount,
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<int> &dist,
                sycl::buffer<int> &pred,
                sycl::buffer<int> &start,
                sycl::buffer<int> &degree,
                sycl::buffer<int> &match,
                sycl::buffer<int> &requests,
                sycl::buffer<bool> &matchable,
                sycl::buffer<uint64_t> &bridgeVertex,
                sycl::buffer<int> &base,
                sycl::buffer<int> &forward,
                sycl::buffer<int> &backward,
                sycl::buffer<bool> &inb,
                const int vertexNum){

  constexpr const size_t SingletonSz = 1;

  const sycl::range Singleton{SingletonSz};

  const size_t numBlocks = vertexNum;
  const sycl::range VertexSize{numBlocks};

  const size_t threadsPerBlock = 32;
  const size_t totalThreads = numBlocks * threadsPerBlock;

  const sycl::range NumWorkItems{totalThreads};
  const sycl::range WorkGroupSize{threadsPerBlock};
  // Expanded
  sycl::buffer<bool> expanded{Singleton};
  sycl::buffer<int> dzs{Singleton};
  sycl::buffer<int> depth{Singleton};
  sycl::buffer<bool> ineligible{VertexSize};


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
    auto b_i = bridgeVertex.get_access<dwrite_t>();
    auto ineligible_i = ineligible.get_access<dwrite_t>();

    
    dzs_i[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
      if (!deg[i])
        ++dzs_i[0];
      if (m[i] < 4){
        d[i] = 0;
      } else {
        d[i] = -1;
      }
      b_i[i] = 0;
      s[i] = i;
      p[i] = i;
      ineligible_i[i] = false;
    }
    dep[0] = -1;
    exp[0] = 0;
    printf("Num degree zero vertices %d\n", dzs_i[0]);
  }



  //printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);

  bool flag = false;
  bool blossomsContracted = false;
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
      auto base_i = base.get_access<read_t>(h);
      auto for_i = forward.get_access<read_t>(h);
      auto start_i = start.get_access<read_t>(h);
      auto inb_i = inb.get_access<read_t>(h);

      h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        //printf("hellow from item %lu thread %lu gr %lu w range %lu \n", item.get_global_linear_id(), threadIdx, src, r[0]);
                        
                        // Not a frontier vertex
                        // Ineligible vertices  are those which are sourced at a vertex which has been augmented (match 4+)
                        // or those newly contracted into a blossom (match 3).
                        if (dist_i[base_i[src]] != depth_i[0] || match_i[start_i[base_i[src]]] > 2) return;

                        // Even depth. expand into all neighbors
                        if (depth_i[0] % 2 == 0){
                          for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                            auto base_col = base_i[col];
                            // Avoid internal blossom edges.
                            if (base_i[src]!=base_col){
                              // Only interact with blossoms through the base.
                                if (dist_i[base_col] == -1){
                                  dist_i[base_col] = dist_i[src] + 1;
                                  pred_i[col] = src;
                                  expanded_i[0] = 1;
                                  // This is a blossom vertex.  Write to all the entries.
                                  if (inb_i[base_col]){
                                    auto base = base_col;
                                    auto curr_u = col;
                                    do{
                                      dist_i[curr_u] = dist_i[src] + 1;
                                      // not sure about this line
                                      //pred_i[curr_u] = src;
                                      curr_u = for_i[curr_u];
                                    } while(curr_u != base);
                                  }
                                }
                            }
                          }
                        // If this edge is matched, I know the next vertex already
                        // Odd level and a matched vertex
                        } else if (match_i[src] >= 4){
                          // Can detect if this forms a blossom immediately by depth.
                          auto col = match_i[src]-4;  
                          auto base_col = base_i[col];
                          // Avoid internal blossom edges.
                          if (base_i[src]!=base_col){
                            // Only interact with blossoms through the base.
                              if (dist_i[base_col] == -1){
                                dist_i[base_col] = dist_i[src] + 1;
                                pred_i[col] = src;
                                expanded_i[0] = 1;
                                // This is a blossom vertex.  Write to all the entries.
                                if (inb_i[base_col]){
                                  auto base = base_col;
                                  auto curr_u = col;
                                  do{
                                    dist_i[curr_u] = dist_i[src] + 1;
                                    // not sure about this line
                                    //pred_i[curr_u] = src;
                                    curr_u = for_i[curr_u];
                                  } while(curr_u != base);
                                }
                              }
                          }
                        // Odd level and an unmatched vertex.
                        } else {
                          // trivial augmenting path. consider doing augmenting here.

                        }
      });
    };
    q.submit(cg2);  

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
        if(depth_i[0]+1 == dist_i[i]) 
          start_i[i] = start_i[pred_i[i]];
      });
    };
    q.submit(cg3);



    {
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;
      bool bad = false;

      auto m = match.get_access<read_t>();
      auto pred_i = pred.get_access<read_t>();

      for (int i = 0; i < g.vertexNum; i++) {
        if (pred_i[i] != i  && !g.has(pred_i[i], i) ){
          printf("CATASTROPHIC ERROR! PRED EDGE %u - %u DNE\n",pred_i[i],i);
          bad = true;
        }
      }
      if (bad){
        printf("BFS FAIL!!!\n");
        fflush(stdout);
        exit(1);
      }else{
        printf("BFS SUCCESS!!!\n");
        fflush(stdout);
      }
    }

    // check for bridges.  Terminate a frontier prematurely if one is found.
    // A bridge is an unmatched edge between two even levels
    // or a matched edge between two odd levels.
    {
      const auto read_t = sycl::access::mode::read;
      // If depth is even, new frontier is odd, check for trivials
      auto dep = depth.get_access<read_t>();
      /*
      if (dep[0] % 2 == 0){
        printf("Frontier is odd!\n");
        augment_trivial_paths(q, 
                              matchCount,
                              pred,
                              dist,
                              start,
                              depth,
                              match,
                              vertexNum);  
      }
      // Match free to each other through bridges (red/blue).
      augment_bridges(q, 
          g,
          matchCount,
          rows, 
          cols, 
          bridgeVertex,
          pred,
          dist,
          start,
          depth,
          match,
          requests,
          matchable,
          ineligible,
          vertexNum);
      
      // Contract blossoms
      blossomsContracted = contract_blossoms(q, 
                                          matchCount,
                                          rows, 
                                          cols, 
                                          bridgeVertex,
                                          pred,
                                          dist,
                                          start,
                                          depth,
                                          match,
                                          requests,
                                          matchable,
                                          base,
                                          forward,
                                          backward,
                                          ineligible,
                                          vertexNum);
      */

    }
    {
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;
      bool bad = false;

      auto m = match.get_access<read_t>();

      for (int i = 0; i < g.vertexNum; i++) {
        if (m[i] >= 4 && !g.has((m[i]-4), (m[(m[i]-4)]-4))){
          printf("Matching between vertices over non-exisiting edge!!! %d %d\n",(m[i]-4),(m[(m[i]-4)]-4));
          bad = true;
        }
      }
      if (bad){
        fflush(stdout);
        exit(1);
      }
    }



    {
      const auto read_t = sycl::access::mode::read;
      auto exp = expanded.get_access<read_t>();
      flag = exp[0];
    }
  } while(flag);

  //#ifdef NDEBUG


  {
    const auto read_t = sycl::access::mode::read;
    const auto read_write_t = sycl::access::mode::read_write;

    auto d = dist.get_access<read_t>();
    auto deg = degree.get_access<read_t>();

    auto s = start.get_access<read_t>();
    auto dep = depth.get_access<read_t>();
    const size_t depths = dep[0];
    const sycl::range DepthCountsRange{depths};
    sycl::buffer<int> DepthCounts{DepthCountsRange};

    auto dc_i = DepthCounts.get_access<read_write_t>();
    int degreeZero = 0;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      dc_i[depth_to_print] = 0;
    }
    //std::cout << "Distance from start is : " << std::endl;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      for (int i = 0; i < vertexNum; i++) {
        if (d[i] == depth_to_print){
          //printf("vertex %d dist %d start %d\n",i, d[i], s[i]);
          ++dc_i[depth_to_print];
        }
        if (deg[i] == 0)
          ++degreeZero;
      }
    }
    int sumOfReached = 0;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      sumOfReached += dc_i[depth_to_print];
      printf("vertices at depth %d : %d\n",depth_to_print, dc_i[depth_to_print]);
    }
    printf("Degree zero vertices %d\n", degreeZero);
    printf("Possibly augmentable vertices %d\n", dc_i[0]-degreeZero);
    printf("Unreached vertices %d\n", g.vertexNum-sumOfReached);


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
void testAlternatingBFSTree(sycl::queue &q, 
                CSRGraph & g,
                int & matchCount,
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<int> &dist,
                sycl::buffer<int> &pred,
                sycl::buffer<int> &start,
                sycl::buffer<int> &degree,
                sycl::buffer<int> &match,
                sycl::buffer<int> &requests,
                sycl::buffer<bool> &matchable,
                sycl::buffer<uint64_t> &bridgeVertex,
                sycl::buffer<int> &base,
                sycl::buffer<int> &forward,
                sycl::buffer<int> &backward,
                sycl::buffer<bool> &inb,
                const int vertexNum){

  constexpr const size_t SingletonSz = 1;

  const sycl::range Singleton{SingletonSz};

  const size_t numBlocks = vertexNum;
  const sycl::range VertexSize{numBlocks};

  const size_t threadsPerBlock = 32;
  const size_t totalThreads = numBlocks * threadsPerBlock;

  const sycl::range NumWorkItems{totalThreads};
  const sycl::range WorkGroupSize{threadsPerBlock};
  // Expanded
  sycl::buffer<bool> expanded{Singleton};
  sycl::buffer<int> dzs{Singleton};
  sycl::buffer<int> depth{Singleton};
  sycl::buffer<bool> ineligible{VertexSize};


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
    auto b_i = bridgeVertex.get_access<dwrite_t>();
    auto ineligible_i = ineligible.get_access<dwrite_t>();

    
    dzs_i[0] = 0;
    for (int i = 0; i < vertexNum; i++) {
      if (!deg[i])
        ++dzs_i[0];
      if (m[i] < 4){
        d[i] = 0;
      } else {
        d[i] = -1;
      }
      b_i[i] = 0;
      s[i] = i;
      p[i] = i;
      ineligible_i[i] = false;
    }
    dep[0] = -1;
    exp[0] = 0;
    printf("Num degree zero vertices %d\n", dzs_i[0]);
  }



  //printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);

  bool flag = false;
  bool blossomsContracted = false;
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
      auto base_i = base.get_access<read_t>(h);
      auto for_i = forward.get_access<read_t>(h);
      auto start_i = start.get_access<read_t>(h);
      auto inb_i = inb.get_access<read_t>(h);

      h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        //printf("hellow from item %lu thread %lu gr %lu w range %lu \n", item.get_global_linear_id(), threadIdx, src, r[0]);
                        
                        // Not a frontier vertex
                        // Ineligible vertices  are those which are sourced at a vertex which has been augmented (match 4+)
                        // or those newly contracted into a blossom (match 3).
                        if (dist_i[base_i[src]] != depth_i[0] || match_i[start_i[base_i[src]]] > 2) return;

                        // Even depth. expand into all neighbors
                          for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                            auto base_col = base_i[col];
                            // Avoid internal blossom edges.
                            if (base_i[src]!=base_col){
                              // Only interact with blossoms through the base.
                                if (dist_i[base_col] == -1){
                                  dist_i[base_col] = dist_i[src] + 1;
                                  pred_i[col] = src;
                                  expanded_i[0] = 1;
                                  // This is a blossom vertex.  Write to all the entries.
                                  if (inb_i[base_col]){
                                    auto base = base_col;
                                    auto curr_u = col;
                                    do{
                                      dist_i[curr_u] = dist_i[src] + 1;
                                      // not sure about this line
                                      //pred_i[curr_u] = src;
                                      curr_u = for_i[curr_u];
                                    } while(curr_u != base);
                                  }
                                }
                            }
                          }
                        
      });
    };
    q.submit(cg2);  

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
        if(depth_i[0]+1 == dist_i[i]) 
          start_i[i] = start_i[pred_i[i]];
      });
    };
    q.submit(cg3);



    {
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;
      bool bad = false;

      auto m = match.get_access<read_t>();
      auto pred_i = pred.get_access<read_t>();

      for (int i = 0; i < g.vertexNum; i++) {
        if (pred_i[i] != i  && !g.has(pred_i[i], i) ){
          printf("CATASTROPHIC ERROR! PRED EDGE %u - %u DNE\n",pred_i[i],i);
          bad = true;
        }
      }
      if (bad){
        printf("BFS FAIL!!!\n");
        fflush(stdout);
        exit(1);
      }else{
        printf("BFS SUCCESS!!!\n");
        fflush(stdout);
      }
    }

    // check for bridges.  Terminate a frontier prematurely if one is found.
    // A bridge is an unmatched edge between two even levels
    // or a matched edge between two odd levels.
    {
      const auto read_t = sycl::access::mode::read;
      // If depth is even, new frontier is odd, check for trivials
      auto dep = depth.get_access<read_t>();
      /*
      if (dep[0] % 2 == 0){
        printf("Frontier is odd!\n");
        augment_trivial_paths(q, 
                              matchCount,
                              pred,
                              dist,
                              start,
                              depth,
                              match,
                              vertexNum);  
      }
      // Match free to each other through bridges (red/blue).
      augment_bridges(q, 
          g,
          matchCount,
          rows, 
          cols, 
          bridgeVertex,
          pred,
          dist,
          start,
          depth,
          match,
          requests,
          matchable,
          ineligible,
          vertexNum);
      
      // Contract blossoms
      blossomsContracted = contract_blossoms(q, 
                                          matchCount,
                                          rows, 
                                          cols, 
                                          bridgeVertex,
                                          pred,
                                          dist,
                                          start,
                                          depth,
                                          match,
                                          requests,
                                          matchable,
                                          base,
                                          forward,
                                          backward,
                                          ineligible,
                                          vertexNum);
      */

    }
    {
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;
      bool bad = false;

      auto m = match.get_access<read_t>();

      for (int i = 0; i < g.vertexNum; i++) {
        if (m[i] >= 4 && !g.has((m[i]-4), (m[(m[i]-4)]-4))){
          printf("Matching between vertices over non-exisiting edge!!! %d %d\n",(m[i]-4),(m[(m[i]-4)]-4));
          bad = true;
        }
      }
      if (bad){
        fflush(stdout);
        exit(1);
      }
    }



    {
      const auto read_t = sycl::access::mode::read;
      auto exp = expanded.get_access<read_t>();
      flag = exp[0];
    }
  } while(flag);

  //#ifdef NDEBUG


  {
    const auto read_t = sycl::access::mode::read;
    const auto read_write_t = sycl::access::mode::read_write;

    auto d = dist.get_access<read_t>();
    auto deg = degree.get_access<read_t>();

    auto s = start.get_access<read_t>();
    auto dep = depth.get_access<read_t>();
    const size_t depths = dep[0];
    const sycl::range DepthCountsRange{depths};
    sycl::buffer<int> DepthCounts{DepthCountsRange};

    auto dc_i = DepthCounts.get_access<read_write_t>();
    int degreeZero = 0;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      dc_i[depth_to_print] = 0;
    }
    //std::cout << "Distance from start is : " << std::endl;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      for (int i = 0; i < vertexNum; i++) {
        if (d[i] == depth_to_print){
          //printf("vertex %d dist %d start %d\n",i, d[i], s[i]);
          ++dc_i[depth_to_print];
        }
        if (deg[i] == 0)
          ++degreeZero;
      }
    }
    int sumOfReached = 0;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      sumOfReached += dc_i[depth_to_print];
      printf("vertices at depth %d : %d\n",depth_to_print, dc_i[depth_to_print]);
    }
    printf("Degree zero vertices %d\n", degreeZero);
    printf("Possibly augmentable vertices %d\n", dc_i[0]-degreeZero);
    printf("Unreached vertices %d\n", g.vertexNum-sumOfReached);


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