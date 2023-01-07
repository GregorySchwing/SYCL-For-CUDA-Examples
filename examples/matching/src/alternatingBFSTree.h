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
#include <CL/sycl.hpp>

void alternatingBFSTree(sycl::queue &q, 
                sycl::buffer<unsigned int> &rows, 
                sycl::buffer<unsigned int> &cols, 
                sycl::buffer<int> &dist,
                sycl::buffer<int> &degree,
                sycl::buffer<int> &match,
                int vertexNum){

  constexpr const size_t SingletonSz = 1;

  const sycl::range Singleton{SingletonSz};
  // Depth
  sycl::buffer<int> depth{Singleton};
  // Expanded
  sycl::buffer<bool> expanded{Singleton};


  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;
    auto deg = degree.get_access<read_t>();
    auto m = match.get_access<read_t>();

    auto d = dist.get_access<dwrite_t>();
    auto dep = depth.get_access<dwrite_t>();
    auto exp = expanded.get_access<dwrite_t>();

    for (int i = 0; i < vertexNum; i++) {
      if (m[i] < 4)
        d[i] = 0;
      else
        d[i] = -1;
    }
    dep[0] = 0;
    exp[0] = 0;
  }

  const int numBlocks = vertexNum;

  const size_t threadsPerBlock = 32;
  const size_t totalThreads = numBlocks * threadsPerBlock;

  const sycl::range NumWorkItems{totalThreads};
  const sycl::range WorkGroupSize{threadsPerBlock};
  sycl::nd_range<1> test{NumWorkItems, WorkGroupSize};


  printf("get_local_range %lu get_global_range %lu get_group_range %lu \n", test.get_local_range()[0],  test.get_global_range()[0],  test.get_group_range()[0]);

  bool flag = false;
  do{

    {
      const auto write_t = sycl::access::mode::write;
      auto exp = expanded.get_access<write_t>();
      exp[0] = false;
    }

    // Command Group creation
    auto cg = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto rows_i = rows.get_access<read_t>(h);
      auto cols_i = cols.get_access<read_t>(h);
      auto depth_i = depth.get_access<read_t>(h);
      auto expanded_i = expanded.get_access<write_t>(h);
      auto dist_i = dist.get_access<read_write_t>(h);

      h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        //printf("hellow from item %lu thread %lu gr %lu w range %lu \n", item.get_global_linear_id(), threadIdx, src, r[0]);
                        
                        // Not a frontier vertex
                        if (dist_i[src] != depth_i[0]) return;
                        
                        for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                          auto col = cols_i[col_index];
                          // atomic isn't neccessary since I don't set predecessor.
                          // even if I set predecessor, all races remain in the universe of
                          // valid solutions.
                          if (dist_i[col] == -1){
                            dist_i[col] = dist_i[src] + 1;
                            expanded_i[0] = 1;
                          }
                        }                     
      });
    };
    q.submit(cg);

    // Command Group creation
    auto cg2 = [&](sycl::handler &h) {    
      const auto read_write_t = sycl::access::mode::read_write;
      auto dep = depth.get_access<read_write_t>(h);
      h.parallel_for(Singleton,
                    [=](sycl::id<1> i) { dep[0] = dep[0]+1; });
    };
    q.submit(cg2);

    {
      const auto read_t = sycl::access::mode::read;
      auto exp = expanded.get_access<read_t>();
      flag = exp[0];
    }
  } while(flag);

  {
    const auto read_t = sycl::access::mode::read;
    auto d = dist.get_access<read_t>();
    auto dep = depth.get_access<read_t>();

    std::cout << "Distance from start is : " << std::endl;
    for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
      for (int i = 0; i < vertexNum; i++) {
        if (d[i] == depth_to_print) printf("vertex %d dist %d\n",i, d[i]);
      }
    }
    std::cout << std::endl;
  }

  return;
}
