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

#include <CL/sycl.hpp>

void reduction(sycl::queue &q, std::vector<int> &data, std::vector<int> &flush,
               int iter, int work_group_size) {
  const size_t data_size = data.size();
  const size_t flush_size = flush.size();
  int sum = 0;

  const sycl::property_list props = {sycl::property::buffer::use_host_ptr()};
  // int vec_size =
  // q.get_device().get_info<sycl::info::device::native_vector_width_int>();
  int num_work_items = data_size / work_group_size;
  sycl::buffer<int> buf(data.data(), data_size, props);
  sycl::buffer<int> flush_buf(flush.data(), flush_size, props);
  sycl::buffer<int> sum_buf(&sum, 1, props);

  //init_data(q, buf, data_size);

  double elapsed = 0;
  for (int i = 0; i < iter; i++) {
    q.submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(1, [=](auto index) { sum_acc[index] = 0; });
    });
    // flush the cache
    q.submit([&](auto &h) {
      sycl::accessor flush_acc(flush_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(flush_size, [=](auto index) { flush_acc[index] = 1; });
    });

    //Timer timer;
    // reductionMapToHWVector main begin
    q.submit([&](auto &h) {
      sycl::accessor buf_acc(buf, h, sycl::read_only);
      sycl::accessor<int, 1, sycl::access::mode::read_write,
                     sycl::access::target::local>
          scratch(work_group_size, h);
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);

      h.parallel_for(
          sycl::nd_range<1>(num_work_items, work_group_size),
          [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
            auto v =
                sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device,
                                 sycl::access::address_space::global_space>(
                    sum_acc[0]);
            int sum = 0;
            int glob_id = item.get_global_id();
            int loc_id = item.get_local_id();
            for (unsigned int i = glob_id; i < data_size; i += num_work_items)
              sum += buf_acc[i];
            scratch[loc_id] = sum;

            for (int i = work_group_size / 2; i > 0; i >>= 1) {
              item.barrier(sycl::access::fence_space::local_space);
              if (loc_id < i)
                scratch[loc_id] += scratch[loc_id + i];
            }

            if (loc_id == 0)
              v.fetch_add(scratch[0]);
          });
    });
    q.wait();
    //elapsed += timer.Elapsed();
    sycl::host_accessor h_acc(sum_buf);
    sum = h_acc[0];
  }
  elapsed = elapsed / iter;
  std::string msg = "with work-groups=" + std::to_string(work_group_size);
  //check_result(elapsed, msg, sum);
} // reduction end

int main(int argc, char *argv[]) {

  Config config = parseArgs(argc,argv);
  printf("\nGraph file: %s",config.graphFileName);
  printf("\nUUID: %s\n",config.outputFilePrefix);


  CSRGraph graph = createCSRGraphFromFile(config.graphFileName);
  performChecks(graph, config);
  printf("finished checking\n");
  constexpr const size_t startSz = 1;

  const sycl::range RowSize{graph.vertexNum+1};
  const sycl::range ColSize{graph.edgeNum*2};
  const sycl::range VertexSize{graph.vertexNum};
  const sycl::range StartSize{startSz};

  // Device input vectors
  sycl::buffer<unsigned int> rows{graph.srcPtr, RowSize};
  sycl::buffer<unsigned int> cols{graph.dst, ColSize};
  sycl::buffer<int> degree{graph.degree, VertexSize};

  // Device intermediate vectorstart
  // Frontier
  sycl::buffer<bool> frontier{VertexSize};

  // Device output vector
  // Dist
  sycl::buffer<int> dist{VertexSize};
  // Start
  sycl::buffer<int> start{StartSize};
  // Depth
  sycl::buffer<int> depth{StartSize};

  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_write_t = sycl::access::mode::read_write;
    auto h_c = degree.get_access<read_t>();
    auto h_d = frontier.get_access<dwrite_t>();
    auto h_e = dist.get_access<dwrite_t>();
    auto h_f = start.get_access<read_write_t>();
    auto h_h = depth.get_access<dwrite_t>();

    for (int i = 0; i < graph.vertexNum; i++) {
      h_d[i] = false;
      h_e[i] = 0;
      if(!h_f[0] && h_c[i]) h_f[0] = i;
    }
    h_d[h_f[0]] = true;
    h_h[0] = 0;
    std::cout << "start " << h_f[0] << " degree " << h_c[h_f[0]] << std::endl;
  }

  auto CUDASelector = [](sycl::device const &dev) {
    if (dev.get_platform().get_backend() == sycl::backend::ext_oneapi_cuda) {
      std::cout << " CUDA device found " << std::endl;
      return 1;
    } else {
      return -1;
    }
  };
  sycl::queue myQueue{CUDASelector};

  // Command Group creation
  auto cg = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto read_write_t = sycl::access::mode::read_write;
    const auto dwrite_t = sycl::access::mode::discard_write;

    auto rows_i = rows.get_access<read_t>(h);
    auto cols_i = cols.get_access<read_t>(h);

    auto frontier_i = frontier.get_access<read_t>(h);
    // dist
    auto dist_i = dist.get_access<read_write_t>(h);

    h.parallel_for(VertexSize,
                   [=](sycl::id<1> i) { if(frontier_i[i]) dist_i[i] = dist_i[i]; });
  };

  myQueue.submit(cg);

  {
    const auto read_t = sycl::access::mode::read;
    auto h_e = dist.get_access<read_t>();
    double sum = 0.0f;
    for (int i = 0; i < graph.vertexNum; i++) {
      sum += h_e[i];
    }
    std::cout << "Sum is : " << sum << std::endl;
  }

  return 0;
}
