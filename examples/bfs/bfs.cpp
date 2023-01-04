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

int main(int argc, char *argv[]) {

  Config config = parseArgs(argc,argv);
  printf("\nGraph file: %s",config.graphFileName);
  printf("\nUUID: %s\n",config.outputFilePrefix);


  CSRGraph graph = createCSRGraphFromFile(config.graphFileName);
  performChecks(graph, config);

  constexpr const size_t N = 100000;
  const sycl::range RowSize{graph.vertexNum+1};
  const sycl::range ColSize{graph.edgeNum*2};
  const sycl::range VertexSize{graph.vertexNum};

  // Device input vectors
  sycl::buffer<unsigned int> bufA{graph.srcPtr, RowSize};
  sycl::buffer<unsigned int> bufB{graph.dst, ColSize};
  sycl::buffer<int> bufC{graph.degree, VertexSize};

  // Device intermediate vector
  sycl::buffer<int> bufD{VertexSize};

  // Device output vector
  sycl::buffer<int> bufE{VertexSize};

  // Initialize input data
  {
    const auto dwrite_t = sycl::access::mode::discard_write;

    auto h_e = bufE.get_access<dwrite_t>();
    for (int i = 0; i < N; i++) {
      h_e[i] = 0;
    }
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
    const auto write_t = sycl::access::mode::write;

    auto c = bufC.get_access<read_t>(h);
    auto e = bufE.get_access<write_t>(h);

    h.parallel_for(VertexSize,
                   [=](sycl::id<1> i) { e[i] = c[i]; });
  };

  myQueue.submit(cg);

  {
    const auto read_t = sycl::access::mode::read;
    auto h_e = bufE.get_access<read_t>();
    double sum = 0.0f;
    for (int i = 0; i < N; i++) {
      sum += h_e[i];
    }
    std::cout << "Sum is : " << sum << std::endl;
  }

  return 0;
}
