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

//Nothing-up-my-sleeve working constants from SHA-256.
const uint dMD5K[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
				0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
				0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
				0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
				0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
				0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
				0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
				0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

//Rotations from MD5.
const uint dMD5R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
				5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
				4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
				6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

const unsigned int hashArraySize = 64;
//const unsigned int selectBarrier_default = 0x8000000;
const	unsigned int selectBarrier_default = 0x88B81733;
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
  const sycl::range HashSize{hashArraySize};

  

  const sycl::range Singleton{SingletonSz};
  const sycl::range Doubleton{DoubletonSz};

  // Device input vectors
  sycl::buffer<unsigned int> rows{graph.srcPtr, RowSize};
  sycl::buffer<unsigned int> cols{graph.dst, ColSize};
  sycl::buffer<int> degree{graph.degree, VertexSize};

  sycl::buffer<unsigned int> MD5K{dMD5K, HashSize};
  sycl::buffer<unsigned int> MD5R{dMD5R, HashSize};

  // Device output vector
  sycl::buffer<int> match{VertexSize};
  // Intermediate vector
  sycl::buffer<int> requests{VertexSize};

  // Determines distribution of red/blue
  sycl::buffer<unsigned int> selectBarrier {Singleton};
  sycl::buffer<unsigned int> random {Singleton};
  sycl::buffer<bool> keepMatching{Singleton};
  sycl::buffer<unsigned int> colsum {Doubleton};


  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto dwrite_t = sycl::access::mode::discard_write;
    auto deg = degree.get_access<read_t>();
    auto m = match.get_access<dwrite_t>();
    auto r = requests.get_access<dwrite_t>();

    auto sb = selectBarrier.get_access<dwrite_t>();
    auto km = keepMatching.get_access<dwrite_t>();
    auto rand = random.get_access<dwrite_t>();
    auto cs = colsum.get_access<dwrite_t>();

    for (int i = 0; i < graph.vertexNum; i++) {
      m[i] = 0;
      r[i] = 0;
    }
    rand[0] = selectBarrier_default;
    sb[0] = selectBarrier_default;
    km[0] = true;
    cs[0] = 0;
    cs[1] = 0;

    std::cout << "selectBarrier " << sb[0] << std::endl;
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

  const int numBlocks = graph.vertexNum;

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
      auto km = keepMatching.get_access<write_t>();
      km[0] = false;
    }

    // Color vertices
    // Command Group creation
    auto cg = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto read_write_t = sycl::access::mode::read_write;

      // dist
      auto sb = selectBarrier.get_access<read_t>(h);
      auto random_a = random.get_access<read_t>(h);

      auto aMD5K = MD5K.get_access<read_t>(h);
      auto aMD5R = MD5R.get_access<read_t>(h);

      auto match_i = match.get_access<read_write_t>(h);

      h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        // Unnecessary
        // if (i >= nrVertices) return;

        //Can this vertex still be matched?
        if (match_i[i] >= 2) return;

        // TODO: template the hash functions in hashing/ for testing here.
        //Start hashing.
        uint h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
        uint a = h0, b = h1, c = h2, d = h3, e, f, g = i;

        for (int j = 0; j < 16; ++j)
        {
          f = (b & c) | ((~b) & d);

          e = d;
          d = c;
          c = b;
          b += LEFTROTATE(a + f + aMD5K[j] + g, aMD5R[j]);
          a = e;

          h0 += a;
          h1 += b;
          h2 += c;
          h3 += d;

          g *= random_a[0];
        }
        match_i[i] = ((h0 + h1 + h2 + h3) < sb[0] ? 0 : 1);
      });
    };
    myQueue.submit(cg);

    {
      const auto read_t = sycl::access::mode::read;
      auto km = keepMatching.get_access<read_t>();
      flag = km[0];
    }
  } while(flag);
  {
    const auto read_t = sycl::access::mode::read;
    const auto read_write_t = sycl::access::mode::read_write;

    auto m = match.get_access<read_t>();
    auto cs = colsum.get_access<read_write_t>();

    for (int i = 0; i < graph.vertexNum; i++) {
      ++cs[m[i]];
      //printf("%d %d\n",i,m[i]);
    }
    std::cout << "red count : " << cs[0] << std::endl;
    std::cout << "blue count : " << cs[1] << std::endl;
  }
  return 0;
}
