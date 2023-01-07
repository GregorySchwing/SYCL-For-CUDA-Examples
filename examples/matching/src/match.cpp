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
#include "bfs.h"

#include <CL/sycl.hpp>
// For min(T a, T b)

// Implementation of the user-defined binary function.
template <typename T>
bool comp(T element1, T element2){
    // Returning the smaller value.
    return (element1 < element2);    
}

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
  constexpr const size_t TripletonSz = 3;

  const sycl::range RowSize{graph.vertexNum+1};
  const sycl::range ColSize{graph.edgeNum*2};
  const sycl::range VertexSize{graph.vertexNum};
  const sycl::range HashSize{hashArraySize};

  const sycl::range Singleton{SingletonSz};
  const sycl::range Doubleton{DoubletonSz};
  const sycl::range Tripleton{TripletonSz};

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
  sycl::buffer<bool> keepMatching{Singleton};
  sycl::buffer<unsigned int> colsum {Tripleton};


  // Initialize input data
  {
    const auto read_t = sycl::access::mode::read;
    const auto dwrite_t = sycl::access::mode::discard_write;
    auto deg = degree.get_access<read_t>();
    auto m = match.get_access<dwrite_t>();
    auto r = requests.get_access<dwrite_t>();

    auto sb = selectBarrier.get_access<dwrite_t>();
    auto km = keepMatching.get_access<dwrite_t>();
    auto cs = colsum.get_access<dwrite_t>();

    for (int i = 0; i < graph.vertexNum; i++) {
      m[i] = 0;
      r[i] = 0;
    }
    sb[0] = selectBarrier_default;
    km[0] = true;
    cs[0] = 0;
    cs[1] = 0;
    cs[2] = 0;

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

  bool flag = false;
  do{
    {
      const auto write_t = sycl::access::mode::write;
      auto km = keepMatching.get_access<write_t>();
      km[0] = false;
    }

    // Color vertices
    // Request vertices - one workitem per workgroup
    // Command Group creation
    auto cg = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto read_write_t = sycl::access::mode::read_write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto write_t = sycl::access::mode::write;

      // dist
      auto sb = selectBarrier.get_access<read_t>(h);
      auto randNum = rand();
      auto aMD5K = MD5K.get_access<read_t>(h);
      auto aMD5R = MD5R.get_access<read_t>(h);

      auto match_i = match.get_access<read_write_t>(h);
      auto km = keepMatching.get_access<write_t>(h);

      h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        // Unnecessary
        // if (i >= nrVertices) return;

        //Can this vertex still be matched?
        if (match_i[i] >= 2) return;
        
        // cant be type dwrite_t (must be write_t) or this is always reacher somehow.
        km[0] = true;
        // Some vertices can still match.
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
          g *= randNum;
        }
        match_i[i] = ((h0 + h1 + h2 + h3) < sb[0] ? 0 : 1);
      });
    };
    myQueue.submit(cg);

    // Request vertices - one workitem per workgroup
    // Command Group creation
    auto cg2 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto rows_i = rows.get_access<read_t>(h);
      auto cols_i = cols.get_access<read_t>(h);
      auto match_i = match.get_access<read_t>(h);
      auto requests_i = requests.get_access<dwrite_t>(h);


      h.parallel_for(VertexSize,
                    [=](sycl::id<1> src) {                         
                        //Look at all blue vertices and let them make requests.
                        if (match_i[src] == 0)
                        {
                          int dead = 1;
                        
                          for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                            auto col = cols_i[col_index];

                            const auto nm = match_i[col];

                            //Do we have an unmatched neighbour?
                            if (nm < 4)
                            {
                              //Is this neighbour red?
                              if (nm == 1)
                              {
                                //Propose to this neighbour.
                                requests_i[src] = col;
                                return;
                              }
                              
                              dead = 0;
                            }
                          }

                          requests_i[src] = nrVertices + dead;
                        }
                        else
                        {
                          //Clear request value.
                          requests_i[src] = nrVertices;
                        }                    
      });
    };
    myQueue.submit(cg2);

    // Respond vertices - one workitem per workgroup
    // Command Group creation
    auto cg3 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto rows_i = rows.get_access<read_t>(h);
      auto cols_i = cols.get_access<read_t>(h);
      auto match_i = match.get_access<read_t>(h);
      auto requests_i = requests.get_access<read_write_t>(h);


      h.parallel_for(VertexSize,
                    [=](sycl::id<1> src) {                         
                        //Look at all red vertices.
                        if (match_i[src] == 1)
                        {
                          //Select first available proposer.
                          for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                            auto col = cols_i[col_index];
                            //Only respond to blue neighbours.
                            if (match_i[col] == 0)
                            {
                              //Avoid data thrashing be only looking at the request value of blue neighbours.
                              if (requests_i[col] == src)
                              {
                                requests_i[src] = col;
                                return;
                              }
                            }
                          }
                        }               
      });
    };
    myQueue.submit(cg3);

    // Match vertices - one workitem per workgroup
    // Command Group creation
    auto cg4 = [&](sycl::handler &h) {    
      const auto read_t = sycl::access::mode::read;
      const auto write_t = sycl::access::mode::write;
      const auto dwrite_t = sycl::access::mode::discard_write;
      const auto read_write_t = sycl::access::mode::read_write;

      auto match_i = match.get_access<dwrite_t>(h);
      auto requests_i = requests.get_access<read_t>(h);


      h.parallel_for(VertexSize,
                    [=](sycl::id<1> src) {                         

                        const auto r = requests_i[src];

                        //Only unmatched vertices make requests.
                        if (r == nrVertices + 1)
                        {
                          //This is vertex without any available neighbours, discard it.
                          match_i[src] = 2;
                        }
                        else if (r < nrVertices)
                        {
                          //This vertex has made a valid request.
                          if (requests_i[r] == src)
                          {
                            //Match the vertices if the request was mutual.
                            // cant get this compile
                            //  match_i[src] = 4 + min(src, r);
                            if (src < r)
                              match_i[src] = 4 + src;
                            else 
                              match_i[src] = 4 + r;
                          }
                        }            
      });
    };
    myQueue.submit(cg4);  

    {
      const auto read_t = sycl::access::mode::read;
      auto km = keepMatching.get_access<read_t>();
      flag = km[0];
    }

    #ifdef NDEBUG
    {
      const auto read_t = sycl::access::mode::read;
      const auto read_write_t = sycl::access::mode::read_write;

      auto m = match.get_access<read_t>();
      auto cs = colsum.get_access<read_write_t>();
      cs[0] = 0;
      cs[1] = 0;
      cs[2] = 0;

      for (int i = 0; i < graph.vertexNum; i++) {
        if(m[i] < 4)
          ++cs[m[i]];
        //printf("%d %d\n",i,m[i]);
      }
      std::cout << "red count : " << cs[0] << std::endl;
      std::cout << "blue count : " << cs[1] << std::endl;
      std::cout << "dead count : " << cs[2] << std::endl;
      std::cout << "matched count : " << graph.vertexNum-(cs[0]+cs[1]+cs[2]) << std::endl;
    }
    #endif
    // just to keep from entering an inf loop till all matching logic is done.
    //flag = false;
  } while(flag);


  {
    const auto read_t = sycl::access::mode::read;
    const auto read_write_t = sycl::access::mode::read_write;

    auto m = match.get_access<read_t>();
    auto cs = colsum.get_access<read_write_t>();

    cs[0] = 0;
    cs[1] = 0;
    cs[2] = 0;

    for (int i = 0; i < graph.vertexNum; i++) {
      if(m[i] < 4)
        ++cs[m[i]];
      //printf("%d %d\n",i,m[i]);
    }
    std::cout << "red count : " << cs[0] << std::endl;
    std::cout << "blue count : " << cs[1] << std::endl;
    std::cout << "dead count : " << cs[2] << std::endl;
    std::cout << "matched count : " << graph.vertexNum-(cs[0]+cs[1]+cs[2]) << std::endl;
  
  }

  // Initialize input data
  sycl::buffer<int> dist{VertexSize};
  
  bfs(myQueue,
      rows, 
      cols, 
      dist,
      degree,
      match,
      graph.vertexNum);

  return 0;
}
