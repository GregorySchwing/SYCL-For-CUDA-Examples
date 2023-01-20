// Need to detect 3 cases (2 augments and 1 non-augmenting)
// Case 1 : trivial augmenting path (end of tree (unmatched) odd depth (>1) vertex)
// Case 2 : non-trivial augmenting path (end of tree at even depth aux matched to a neighbor in my depth)
// Case 3 : blossom


// Successful augment/blossom
// Level must be > 0.

// Odd level (blossom/augpath)
//  primary match (true) (indicates direction of tree growth.)
//  secondary match (true) (depth of both matched vertices are equal.)
//  odd levels can't match in secondary match unless the branch has no deeper vertex.

// Even level (blossom/augpath)
//  secondary match (true) (depth of both matched vertices are equal.)

// Matching contains minimum vertex of matched pair.

// If we contract a blossom, we need to perform this whole algorithm again
// starting with alternating BFS Tree.

// If we only augment paths, I don't think we need another iteration.

// Therefore, there are two kernels, augment_bridges which will only augment paths
// Then if any start vertices remain, they must be blossom structures or unaugmentable.
// If any blossoms are contracted in augment_b, reconstruct BFS.

// do {
//      BFS
//      AuxMatch
//      augment_bridges
//      augment_b_returned_true = augment_b
// } while(augment_b_returned_true)


#ifndef AUGMENT_h
#define AUGMENT_h
#include "match.h"

// Only augments bridges.  Not trivial paths or blossom contractions
void augment_bridges(sycl::queue &q, 
                CSRGraph & g,
                int & matchCount,
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<uint64_t> &bridgeVertex, 
                sycl::buffer<int> &pred,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                sycl::buffer<int> &requests,
                sycl::buffer<bool> &matchable,
                const int vertexNum){


    constexpr const size_t SingletonSz = 1;
    const sycl::range Singleton{SingletonSz};

    constexpr const size_t TripletonSz = 3;
    const sycl::range Tripleton{TripletonSz};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};

    const size_t threadsPerBlock = 32;
    const size_t totalThreads = numBlocks * threadsPerBlock;

    const sycl::range NumWorkItems{totalThreads};
    const sycl::range WorkGroupSize{threadsPerBlock};
  
    sycl::buffer<bool> keepMatching{Singleton};
    sycl::buffer<unsigned int> selectBarrier {Singleton};
    sycl::buffer<unsigned int> colsum {Tripleton};

    // Initialize input data
    {
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto dwrite_t = sycl::access::mode::discard_write;
        const auto read_write_t = sycl::access::mode::read_write;
        auto km = keepMatching.get_access<write_t>();
        auto sb = selectBarrier.get_access<write_t>();

        km[0] = true;
        sb[0] = 0x88B81733;
    }

    bool flag = false;
    int iteration = 0;
    do{
        {
        const auto write_t = sycl::access::mode::write;
        auto km = keepMatching.get_access<write_t>();
        km[0] = false;
        }
        auto cg = [&](sycl::handler &h) {    
        const auto dwrite_t = sycl::access::mode::discard_write;
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;
        auto m_i = match.get_access<read_write_t>(h);

        auto matchable_i = matchable.get_access<dwrite_t>(h);
        h.parallel_for(VertexSize,
                        [=](sycl::id<1> i) { 
            matchable_i[i] = false;
            if (m_i[i]<4) m_i[i]=2;
        });
        };
        q.submit(cg);

        // Command Group creation
        // sets vertices in this next frontier which can augment/blossom and thus terminate.
        auto cg4 = [&](sycl::handler &h) { 
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto depth_i = depth.get_access<read_t>(h);
        auto match_i = match.get_access<read_t>(h);
        auto matchable_i = matchable.get_access<write_t>(h);

        //auto b_i = bridgeVertex.get_access<write_t>(h);
        auto start_i = start.get_access<read_t>(h);

        auto dist_i = dist.get_access<read_t>(h);
        auto pred_i = pred.get_access<read_t>(h);

        h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                            sycl::group<1> gr = item.get_group();
                            sycl::range<1> r = gr.get_local_range();
                            size_t src = gr.get_group_linear_id();
                            size_t blockDim = r[0];
                            size_t threadIdx = item.get_local_id();
                            
                            //printf("src %lu dist %d depth %d start %d\n", src, dist_i[src], depth_i[0], start_i[src]);

                            // Not a new frontier vertex with an unmatched src.
                            if (dist_i[src] != (depth_i[0]+1)  || match_i[start_i[src]] >= 4) return;
                            //printf("possible to color the src in %lu\n", src);
                            // If you win the first race, you make sure you get written in your slot then return.
                            //sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                            //sycl::access::address_space::global_space> ref_b_src(b_i[start_i[src]]);

                            // A bridge is an unmatched edge between two even levels
                            if ((depth_i[0]+1) % 2 == 0){
                            for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                            //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){                            
                                auto col = cols_i[col_index];
                                // An edge to a vertex in my even level.
                                if (dist_i[col] == dist_i[src]){
                                    // If you win the first race, you make sure you get written in your slot then return. 
                                    if (match_i[start_i[col]] < 4){   
                                        // Prevent blossoms
                                        if(start_i[src] != start_i[col]){
                                            matchable_i[start_i[src]] = true;
                                            matchable_i[src] = true;
                                            //printf("Matchable pair %lu %u\n", src, col);
                                            return;
                                        }
                                    }
                                }
                            }
                            } else {
                            for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                            //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                                auto col = cols_i[col_index];
                                // A matched edge to a vertex in my odd level.
                                if (match_i[col] == match_i[src] &&
                                    dist_i[col] == dist_i[src]){
                                    // If you win the first race, you make sure you get written in your slot then return.      
                                    // Prevent blossoms
                                    if(start_i[src] != start_i[col]){
                                        matchable_i[start_i[src]] = true;
                                        matchable_i[src] = true;
                                        //printf("Matchable pair %lu %u\n", src, col);
                                        return;
                                    }
                                }
                            }
                            }       
        });
        };
        q.submit(cg4);
        //fflush(stdout);

        // Color vertices
        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cgC = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto read_write_t = sycl::access::mode::read_write;
            const auto dwrite_t = sycl::access::mode::discard_write;
            const auto write_t = sycl::access::mode::write;

            // dist
            auto sb = selectBarrier.get_access<read_t>(h);
            auto randNum = rand();
            auto aMD5K = MD5K.get_access<read_t>(h);
            auto aMD5R = MD5R.get_access<read_t>(h);

            auto dist_i = dist.get_access<read_t>(h);

            auto matchable_i = matchable.get_access<read_t>(h);
            auto match_i = match.get_access<read_write_t>(h);
            auto km = keepMatching.get_access<write_t>(h);

            h.parallel_for(VertexSize,
                            [=](sycl::id<1> i) { 
                // Unnecessary
                // if (i >= vertexNum) return;

                //The dest 0 vertices are colored red/blue
                if (!matchable_i[i] || dist_i[i] != 0 || match_i[i] >= 4) return;

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
        q.submit(cgC);


        // check for bridges.  Terminate a frontier prematurely if one is found.
        // A bridge is an unmatched edge between two even levels
        // or a matched edge between two odd levels.

        // Command Group creation
        // sets vertices in this next frontier which can augment/blossom and thus terminate.
        auto cg5 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto depth_i = depth.get_access<read_t>(h);
        auto match_i = match.get_access<read_t>(h);
        auto matchable_i = matchable.get_access<write_t>(h);

        auto requests_i = requests.get_access<write_t>(h);

        //auto b_i = bridgeVertex.get_access<write_t>(h);
        auto start_i = start.get_access<read_t>(h);

        auto dist_i = dist.get_access<read_t>(h);
        auto pred_i = pred.get_access<read_t>(h);

        h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                            sycl::group<1> gr = item.get_group();
                            sycl::range<1> r = gr.get_local_range();
                            size_t src = gr.get_group_linear_id();
                            size_t blockDim = r[0];
                            size_t threadIdx = item.get_local_id();
                            auto srcStart = start_i[src];
                            // Not a new frontier vertex with a matchable src.
                            if (!matchable_i[src] || dist_i[src] != depth_i[0]+1)
                                return; 

                            // I am blue
                            if (match_i[srcStart] == 0){
                            int dead = 1;

                            // A bridge is an unmatched edge between two even levels
                            if ((depth_i[0]+1) % 2 == 0){
                                for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                                //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){                            
                                auto col = cols_i[col_index];
                                // An edge to a vertex in my even level.
                                if (dist_i[col] == dist_i[src]){

                                    const auto nm = match_i[start_i[col]];

                                    //Do we have an unmatched neighbour?
                                    if (nm < 4)
                                    {
                                    //Is this neighbour red?
                                    if (nm == 1)
                                    {
                                        //Propose to this neighbour.
                                        requests_i[srcStart] = start_i[col];
                                        return;
                                    }
                                    
                                    dead = 0;
                                    }
                                }
                                }
                                // Dont bother killing vertices.
                                // All the neighbors tried.
                                //requests_i[src] = vertexNum + dead;
                            } else {
                                for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                                //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                                auto col = cols_i[col_index];
                                // A matched edge to a vertex in my odd level.
                                if (match_i[col] == match_i[src] &&
                                    dist_i[col] == dist_i[src]){

                                    const auto nm = match_i[start_i[col]];

                                    //Do we have an unmatched neighbour?
                                    if (nm < 4)
                                    {
                                    //Is this neighbour red?
                                    if (nm == 1)
                                    {
                                        //Propose to this neighbour.
                                        requests_i[srcStart] = start_i[col];
                                        return;
                                    }
                                    
                                    dead = 0;
                                    }
                                }
                                // Dont bother killing vertices.
                                // All the neighbors tried.
                                //requests_i[src] = vertexNum + dead;                            }
                                }       
                            } 
                            }
                            //else
                            //{
                            //Clear request value.
                            //requests_i[src] = vertexNum;
                            //}  
        });
        };
        q.submit(cg5);


        // Command Group creation
        // sets vertices in this next frontier which can augment/blossom and thus terminate.
        auto cg6 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto depth_i = depth.get_access<read_t>(h);
        auto match_i = match.get_access<read_t>(h);
        auto matchable_i = matchable.get_access<write_t>(h);

        auto requests_i = requests.get_access<write_t>(h);

        auto b_i = bridgeVertex.get_access<write_t>(h);
        auto start_i = start.get_access<read_t>(h);

        auto dist_i = dist.get_access<read_t>(h);
        auto pred_i = pred.get_access<read_t>(h);

        h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                            sycl::group<1> gr = item.get_group();
                            sycl::range<1> r = gr.get_local_range();
                            size_t src = gr.get_group_linear_id();
                            size_t blockDim = r[0];
                            size_t threadIdx = item.get_local_id();
                            auto srcStart = start_i[src];
                            // Not a new frontier vertex with a matchable src.
                            if (!matchable_i[src] || dist_i[src] != depth_i[0]+1)
                                return; 


                            // If you win the first race, you make sure you get written in your slot then return.
                            //sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                            //sycl::access::address_space::global_space> ref_b_src(b_i[start_i[src]]);

                            // I am red
                            if (match_i[srcStart] == 1){
                            int dead = 1;

                            // A bridge is an unmatched edge between two even levels
                            if ((depth_i[0]+1) % 2 == 0){
                                for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                                //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){                            
                                auto col = cols_i[col_index];
                                // An edge to a vertex in my even level.
                                if (dist_i[col] == dist_i[src]){

                                    const auto nm = match_i[start_i[col]];
                                    //Only respond to blue neighbours.
                                    if (nm == 0)
                                    {
                                    //Avoid data thrashing be only looking at the request value of blue neighbours.
                                    if (requests_i[start_i[col]] == srcStart)
                                    {
                                        requests_i[srcStart] = start_i[col];
                                        uint32_t leastSignificantWord = src;
                                        uint32_t mostSignificantWord = col;
                                        uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
                                        b_i[srcStart] = edgePair;
                                        return;
                                    }
                                    }
                                }
                                }
                                // Dont bother killing vertices.
                                // All the neighbors tried.
                                //requests_i[src] = vertexNum + dead;
                            } else {
                                for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                                //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                                auto col = cols_i[col_index];
                                // A matched edge to a vertex in my odd level.
                                if (match_i[col] == match_i[src] &&
                                    dist_i[col] == dist_i[src]){

                                    const auto nm = match_i[start_i[col]];
                                    //Only respond to blue neighbours.
                                    if (nm == 0)
                                    {
                                    //Avoid data thrashing be only looking at the request value of blue neighbours.
                                    if (requests_i[start_i[col]] == start_i[src])
                                    {
                                        requests_i[start_i[src]] = start_i[col];
                                        uint32_t leastSignificantWord = src;
                                        uint32_t mostSignificantWord = col;
                                        uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
                                        b_i[start_i[src]] = edgePair;
                                        return;
                                    }
                                    }
                                }
                                // Dont bother killing vertices.
                                // All the neighbors tried.
                                //requests_i[src] = vertexNum + dead;                            }
                                }       
                            } 
                            }
                            //else
                            //{
                            //Clear request value.
                            //requests_i[src] = vertexNum;
                            //}  
        });
        };
        q.submit(cg6);



        // Command Group creation
        // sets vertices in this next frontier which can augment/blossom and thus terminate.
        auto cg7 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto depth_i = depth.get_access<read_t>(h);
        auto match_i = match.get_access<write_t>(h);
        auto matchable_i = matchable.get_access<write_t>(h);

        auto requests_i = requests.get_access<write_t>(h);

        //auto b_i = bridgeVertex.get_access<write_t>(h);
        auto start_i = start.get_access<read_t>(h);
        auto b_i = bridgeVertex.get_access<read_write_t>(h);

        auto dist_i = dist.get_access<read_t>(h);
        auto pred_i = pred.get_access<read_t>(h);

        h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                            sycl::group<1> gr = item.get_group();
                            sycl::range<1> ra = gr.get_local_range();
                            size_t src = gr.get_group_linear_id();
                            size_t blockDim = ra[0];
                            size_t threadIdx = item.get_local_id();
                            auto srcStart = start_i[src];
                            // Not a new frontier vertex with a matchable src.
                            //if (!matchable_i[src] || dist_i[src] != depth_i[0]+1)
                            if (!matchable_i[src] || dist_i[src] != 0)
                                return; 

                                const auto r = requests_i[src];

                                //This vertex has made a valid request.
                                if (requests_i[r] == src)
                                {
                                    //Match the vertices if the request was mutual.
                                    // matched vertices point to each other.
                                    // If blue copy the bridge into my bridge array entry
                                    // Flip the bridge so the LSW is my start.
                                    if (match_i[src] == 0){
                                        auto edgePair = b_i[r];
                                        uint32_t leastSignificantWord = (uint32_t)edgePair;
                                        uint32_t mostSignificantWord = (edgePair >> 32);
                                        uint64_t edgePairFlipped = (uint64_t) leastSignificantWord << 32 | mostSignificantWord;
                                        b_i[src] = edgePairFlipped;
                                        //printf("Bridge %u s=%d -> %u s=%d; I am src %lu, my start is %d; Flipping bridge %u -> %u\n",leastSignificantWord,start_i[leastSignificantWord],mostSignificantWord,start_i[mostSignificantWord],src,start_i[src], mostSignificantWord,leastSignificantWord);
                                    }
                                    match_i[src] = 4 + r;
                                    matchable_i[src] = false;
                                }

        });
        };
        q.submit(cg7);


        // Command Group creation
        // sets vertices in this next frontier which can augment/blossom and thus terminate.
        auto cg8 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto depth_i = depth.get_access<read_t>(h);
        auto match_i = match.get_access<write_t>(h);
        auto matchable_i = matchable.get_access<write_t>(h);

        auto requests_i = requests.get_access<write_t>(h);

        //auto b_i = bridgeVertex.get_access<write_t>(h);
        auto start_i = start.get_access<read_t>(h);
        auto b_i = bridgeVertex.get_access<read_write_t>(h);

        auto dist_i = dist.get_access<read_t>(h);
        auto pred_i = pred.get_access<read_t>(h);
    sycl::stream out(1024, 256, h);

        h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                            sycl::group<1> gr = item.get_group();
                            sycl::range<1> ra = gr.get_local_range();
                            size_t src = gr.get_group_linear_id();
                            size_t blockDim = ra[0];
                            size_t threadIdx = item.get_local_id();
                            auto srcStart = start_i[src];
                            // Not a new frontier vertex with a matchable src.
                            // The only way this is a matched src vertex that is matchable
                            // are newly matched vertices, thus we need to augment them.
                            if (matchable_i[src] || dist_i[src] != 0 || match_i[src]<4)
                                return; 

                            auto edgePair = b_i[src];
                            uint32_t leastSignificantWord = (uint32_t)edgePair;
                            uint32_t mostSignificantWord = (edgePair >> 32);

                            //if (start_i[leastSignificantWord] != src)
                                //printf("CATASTROPHIC ERROR! %u != %d\n", start_i[leastSignificantWord], src);

                            uint32_t curr_u = leastSignificantWord;
                            uint32_t curr_v = mostSignificantWord;
                            auto depth_u = dist_i[curr_u];

                            // Even bridge, match the bridge vertices.
                            if (depth_u % 2 == 0 && curr_u < curr_v){
                                bool has = false;
                                for (auto col_index = rows_i[curr_u]; col_index < rows_i[curr_u+1]; ++col_index){
                                    auto col = cols_i[col_index];
                                    has |= curr_v == col;                           
                                }
                                if (!has)
                                    out << "CATASTROPHIC ERROR! BRIDGE EDGE  " << curr_u << "-" << curr_v << " DNE" <<  cl::sycl::endl;
                                    //printf("CATASTROPHIC ERROR! BRIDGE EDGE %u - %u DNE\n", curr_u, curr_v);

                                match_i[curr_u] = 4+curr_v;
                                match_i[curr_v] = 4+curr_u;
                            }

                            for (auto curr_depth = depth_u; curr_depth > 0; --curr_depth){
                                auto parent_u = pred_i[curr_u];
                                bool has = false;
                                for (auto col_index = rows_i[curr_u]; col_index < rows_i[curr_u+1]; ++col_index){
                                    auto col = cols_i[col_index];
                                    has |= parent_u == col;                           
                                }
                                if (!has)
                                    out << "CATASTROPHIC ERROR! PRED EDGE  " << curr_u << "-" << curr_v << " DNE" <<  cl::sycl::endl;
                                //    printf("CATASTROPHIC ERROR! PRED EDGE %u - %u DNE\n", curr_u, curr_v);
                                if (curr_depth % 2 == 1){
                                    match_i[curr_u] = 4 + parent_u;
                                    match_i[parent_u] = 4 + curr_u;
                                }
                                curr_u = parent_u;
                            }

        });
        };
        q.submit(cg8);


        {
        const auto read_t = sycl::access::mode::read;
        auto km = keepMatching.get_access<read_t>();
        flag = km[0];
        }

        //#ifdef NDEBUG
        {
        const auto read_t = sycl::access::mode::read;
        const auto read_write_t = sycl::access::mode::read_write;
        auto depth_i = depth.get_access<read_t>();

        auto m = match.get_access<read_t>();
        auto cs = colsum.get_access<read_write_t>();
        cs[0] = 0;
        cs[1] = 0;
        cs[2] = 0;

        for (int i = 0; i < vertexNum; i++) {
            if(m[i] < 4)
            ++cs[m[i]];
            //printf("%d %d\n",i,m[i]);
        }
        
        

        bool bad = false;
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
        
        std::cout << "red count : " << cs[0] << std::endl;
        std::cout << "blue count : " << cs[1] << std::endl;
        std::cout << "dead count : " << cs[2] << std::endl;
        std::cout << "matched count : " << (vertexNum-(cs[0]+cs[1]+cs[2])) << std::endl;
        printf("Iteration %d depth %d\n", iteration++, depth_i[0]+1);
        //fflush(stdout);
        }
        //#endif
        // just to keep from entering an inf loop till all matching logic is done.
        //flag = false;
    } while(flag);

    
    sycl::buffer<int> checkMatch{VertexSize};

    {
        const auto write_t = sycl::access::mode::write;

        auto cm_i = checkMatch.get_access<write_t>();

        for (int i = 0; i < vertexNum; i++) {
            cm_i[i] = 0;
        }
    }

    bool validMatch = true;
    {
        const auto read_t = sycl::access::mode::read;
        const auto read_write_t = sycl::access::mode::read_write;

        auto m = match.get_access<read_t>();
        auto cs = colsum.get_access<read_write_t>();
        auto cm_i = checkMatch.get_access<read_write_t>();
        cs[0] = 0;
        cs[1] = 0;
        cs[2] = 0;

        for (int i = 0; i < vertexNum; i++) {
            if(m[i] < 4)
                ++cs[m[i]];
            else if(m[i] >= 4)
                ++cm_i[i];
        }
        #ifdef NDEBUG
        std::cout << "red count : " << cs[0] << std::endl;
        std::cout << "blue count : " << cs[1] << std::endl;
        std::cout << "dead count : " << cs[2] << std::endl;
        std::cout << "matched count : " << (vertexNum-(cs[0]+cs[1]+cs[2]))/2 << std::endl;
        for (int i = 0; i < vertexNum; i++) {
            if(cm_i[i] > 1){
                validMatch = false;
                printf("Error %d is matched %d times\n", i, cm_i[i]);
            }
        }  
        #endif
        matchCount = (vertexNum-(cs[0]+cs[1]+cs[2]));

    }
    if(validMatch){
        printf("post augment match is valid\n");
    }
    return;
}


// Only augments trivial paths.  Unmatched odd level vertices
void augment_trivial_paths(sycl::queue &q, 
                int & matchCount,
                sycl::buffer<int> &pred,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                const int vertexNum){

    constexpr const size_t SingletonSz = 1;
    const sycl::range Singleton{SingletonSz};

    constexpr const size_t TripletonSz = 3;
    const sycl::range Tripleton{TripletonSz};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};

    const size_t threadsPerBlock = 32;
    const size_t totalThreads = numBlocks * threadsPerBlock;

    const sycl::range NumWorkItems{totalThreads};
    const sycl::range WorkGroupSize{threadsPerBlock};
  
    sycl::buffer<bool> keepMatching{Singleton};
    sycl::buffer<unsigned int> selectBarrier {Singleton};
    sycl::buffer<unsigned int> colsum {Tripleton};

    auto cg = [&](sycl::handler &h) {    
    sycl::stream out(1024, 256, h);

    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;
    auto match_i = match.get_access<read_write_t>(h);
    auto dep = depth.get_access<read_t>(h);
    auto dist_i = dist.get_access<read_t>(h);
    auto start_i = start.get_access<read_t>(h);
    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        if (dist_i[i] != dep[0] + 1 || match_i[i] >= 4)
            return;
        //printf("Trivial path candidate %lu %d\n!", i[0], start_i[i]);
        out << "Trivial path candidate   " << i[0] << "-" << start_i[i]  <<  cl::sycl::endl;

    });
    };
    q.submit(cg);

}

// Contracts blossoms defined as bridges between vertices that share a source.
// This method is very similar to augment_bridges.
// Differences are:
//
// Prevent blossoms
// if(start_i[src] != start_i[col]){
// is replaced by 
// Detect blossoms
// if(start_i[src] == start_i[col]){
// 
// The maximal matching is performed on the set of blossom bridge vertices.
// An additional kernel to identify the base     then update the base of all blossom vertices.
bool contract_blossoms_2(sycl::queue &q, 
                int & matchCount,
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<uint64_t> &bridgeVertex, 
                sycl::buffer<int> &pred,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                sycl::buffer<int> &requests,
                sycl::buffer<bool> &matchable,
                sycl::buffer<int> &base,
                sycl::buffer<int> &forward,
                sycl::buffer<int> &backward,
                const int vertexNum){


    constexpr const size_t SingletonSz = 1;
    const sycl::range Singleton{SingletonSz};

    constexpr const size_t TripletonSz = 3;
    const sycl::range Tripleton{TripletonSz};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};

    const size_t threadsPerBlock = 32;
    const size_t totalThreads = numBlocks * threadsPerBlock;

    const sycl::range NumWorkItems{totalThreads};
    const sycl::range WorkGroupSize{threadsPerBlock};
  
    sycl::buffer<bool> keepMatching{Singleton};
    sycl::buffer<unsigned int> selectBarrier {Singleton};
    sycl::buffer<unsigned int> colsum {Tripleton};

    // Initialize input data
    {
        const auto write_t = sycl::access::mode::write;
        auto km = keepMatching.get_access<write_t>();
        km[0] = false;
    }

    auto cg = [&](sycl::handler &h) {    
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    auto b_i = bridgeVertex.get_access<dwrite_t>(h);

    auto matchable_i = matchable.get_access<dwrite_t>(h);
    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        matchable_i[i] = false;
        b_i[i]=0;
    });
    };
    q.submit(cg);

    // Command Group creation
    // sets vertices in this next frontier which can augment/blossom and thus terminate.
    auto cg4 = [&](sycl::handler &h) { 
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto km = keepMatching.get_access<write_t>(h);

    auto rows_i = rows.get_access<read_t>(h);
    auto cols_i = cols.get_access<read_t>(h);
    auto depth_i = depth.get_access<read_t>(h);
    auto match_i = match.get_access<read_t>(h);
    auto matchable_i = matchable.get_access<write_t>(h);

    //auto b_i = bridgeVertex.get_access<write_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto b_i = bridgeVertex.get_access<write_t>(h);

    auto dist_i = dist.get_access<read_t>(h);
    auto pred_i = pred.get_access<read_t>(h);
    sycl::stream out(1024, 256, h);

    h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        
                        //printf("src %lu dist %d depth %d start %d\n", src, dist_i[src], depth_i[0], start_i[src]);

                        // Not a new frontier vertex with an unmatched src.
                        // It's very important that src is unmatched, since this is called after augment_paths.
                        if (dist_i[src] != (depth_i[0]+1)  || match_i[start_i[src]] >= 4) return;
                        //printf("possible to color the src in %lu\n", src);
                        // If you win the first race, you make sure you get written in your slot then return.
                        //sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                        //sycl::access::address_space::global_space> ref_b_src(b_i[start_i[src]]);

                        // A bridge is an unmatched edge between two even levels
                        if ((depth_i[0]+1) % 2 == 0){
                        for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                        //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){                            
                            auto col = cols_i[col_index];
                            // An edge to a vertex in my even level.
                            if (dist_i[col] == dist_i[src]){
                                // If you win the first race, you make sure you get written in your slot then return. 
                                if (match_i[start_i[col]] < 4){   
                                    // Detect blossoms
                                    if(start_i[src] == start_i[col]){
                                        matchable_i[start_i[src]] = true;
                                        matchable_i[src] = true;
                                        uint32_t leastSignificantWord = src;
                                        uint32_t mostSignificantWord = col;
                                        uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
                                        b_i[start_i[src]] = edgePair;
                                        out << "Blossom bridge pair   " << src << "-" << col  <<  cl::sycl::endl;
                                        km[0] = true;
                                        return;
                                    }
                                }
                            }
                        }
                        } else {
                        for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                        //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                            // A matched edge to a vertex in my odd level.
                            if (match_i[col] == match_i[src] &&
                                dist_i[col] == dist_i[src]){
                                // If you win the first race, you make sure you get written in your slot then return.      
                                // Detect blossoms
                                if(start_i[src] == start_i[col]){
                                    matchable_i[start_i[src]] = true;
                                    matchable_i[src] = true;
                                    uint32_t leastSignificantWord = src;
                                    uint32_t mostSignificantWord = col;
                                    uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
                                    b_i[start_i[src]] = edgePair;
                                    //printf("Blossom bridge pair %lu %u\n", src, col);
                                    out << "Blossom bridge pair   " << src << "-" << col  <<  cl::sycl::endl;
                                    km[0] = true;
                                    return;
                                }
                            }
                        }
                        }       
    });
    };
    q.submit(cg4);
    fflush(stdout);

    // Command Group creation
    // sets vertices in this next frontier which can augment/blossom and thus terminate.
    auto cg8 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto rows_i = rows.get_access<read_t>(h);
    auto cols_i = cols.get_access<read_t>(h);
    auto depth_i = depth.get_access<read_t>(h);

    auto match_i = match.get_access<write_t>(h);
    auto matchable_i = matchable.get_access<read_write_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto b_i = bridgeVertex.get_access<read_t>(h);
    auto dist_i = dist.get_access<read_t>(h);
    auto pred_i = pred.get_access<read_t>(h);

    auto base_i = base.get_access<write_t>(h);
    auto for_i = forward.get_access<write_t>(h);
    auto back_i = backward.get_access<write_t>(h);
        sycl::stream out(1024, 256, h);

    h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> ra = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = ra[0];
                        size_t threadIdx = item.get_local_id();
                        auto srcStart = start_i[src];
                        if (!matchable_i[src] || dist_i[src] == 0 || b_i[srcStart] == 0)
                            return; 

                        auto edgePair = b_i[srcStart];
                        uint32_t leastSignificantWord = (uint32_t)edgePair;
                        uint32_t mostSignificantWord = (edgePair >> 32);
                        //if (leastSignificantWord != src && mostSignificantWord != src)
                        if (leastSignificantWord != src)
                            return;

                        if (threadIdx == 0){    
                            out << "contracting Blossom bridge pair   " << leastSignificantWord << "-" << mostSignificantWord  <<  cl::sycl::endl;

                            uint32_t curr_u = leastSignificantWord;
                            uint32_t curr_v = mostSignificantWord;
                            auto parent_u = pred_i[curr_u];
                            auto parent_v = pred_i[curr_v];
                            out << "finding base   " << curr_u << "-" << parent_u  <<  cl::sycl::endl;
                            out << "finding base   " << curr_v << "-" << parent_v  <<  cl::sycl::endl;
                            for_i[curr_u]=curr_v;
                            back_i[curr_v]=curr_u;

                            while(curr_u != curr_v){
                                out << "finding base   " << curr_u << "-" << parent_u  <<  cl::sycl::endl;
                                out << "finding base   " << curr_v << "-" << parent_v  <<  cl::sycl::endl;

                                back_i[curr_u]=parent_u;
                                for_i[curr_v]=parent_v;
                                curr_u  = parent_u;
                                curr_v  = parent_v;
                                parent_u = pred_i[curr_u];
                                parent_v = pred_i[curr_v];

                                //printf("curr_u %u curr_v %u parent_u %u parent_v %u\n", curr_u,curr_v,parent_u,parent_v);
                            }

                            auto base = curr_u;
                            base_i[base] = base;

                            curr_u = leastSignificantWord;
                            curr_v = mostSignificantWord;
                            parent_u = pred_i[curr_u];
                            parent_v = pred_i[curr_v];
                            while(curr_u != curr_v){
                                out << "setting base   " << curr_u << "-" << parent_u  <<  cl::sycl::endl;
                                out << "setting base   " << curr_v << "-" << parent_v  <<  cl::sycl::endl;
                                
                                base_i[curr_u]=base;
                                base_i[curr_v]=base;
                                curr_u  = parent_u;
                                curr_v  = parent_v;
                                parent_u = pred_i[curr_u];
                                parent_v = pred_i[curr_v];
//printf("curr_u %u curr_v %u base %u base %u\n", curr_u,curr_v,base,base);
                            }
                        }

    });
    };
    fflush(stdout);
    q.submit(cg8);
        // Color vertices
        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cg9 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
    sycl::stream out(1024, 256, h);

            // dist
            auto base_i = base.get_access<read_t>(h);
            auto for_i = forward.get_access<read_t>(h);

            h.parallel_for(VertexSize,
                            [=](sycl::id<1> i) { 
                if (base_i[i] != i)
                    return;
                auto base_u = base_i[i];
                auto curr_ut = base_u;
                //printf("u %u base_u %d\n", i[0], base_u);
                out << "u " << i[0] << " base_u " << base_u  <<  cl::sycl::endl;
                while(for_i[curr_ut] != base_u){
                    //printf("%d -> %d\n", curr_ut, for_i[curr_ut]);
                        out << curr_ut << "->" << curr_ut  <<  cl::sycl::endl;

                    curr_ut = for_i[curr_ut];
                }

            });
    };
    q.submit(cg9);
    {
        const auto read_t = sycl::access::mode::read;
        auto km = keepMatching.get_access<read_t>();
        return km[0];
    }
}

bool contract_blossoms(sycl::queue &q, 
                int & matchCount,
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<uint64_t> &bridgeVertex, 
                sycl::buffer<int> &pred,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                sycl::buffer<int> &requests,
                sycl::buffer<bool> &matchable,
                sycl::buffer<int> &base,
                sycl::buffer<int> &forward,
                sycl::buffer<int> &backward,
                const int vertexNum){


    constexpr const size_t SingletonSz = 1;
    const sycl::range Singleton{SingletonSz};

    constexpr const size_t TripletonSz = 3;
    const sycl::range Tripleton{TripletonSz};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};

    const size_t threadsPerBlock = 32;
    const size_t totalThreads = numBlocks * threadsPerBlock;

    const sycl::range NumWorkItems{totalThreads};
    const sycl::range WorkGroupSize{threadsPerBlock};
  
    sycl::buffer<bool> keepMatching{Singleton};
    sycl::buffer<unsigned int> selectBarrier {Singleton};
    sycl::buffer<unsigned int> colsum {Tripleton};

    // Initialize input data
    {
        const auto write_t = sycl::access::mode::write;
        auto km = keepMatching.get_access<write_t>();
        km[0] = false;
    }

    auto cg = [&](sycl::handler &h) {    
    const auto dwrite_t = sycl::access::mode::discard_write;
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    auto b_i = bridgeVertex.get_access<dwrite_t>(h);

    auto matchable_i = matchable.get_access<dwrite_t>(h);
    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        matchable_i[i] = false;
        b_i[i]=0;
    });
    };
    q.submit(cg);

    // Command Group creation
    // sets vertices in this next frontier which can augment/blossom and thus terminate.
    auto cg4 = [&](sycl::handler &h) { 
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto km = keepMatching.get_access<write_t>(h);

    auto rows_i = rows.get_access<read_t>(h);
    auto cols_i = cols.get_access<read_t>(h);
    auto depth_i = depth.get_access<read_t>(h);
    auto match_i = match.get_access<read_t>(h);
    auto matchable_i = matchable.get_access<write_t>(h);

    //auto b_i = bridgeVertex.get_access<write_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto b_i = bridgeVertex.get_access<write_t>(h);

    auto dist_i = dist.get_access<read_t>(h);
    auto pred_i = pred.get_access<read_t>(h);
    sycl::stream out(1024, 256, h);

    h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
                        sycl::group<1> gr = item.get_group();
                        sycl::range<1> r = gr.get_local_range();
                        size_t src = gr.get_group_linear_id();
                        size_t blockDim = r[0];
                        size_t threadIdx = item.get_local_id();
                        
                        //printf("src %lu dist %d depth %d start %d\n", src, dist_i[src], depth_i[0], start_i[src]);

                        // Not a new frontier vertex with an unmatched src.
                        // It's very important that src is unmatched, since this is called after augment_paths.
                        if (dist_i[src] != (depth_i[0]+1)  || match_i[start_i[src]] >= 4) return;
                        //printf("possible to color the src in %lu\n", src);
                        // If you win the first race, you make sure you get written in your slot then return.
                        //sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                        //sycl::access::address_space::global_space> ref_b_src(b_i[start_i[src]]);

                        // A bridge is an unmatched edge between two even levels
                        if ((depth_i[0]+1) % 2 == 0){
                        for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                        //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){                            
                            auto col = cols_i[col_index];
                            // An edge to a vertex in my even level.
                            if (dist_i[col] == dist_i[src]){
                                // If you win the first race, you make sure you get written in your slot then return. 
                                if (match_i[start_i[col]] < 4){   
                                    // Detect blossoms
                                    if(start_i[src] == start_i[col]){
                                        matchable_i[start_i[src]] = true;
                                        matchable_i[src] = true;
                                        matchable_i[col] = true;
                                        uint32_t leastSignificantWord = src;
                                        uint32_t mostSignificantWord = col;
                                        uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
                                        b_i[start_i[src]] = edgePair;
                                        out << "Blossom bridge pair   " << src << "-" << col  <<  cl::sycl::endl;
                                        km[0] = true;
                                        return;
                                    }
                                }
                            }
                        }
                        } else {
                        for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                        //for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){
                            auto col = cols_i[col_index];
                            // A matched edge to a vertex in my odd level.
                            if (match_i[col] == match_i[src] &&
                                dist_i[col] == dist_i[src]){
                                // If you win the first race, you make sure you get written in your slot then return.      
                                // Detect blossoms
                                if(start_i[src] == start_i[col]){
                                    matchable_i[start_i[src]] = true;
                                    matchable_i[src] = true;
                                    matchable_i[col] = true;
                                    uint32_t leastSignificantWord = src;
                                    uint32_t mostSignificantWord = col;
                                    uint64_t edgePair = (uint64_t) mostSignificantWord << 32 | leastSignificantWord;
                                    b_i[start_i[src]] = edgePair;
                                    //printf("Blossom bridge pair %lu %u\n", src, col);
                                    out << "Blossom bridge pair   " << src << "-" << col  <<  cl::sycl::endl;
                                    km[0] = true;
                                    return;
                                }
                            }
                        }
                        }       
    });
    };
    q.submit(cg4);
    fflush(stdout);


    auto cg5 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto km = keepMatching.get_access<write_t>(h);

    auto rows_i = rows.get_access<read_t>(h);
    auto cols_i = cols.get_access<read_t>(h);
    auto depth_i = depth.get_access<read_t>(h);
    auto match_i = match.get_access<read_t>(h);
    auto matchable_i = matchable.get_access<write_t>(h);

    //auto b_i = bridgeVertex.get_access<write_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto b_i = bridgeVertex.get_access<write_t>(h);

    auto dist_i = dist.get_access<read_t>(h);
    auto pred_i = pred.get_access<read_t>(h);

    auto base_i = base.get_access<write_t>(h);
    auto for_i = forward.get_access<write_t>(h);
    auto back_i = backward.get_access<write_t>(h);
    sycl::stream out(1024, 256, h);

    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) { 
        if (!matchable_i[i] || dist_i[i] != 0 || b_i[i] == 0)
            return; 

        auto edgePair = b_i[i];
        uint32_t leastSignificantWord = (uint32_t)edgePair;
        uint32_t mostSignificantWord = (edgePair >> 32);
        out << "Contracting Blossom bridge pair   " << leastSignificantWord << "-" << mostSignificantWord  <<  cl::sycl::endl;

        uint32_t curr_u = leastSignificantWord;
        uint32_t curr_v = mostSignificantWord;
        auto parent_u = pred_i[curr_u];
        auto parent_v = pred_i[curr_v];
        out << "finding base   " << curr_u << "-" << parent_u  <<  cl::sycl::endl;
        out << "finding base   " << curr_v << "-" << parent_v  <<  cl::sycl::endl;
        for_i[curr_u]=curr_v;
        back_i[curr_v]=curr_u;

        while(curr_u != curr_v){
            out << "finding base   " << curr_u << "-" << parent_u  <<  cl::sycl::endl;
            out << "finding base   " << curr_v << "-" << parent_v  <<  cl::sycl::endl;

            back_i[curr_u]=parent_u;
            for_i[curr_v]=parent_v;
            curr_u  = parent_u;
            curr_v  = parent_v;
            parent_u = pred_i[curr_u];
            parent_v = pred_i[curr_v];

            //printf("curr_u %u curr_v %u parent_u %u parent_v %u\n", curr_u,curr_v,parent_u,parent_v);
        }

        auto base = curr_u;
        base_i[base] = base;

        curr_u = leastSignificantWord;
        curr_v = mostSignificantWord;
        parent_u = pred_i[curr_u];
        parent_v = pred_i[curr_v];
        while(curr_u != curr_v){
            out << "setting base   " << curr_u << "-" << parent_u  <<  cl::sycl::endl;
            out << "setting base   " << curr_v << "-" << parent_v  <<  cl::sycl::endl;
            
            base_i[curr_u]=base;
            base_i[curr_v]=base;
            curr_u  = parent_u;
            curr_v  = parent_v;
            parent_u = pred_i[curr_u];
            parent_v = pred_i[curr_v];
//printf("curr_u %u curr_v %u base %u base %u\n", curr_u,curr_v,base,base);
        }
                        
        //if (leastSignificantWord != src && mostSignificantWord != src)

    });
    };

    q.submit(cg5);
    fflush(stdout);
        // Color vertices
        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cg9 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            sycl::stream out(1024, 256, h);

            // dist
            auto base_i = base.get_access<read_t>(h);
            auto for_i = forward.get_access<read_t>(h);

            h.parallel_for(VertexSize,
                            [=](sycl::id<1> i) { 
                if (base_i[i] != i)
                    return;
                auto base_u = base_i[i];
                auto curr_ut = base_u;
                //printf("u %u base_u %d\n", i[0], base_u);
                out << "u " << i[0] << " base_u " << base_u  <<  cl::sycl::endl;
                while(for_i[curr_ut] != base_u && for_i[curr_ut] != -1){
                    //printf("%d -> %d\n", curr_ut, for_i[curr_ut]);
                        out << curr_ut << "->" << curr_ut  <<  cl::sycl::endl;

                    curr_ut = for_i[curr_ut];
                }

            });
    };
    q.submit(cg9);
    fflush(stdout);
    {
        const auto read_t = sycl::access::mode::read;
        auto km = keepMatching.get_access<read_t>();
        return km[0];
    }
}


#endif