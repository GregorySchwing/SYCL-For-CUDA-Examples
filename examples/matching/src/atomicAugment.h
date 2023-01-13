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

// Therefore, there are two kernels, augment_a which will only augment paths
// Then if any start vertices remain, they must be blossom structures or unaugmentable.
// If any blossoms are contracted in augment_b, reconstruct BFS.

// do {
//      BFS
//      AuxMatch
//      augment_a
//      augment_b_returned_true = augment_b
// } while(augment_b_returned_true)


#ifndef ATOMIC_AUGMENT_h
#define ATOMIC_AUGMENT_h

#include <CL/sycl.hpp>
void atomicAugment_a(sycl::queue &q, 
                int & matchCount,
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<int> &pred,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                sycl::buffer<int> &requests,
                sycl::buffer<uint64_t> &bridgeVertex,
                const int vertexNum,
                const unsigned int barrier = 0x88B81733){

    constexpr const size_t SingletonSz = 1;
    constexpr const size_t TripletonSz = 1;

    const sycl::range Singleton{SingletonSz};
    const sycl::range Tripleton{TripletonSz};

    sycl::buffer<unsigned int> selectBarrier {Singleton};
    sycl::buffer<bool> keepMatching{Singleton};
    sycl::buffer<unsigned int> colsum {Tripleton};

    // Expanded
    sycl::buffer<bool> expanded{Singleton};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};
    sycl::buffer<bool> matchable{VertexSize};

    const size_t threadsPerBlock = 32;
    const size_t totalThreads = numBlocks * threadsPerBlock;

    const sycl::range NumWorkItems{totalThreads};
    const sycl::range WorkGroupSize{threadsPerBlock};
    sycl::nd_range<1> test{NumWorkItems, WorkGroupSize};


    // Initialize input data
    {
        const auto read_t = sycl::access::mode::read;
        const auto dwrite_t = sycl::access::mode::discard_write;

        auto sb = selectBarrier.get_access<dwrite_t>();
        auto km = keepMatching.get_access<dwrite_t>();
        auto cs = colsum.get_access<dwrite_t>();
        sb[0] = barrier;
        km[0] = true;
        cs[0] = 0;
        cs[1] = 0;
        cs[2] = 0;

        std::cout << "selectBarrier " << sb[0] << std::endl;
    }

    // Initialize input data
    // Command Group creation
    auto cg = [&](sycl::handler &h) {    
        const auto dwrite_t = sycl::access::mode::discard_write;

        auto b_i = bridgeVertex.get_access<dwrite_t>(h);
        auto requests_i = requests.get_access<dwrite_t>(h);
        auto matchable_i = matchable.get_access<dwrite_t>(h);

        h.parallel_for(VertexSize,
                        [=](sycl::id<1> i) { 
                            b_i[i] = 0; 
                            requests_i[i] = -1;
                            matchable_i[i] = false;});
    };
    q.submit(cg);


    // Write all the 32bit-32bit uint bridges as 64bit uints into a shared array
    // If more than 1 bridge share a common src, they overwrite each other without
    // locking.  This is an acceptable race, since we can only use one anyway.
    // Last bridge to write to bridges[src] is a contender.
    // Then bridges[src] competes with bridges[col]
    // If bridges[src] == bridges[col], no competition.
    // If bridges[src] < bridges[col], bridges[src] = 0
    // If bridges[src] > bridges[col], bridges[col] = 0

    // The order of kernels should be 
    //  1)indicate those that are eligible srcs
    //  2)color these red/blue
    //  3)allow all bridges which are src'ed at a vertex to allow blue req[srcStart]s to atomically request red colStarts.




    bool flag = false;
    int iter = 0;
    do{
        {
        const auto write_t = sycl::access::mode::write;
        auto km = keepMatching.get_access<write_t>();
        km[0] = false;
        }

        //Initialize matchable srcs bool array.
        auto cg2 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto write_t = sycl::access::mode::write;
            const auto read_write_t = sycl::access::mode::read_write;

            auto rows_i = rows.get_access<read_t>(h);
            auto cols_i = cols.get_access<read_t>(h);
            auto match_i = match.get_access<read_t>(h);
            auto matchable_i = matchable.get_access<write_t>(h);
            auto dist_i = dist.get_access<read_t>(h);
            auto start_i = start.get_access<read_t>(h);

            h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
            //h.parallel_for(VertexSize,[=](sycl::id<1> src) {                         
                                sycl::group<1> gr = item.get_group();
                                sycl::range<1> ra = gr.get_local_range();
                                sycl::range<1> numV = gr.get_group_range();
                                size_t src = gr.get_group_linear_id();
                                size_t blockDim = ra[0];
                                size_t threadIdx = item.get_local_id();
                                auto srcStart = start_i[src];
                                // This is how I loop.
                                // In a bridge successfully matched then the srcStart will be matched.
                                if (match_i[srcStart] >= 4)
                                    return;

                                auto srcLevel = dist_i[src];
                                auto srcMatch = match_i[src];

                                for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){

                                    auto col = cols_i[col_index];

                                    // Case 1 : trivial augmenting path (end of tree (unmatched) 
                                    // odd depth vertex.
                                    if (srcLevel % 2 == 1 &&
                                        srcMatch < 4)
                                    {   
                                        matchable_i[srcStart] = true;

                                    // Odd level aug-path
                                    // (start_i[i] != start_i[auxMatch_i[i]])
                                    // prevents blossoms from claiming a stake
                                    // last condition  match_i[start_i[col]] < 4 lets me loop the augment
                                    } else if (srcLevel % 2 == 1 &&
                                                srcMatch >= 4 &&
                                                dist_i[col] % 2 == 1 &&
                                                match_i[col] >= 4 &&
                                                srcStart != start_i[col] && 
                                                match_i[start_i[col]] < 4){
                                        matchable_i[srcStart] = true;

                                    // Even level aug-path
                                    // (start_i[i] != start_i[auxMatch_i[i]])
                                    // prevents blossoms from claiming a stake
                                    // i < match_i[i]  ensures only 1 vertex from
                                    // the match tries to claim the SV.
                                    } else if (srcLevel % 2 == 0 &&
                                            dist_i[col] % 2 == 0 &&
                                            srcMatch == match_i[col] &&
                                            srcStart != start_i[col] && 
                                            match_i[start_i[col]] < 4){
                                        matchable_i[srcStart] = true;
                                    }
                                }
            });
        };
        q.submit(cg2);

        // Color vertices
        // Request vertices - one workitem per workgroup
        // Command Group creation
        // Could be a problem from edgepairs which can't match not dying.
        auto cg3 = [&](sycl::handler &h) { 
            const auto read_t = sycl::access::mode::read;
            const auto read_write_t = sycl::access::mode::read_write;
            const auto dwrite_t = sycl::access::mode::discard_write;
            const auto write_t = sycl::access::mode::write;

            // dist
            auto sb = selectBarrier.get_access<read_t>(h);
            auto randNum = rand();
            auto aMD5K = MD5K.get_access<read_t>(h);
            auto aMD5R = MD5R.get_access<read_t>(h);

            auto b_i = bridgeVertex.get_access<read_t>(h);
            auto matchable_i = matchable.get_access<read_t>(h);

            auto match_i = match.get_access<read_write_t>(h);
            auto km = keepMatching.get_access<write_t>(h);

            h.parallel_for(VertexSize,
                            [=](sycl::id<1> i) { 
                // Unnecessary
                // if (i >= vertexNum) return;

                // Only match the srcs of bridges which haven't been matched.
                if (!matchable_i[i])
                    return;
                
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
        q.submit(cg3);

        // Request atomically
        auto cg4 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto write_t = sycl::access::mode::write;
            const auto read_write_t = sycl::access::mode::read_write;

            auto rows_i = rows.get_access<read_t>(h);
            auto cols_i = cols.get_access<read_t>(h);
            auto match_i = match.get_access<read_t>(h);
            auto b_i = bridgeVertex.get_access<read_write_t>(h);
            auto dist_i = dist.get_access<read_t>(h);
            auto start_i = start.get_access<read_t>(h);
            auto matchable_i = matchable.get_access<read_t>(h);
            auto requests_i = requests.get_access<write_t>(h);

            h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {
            //h.parallel_for(VertexSize,[=](sycl::id<1> src) {                         
                                sycl::group<1> gr = item.get_group();
                                sycl::range<1> ra = gr.get_local_range();
                                sycl::range<1> numV = gr.get_group_range();
                                size_t src = gr.get_group_linear_id();
                                size_t blockDim = ra[0];
                                size_t threadIdx = item.get_local_id();
                                auto srcStart = start_i[src];

                                // I am blue
                                if (match_i[srcStart] == 0){

                                    auto srcLevel = dist_i[src];
                                    auto srcMatch = match_i[src];

                                    // I know that someone src'ed at srcstart had an bridge.                                    
                                    
                                    for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){

                                        auto col = cols_i[col_index];

                                        // Odd level aug-path
                                        // (start_i[i] != start_i[auxMatch_i[i]])
                                        // prevents blossoms from claiming a stake
                                        // last condition  match_i[start_i[col]] < 4 lets me loop the augment
                                        if (srcLevel % 2 == 1 &&
                                            srcMatch >= 4 &&
                                            dist_i[col] % 2 == 1 &&
                                            match_i[col] >= 4 &&
                                            srcStart != start_i[col] && 
                                            match_i[start_i[col]] < 4){
                                                const auto colStart = start_i[col];
                                                const auto nm = match_i[colStart];
                                                //Is this neighbour red?
                                                if (nm == 1)
                                                {
                                                    // I request a blue vertex
                                                    requests_i[srcStart] = colStart;
                                                    return;
                                                }     

                                        // Even level aug-path
                                        // (start_i[i] != start_i[auxMatch_i[i]])
                                        // prevents blossoms from claiming a stake
                                        // i < match_i[i]  ensures only 1 vertex from
                                        // the match tries to claim the SV.
                                        } else if (srcLevel % 2 == 0 &&
                                            dist_i[col] % 2 == 0 &&
                                            srcMatch == match_i[col] &&
                                            srcStart != start_i[col] && 
                                            match_i[start_i[col]] < 4){
                                                const auto colStart = start_i[col];
                                                const auto nm = match_i[colStart];
                                                //Is this neighbour red?
                                                if (nm == 1)
                                                {
                                                    // I request a blue vertex
                                                    requests_i[srcStart] = colStart;
                                                    return;
                                                }     
                                        }
                                    }
                                }
            });
        };
        q.submit(cg4);


        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cg5 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto write_t = sycl::access::mode::write;
            const auto read_write_t = sycl::access::mode::read_write;

            auto rows_i = rows.get_access<read_t>(h);
            auto cols_i = cols.get_access<read_t>(h);
            auto match_i = match.get_access<read_t>(h);
            auto b_i = bridgeVertex.get_access<read_t>(h);
            auto start_i = start.get_access<read_t>(h);
            auto requests_i = requests.get_access<write_t>(h);
            auto dist_i = dist.get_access<read_t>(h);

            h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {                
                            //Look at all blue vertices and let them make requests.
                                //Look at all blue vertices and let them make requests.
                                sycl::group<1> gr = item.get_group();
                                sycl::range<1> ra = gr.get_local_range();
                                sycl::range<1> numV = gr.get_group_range();
                                size_t src = gr.get_group_linear_id();
                                size_t blockDim = ra[0];
                                size_t threadIdx = item.get_local_id();
                                auto srcStart = start_i[src];

                                // I am red
                                if (match_i[srcStart] == 1){

                                    auto srcLevel = dist_i[src];
                                    auto srcMatch = match_i[src];

                                    // I know that someone src'ed at srcstart had an bridge.                                    
                                    
                                    for (auto col_index = rows_i[src] + threadIdx; col_index < rows_i[src+1]; col_index+=blockDim){

                                        auto col = cols_i[col_index];
                                        const auto colStart = start_i[col];

                                        // Odd level aug-path
                                        // (start_i[i] != start_i[auxMatch_i[i]])
                                        // prevents blossoms from claiming a stake
                                        // last condition  match_i[start_i[col]] < 4 lets me loop the augment
                                        if (srcLevel % 2 == 1 &&
                                            srcMatch >= 4 &&
                                            dist_i[col] % 2 == 1 &&
                                            match_i[col] >= 4 &&
                                            srcStart != colStart && 
                                            match_i[colStart] < 4){
                                                const auto nm = match_i[colStart];
                                                if (nm == 0)
                                                {
                                                    //Avoid data thrashing be only looking at the request value of blue neighbours.
                                                    if (requests_i[colStart] == srcStart)
                                                    {
                                                        requests_i[srcStart] = colStart;
                                                        return;
                                                    }
                                                }    

                                        // Even level aug-path
                                        // (start_i[i] != start_i[auxMatch_i[i]])
                                        // prevents blossoms from claiming a stake
                                        // i < match_i[i]  ensures only 1 vertex from
                                        // the match tries to claim the SV.
                                        } else if (srcLevel % 2 == 0 &&
                                            dist_i[col] % 2 == 0 &&
                                            srcMatch == match_i[col] &&
                                            srcStart != colStart && 
                                            match_i[colStart] < 4){

                                                const auto nm = match_i[colStart];
                                                //Is this neighbour blue?
                                                if (nm == 0)
                                                {
                                                    //Avoid data thrashing be only looking at the request value of blue neighbours.
                                                    if (requests_i[colStart] == srcStart)
                                                    {
                                                        requests_i[srcStart] = colStart;
                                                        // pack bridges here?
                                                        return;
                                                    }
                                                }     
                                        }
                                    }
                                }
                
        });
        };
        q.submit(cg5);


        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cg6 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto write_t = sycl::access::mode::write;
            const auto read_write_t = sycl::access::mode::read_write;

            auto match_i = match.get_access<read_write_t>(h);
            auto b_i = bridgeVertex.get_access<read_t>(h);
            auto start_i = start.get_access<read_t>(h);
            auto requests_i = requests.get_access<read_t>(h);

            h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {                
                            //Look at all blue vertices and let them make requests.
                                //Look at all blue vertices and let them make requests.
                                sycl::group<1> gr = item.get_group();
                                sycl::range<1> ra = gr.get_local_range();
                                sycl::range<1> numV = gr.get_group_range();
                                size_t src = gr.get_group_linear_id();
                                size_t blockDim = ra[0];
                                size_t threadIdx = item.get_local_id();
                                auto srcStart = start_i[src];

                                // This is necessarily to prevent from trying to match non-bridge srcs or already matched srcs
                                if (match_i[srcStart]>=4)
                                    return;

                                const auto r = requests_i[srcStart];

                                //This vertex has made a valid request.
                                if (requests_i[r] == srcStart)
                                {
                                    //Match the vertices if the request was mutual.
                                    // matched vertices point to each other.
                                    match_i[srcStart] = 4 + r;
                                }
                                   
                
        });
        };
        q.submit(cg6);



        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cg7 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto write_t = sycl::access::mode::write;
            const auto read_write_t = sycl::access::mode::read_write;

            auto match_i = match.get_access<read_t>(h);
            auto b_i = bridgeVertex.get_access<read_write_t>(h);
            auto start_i = start.get_access<read_t>(h);
            auto requests_i = requests.get_access<read_t>(h);

            h.parallel_for(sycl::nd_range<1>{NumWorkItems, WorkGroupSize}, [=](sycl::nd_item<1> item) {                
                            //Look at all blue vertices and let them make requests.
                                //Look at all blue vertices and let them make requests.
                                sycl::group<1> gr = item.get_group();
                                sycl::range<1> ra = gr.get_local_range();
                                sycl::range<1> numV = gr.get_group_range();
                                size_t src = gr.get_group_linear_id();
                                size_t blockDim = ra[0];
                                size_t threadIdx = item.get_local_id();
                                auto edgePair = b_i[src];
                                // This is necessarily a smaller number
                                // Since either u or v in (u,v) can be present
                                // in more than one edgePair, we need atomics below.
                                // Minimizing the amount of serialization from atomics

                                uint32_t srcStart = start_i[(uint32_t)edgePair];
                                uint32_t colStart = start_i[(edgePair >> 32)];
                                // This is necessarily to prevent from trying to match non-bridge srcs or already matched srcs
                                if (edgePair == 0 || match_i[srcStart]>=4)
                                    return;

                                // Reset bridges, so bridges which were once eligible
                                // will be ineligible next round.
                                // i.e. x -> y <- z __ a
                                // x matched y
                                // y <- z is now inelibigle and now next round
                                // x -> y __ z -> a 
                                if (match_i[colStart]>=4)     
                                    b_i[src] = 0;
                
        });
        };
        q.submit(cg7);



        {
        const auto read_t = sycl::access::mode::read;
        auto km = keepMatching.get_access<read_t>();
        flag = km[0];
        }

        //#ifdef NDEBUG
        {
        const auto read_t = sycl::access::mode::read;
        const auto read_write_t = sycl::access::mode::read_write;

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
        std::cout << "red count : " << cs[0] << std::endl;
        std::cout << "blue count : " << cs[1] << std::endl;
        std::cout << "dead count : " << cs[2] << std::endl;
        std::cout << "matched count : " << vertexNum-(cs[0]+cs[1]+cs[2]) << std::endl;
        }
        //#endif
        // just to keep from entering an inf loop till all matching logic is done.
        //flag = false;
        printf("Augment iteration %d\n", iter++);
    } while(flag);

/*
                            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                            sycl::access::address_space::global_space> ref_b_u(b_i[start_i[bridge_u]]);
    
                            sycl::atomic_ref<uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
                            sycl::access::address_space::global_space> ref_b_v(b_i[start_i[bridge_v]]);

*/

    /*
    // Can't figure out how to do parallel.  I could use a mutex.
    {
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;
        auto pred_i = pred.get_access<read_t>();
        auto match_i = match.get_access<write_t>();
        auto auxMatch_i = auxMatch.get_access<read_t>();
        auto dist_i = dist.get_access<read_t>();
        auto start_i = start.get_access<read_t>();
        auto wAP_i = winningAugmentingPath.get_access<read_t>();
        for (int i = 0; i < vertexNum; i++) {

            // Case 1 : trivial augmenting path (end of tree (unmatched) 
            // odd depth vertex.
            if (dist_i[i] % 2 == 1 &&
                match_i[i] < 4 &&
                wAP_i[start_i[i]] == i)
            {
                auto current = i;
                auto parent = pred_i[current];
                for (int pathDepth = dist_i[i]; pathDepth > 0; --pathDepth){
                    if (pathDepth % 2 == 1){
                        match_i[current] = 4+parent;
                        match_i[parent] = 4+current;
                    }
                    current = parent;
                    parent = pred_i[current]; 
                }
            // Odd level aug-path
            // (start_i[i] != start_i[auxMatch_i[i]])
            // prevents blossoms from claiming a stake
            } else if (dist_i[i] % 2 == 1 &&
                        match_i[i] >= 4 &&
                        auxMatch_i[i] >= 4 &&
                        wAP_i[start_i[i]] == i &&
                        wAP_i[start_i[auxMatch_i[i]]] == auxMatch_i[i]){
                auto current = i;
                auto parent = pred_i[current];
                for (int pathDepth = dist_i[i]; pathDepth > 0; --pathDepth){
                    if (pathDepth % 2 == 1){
                        match_i[current] = 4+parent;
                        match_i[parent] = 4+current;
                    }
                    current = parent;
                    parent = pred_i[current]; 
                }
            // Even level aug-path
            // (start_i[i] != start_i[auxMatch_i[i]])
            // prevents blossoms from claiming a stake
            // i < match_i[i]  ensures only 1 vertex from
            // the match tries to claim the SV.
            } else if (dist_i[i] % 2 == 0 &&
                        auxMatch_i[i] >= 4 &&
                        wAP_i[start_i[i]] == i &&
                        wAP_i[start_i[auxMatch_i[i]]] == auxMatch_i[i]){
                match_i[i] = 4+auxMatch_i[i];
                auto current = i;
                auto parent = pred_i[current];
                for (int pathDepth = dist_i[i]; pathDepth > 0; --pathDepth){
                    if (pathDepth % 2 == 0){
                        match_i[current] = 4+parent;
                        match_i[parent] = 4+current;
                    }
                    current = parent;
                    parent = pred_i[current]; 
                }
            }
        }
    }
    */
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
        //#ifdef NDEBUG
        std::cout << "red count : " << cs[0] << std::endl;
        std::cout << "blue count : " << cs[1] << std::endl;
        std::cout << "dead count : " << cs[2] << std::endl;
        std::cout << "matched count : " << vertexNum-(cs[0]+cs[1]+cs[2]) << std::endl;
        for (int i = 0; i < vertexNum; i++) {
            if(cm_i[i] > 1){
                validMatch = false;
                printf("Error %d is matched %d times\n", i, cm_i[i]);
            }
        }  
        //#endif
        matchCount = vertexNum-(cs[0]+cs[1]+cs[2]);
    }
    if(validMatch){
        printf("Match 3 is valid\n");
    }

    return;
}

#endif