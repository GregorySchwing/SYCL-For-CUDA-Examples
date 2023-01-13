#ifndef AUX_MATCH_H
#define AUX_MATCH_H
// auxilliary match

// To ensure no single vertx in an even level is used to form 
// 2 or more augmenting paths in one iteration. 
// Trivial augmenting paths are odd depth vertices unmatched in both matches.
// Matches form either blossoms or augmenting paths (check start vertices)
void maximalMatching(sycl::queue &q, 
                sycl::buffer<uint32_t> &rows, 
                sycl::buffer<uint32_t> &cols, 
                sycl::buffer<int> &requests,
                sycl::buffer<int> &match,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &auxMatch,
                const size_t vertexNum,
                const unsigned int barrier = 0x88B81733){

    constexpr const size_t SingletonSz = 1;
    const sycl::range Singleton{SingletonSz};

    constexpr const size_t TripletonSz = 3;
    const sycl::range Tripleton{TripletonSz};

    const sycl::range VertexSize{vertexNum};

    // Determines distribution of red/blue
    sycl::buffer<unsigned int> selectBarrier {Singleton};
    sycl::buffer<bool> keepMatching{Singleton};
    sycl::buffer<unsigned int> colsum {Tripleton};

    // Initialize input data
    {
        const auto read_t = sycl::access::mode::read;
        const auto dwrite_t = sycl::access::mode::discard_write;
        //auto deg = degree.get_access<read_t>();
        auto m = auxMatch.get_access<dwrite_t>();
        auto r = requests.get_access<dwrite_t>();

        auto sb = selectBarrier.get_access<dwrite_t>();
        auto km = keepMatching.get_access<dwrite_t>();
        auto cs = colsum.get_access<dwrite_t>();

        auto depth_i = depth.get_access<read_t>();
        auto dist_i = dist.get_access<read_t>();
        printf("matching inside depths > 0\n");
        for (int i = 0; i < vertexNum; i++) {
            // Only auxMatch in even levels greater than 0.
            if (dist_i[i] > 0 && dist_i[i] % 2 == 0){
                m[i] = 0;
                r[i] = 0;
            } else {
                m[i] = 2;
                r[i] = vertexNum + 1;
            }
        }
        sb[0] = barrier;
        km[0] = true;
        cs[0] = 0;
        cs[1] = 0;
        cs[2] = 0;

        std::cout << "selectBarrier " << sb[0] << std::endl;
    }

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

            auto match_i = auxMatch.get_access<read_write_t>(h);
            auto km = keepMatching.get_access<write_t>(h);
            auto depth_i = depth.get_access<read_t>(h);
            auto dist_i = dist.get_access<read_t>(h);

            h.parallel_for(VertexSize,
                            [=](sycl::id<1> i) { 
                // Unnecessary
                // if (i >= vertexNum) return;

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
                //printf("vertex %u colored %d\n", i[0],match_i[i] );
            });
        };
        q.submit(cg);

        // Request vertices - one workitem per workgroup
        // Command Group creation
        auto cg2 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto dwrite_t = sycl::access::mode::discard_write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto match_i = auxMatch.get_access<read_t>(h);
        auto requests_i = requests.get_access<dwrite_t>(h);
        auto dist_i = dist.get_access<read_t>(h);


        h.parallel_for(VertexSize,
                        [=](sycl::id<1> src) {                         
                            //Look at all blue vertices and let them make requests.
                            if (match_i[src] == 0)
                            {
                                int dead = 1;
                                
                                for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                                    auto col = cols_i[col_index];

                                    const auto nm = match_i[col];

                                    //Do we have an unmatched neighbour in my level?
                                    //Use <2 in this if-conditon so odd levels can start dead.
                                    if (nm < 2 && dist_i[src]==dist_i[col])
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

                                requests_i[src] = vertexNum + dead;
                            }
                            else
                            {
                                //Clear request value.
                                requests_i[src] = vertexNum;
                            }                    
        });
        };
        q.submit(cg2);

        // Respond vertices - one workitem per workgroup
        // Command Group creation
        auto cg3 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto dwrite_t = sycl::access::mode::discard_write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto rows_i = rows.get_access<read_t>(h);
        auto cols_i = cols.get_access<read_t>(h);
        auto match_i = auxMatch.get_access<read_t>(h);
        auto requests_i = requests.get_access<read_write_t>(h);
        auto dist_i = dist.get_access<read_t>(h);


        h.parallel_for(VertexSize,
                        [=](sycl::id<1> src) {                         
                            //Look at all red vertices.
                            if (match_i[src] == 1)
                            {
                                //Select first available proposer.
                                for (auto col_index = rows_i[src]; col_index < rows_i[src+1]; ++col_index){
                                    auto col = cols_i[col_index];
                                    //Only respond to blue neighbours in my level.
                                    if (match_i[col] == 0  && dist_i[src]==dist_i[col])
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
        q.submit(cg3);

        // Match vertices - one workitem per workgroup
        // Command Group creation
        auto cg4 = [&](sycl::handler &h) {    
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto dwrite_t = sycl::access::mode::discard_write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto match_i = auxMatch.get_access<write_t>(h);
        auto requests_i = requests.get_access<read_t>(h);


        h.parallel_for(VertexSize,
                        [=](sycl::id<1> src) {                         

                            const auto r = requests_i[src];

                            //Only unmatched vertices make requests.
                            if (r == vertexNum + 1)
                            {
                                //This is vertex without any available neighbours, discard it.
                                match_i[src] = 2;
                            }
                            else if (r < vertexNum)
                            {
                                //This vertex has made a valid request.
                                if (requests_i[r] == src)
                                {
                                    //Match the vertices if the request was mutual.
                                    // cant get this compile
                                    //  match_i[src] = 4 + min(src, r);
                                    //if (src < r)
                                    //match_i[src] = 4 + src;
                                    //else 
                                    //match_i[src] = 4 + r;
                                    // This way the matched vertices point to each other.
                                    match_i[src] = 4 + r;

                                }
                            }            
        });
        };
        q.submit(cg4);  

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

        for (int i = 0; i < vertexNum; i++) {
            if(m[i] < 4)
            ++cs[m[i]];
            //printf("%d %d\n",i,m[i]);
        }
        std::cout << "red count : " << cs[0] << std::endl;
        std::cout << "blue count : " << cs[1] << std::endl;
        std::cout << "dead count : " << cs[2] << std::endl;
        std::cout << "aux matched count : " << vertexNum-(cs[0]+cs[1]+cs[2]) << std::endl;
        }
        #endif
        // just to keep from entering an inf loop till all matching logic is done.
        //flag = false;

    } while(flag);

    // Match odd level vertices - one workitem per workgroup
    // Command Group creation
    auto cg4 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto match_i = match.get_access<read_t>(h);
    auto auxMatch_i = auxMatch.get_access<write_t>(h);
    auto dist_i = dist.get_access<read_t>(h);


    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) {  
                            // An odd-level vertex matched to another odd-level vertex.                     
                            if (dist_i[i] % 2 == 1 && 
                                match_i[i] >= 4 && 
                                dist_i[match_i[i]]  % 2 == 1)
                            {
                                //Match the vertices if the request was mutual.
                                // cant get this compile
                                //  match_i[src] = 4 + min(src, r);
                                auxMatch_i[i] = match_i[i];
                            }
    });
    };
    q.submit(cg4);  


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

        auto m = auxMatch.get_access<read_t>();
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

    }
    if(validMatch){
        printf("Match 2 is valid\n");
    }

    return;
}
#endif