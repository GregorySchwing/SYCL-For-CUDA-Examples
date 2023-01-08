// Need to detect 3 cases (2 augments and 1 non-augmenting)
// Case 1 : trivial augmenting path (even depth with no neighbors in my depth)
// Case 2 : non-trivial augmenting path (even depth aux matched to a neighbor in my depth)
// Case 3 : blossom 

// Frontier level synchronization w pred
void augment_a(sycl::queue &q, 
                sycl::buffer<unsigned int> &rows, 
                sycl::buffer<unsigned int> &cols, 
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                sycl::buffer<int> &auxMatch,
                const int vertexNum){

    constexpr const size_t SingletonSz = 1;

    const sycl::range Singleton{SingletonSz};

    // Expanded
    sycl::buffer<bool> expanded{Singleton};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};

    const auto read_t = sycl::access::mode::read;
    auto d = depth.get_access<read_t>();
    for (int level = d[0]; level >= 0; --level){

        // Necessary to avoid atomics in setting pred/dist/start
        // Conflicts could come from setting pred and start non-atomically.
        // (Push phase) As dist is race-proof, only set pred in the frontier expansion 
        // (Pull phase) pull start into new frontier.
        // Command Group creation
        auto cg3 = [&](sycl::handler &h) {    
            const auto read_t = sycl::access::mode::read;
            const auto write_t = sycl::access::mode::write;
            const auto read_write_t = sycl::access::mode::read_write;

            auto depth_i = depth.get_access<read_t>(h);
            auto dist_i = dist.get_access<read_t>(h);
            auto auxM_i = auxMatch.get_access<read_write_t>(h);

            auto start_i = start.get_access<write_t>(h);
            h.parallel_for(VertexSize,
                            [=](sycl::id<1> i) { 
                if(level == dist_i[i] && auxM_i[start_i[i]] == start_i[i]){ 
                    start_i[i] = start_i[i];
                }
            });
        };
        q.submit(cg3);
    } 

    {
        const auto read_t = sycl::access::mode::read;
        auto d = dist.get_access<read_t>();
        auto s = start.get_access<read_t>();
        auto dep = depth.get_access<read_t>();

        std::cout << "Distance from start is : " << std::endl;
        for (int depth_to_print = 0; depth_to_print <= dep[0]; depth_to_print++) {
        for (int i = 0; i < vertexNum; i++) {
            if (d[i] == depth_to_print) printf("vertex %d dist %d start %d\n",i, d[i], s[i]);
        }
        }
        std::cout << std::endl;
    }

    return;
}
