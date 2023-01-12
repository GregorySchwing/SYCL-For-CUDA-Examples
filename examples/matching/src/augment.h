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


#ifndef AUGMENT_h
#define AUGMENT_h

void augment_a(sycl::queue &q, 
                int & matchCount,
                sycl::buffer<unsigned int> &rows, 
                sycl::buffer<unsigned int> &cols, 
                sycl::buffer<int> &pred,
                sycl::buffer<int> &dist,
                sycl::buffer<int> &start,
                sycl::buffer<int> &depth,
                sycl::buffer<int> &match,
                sycl::buffer<int> &auxMatch,
                sycl::buffer<int> &winningAugmentingPath,
                const int vertexNum){

    constexpr const size_t SingletonSz = 1;

    const sycl::range Singleton{SingletonSz};

    // Expanded
    sycl::buffer<bool> expanded{Singleton};

    const size_t numBlocks = vertexNum;
    const sycl::range VertexSize{numBlocks};


    // Initialize input data
    {
        const auto read_t = sycl::access::mode::read;
        const auto dwrite_t = sycl::access::mode::discard_write;
        //auto deg = degree.get_access<read_t>();
        auto wAP_i = winningAugmentingPath.get_access<dwrite_t>();

        for (int i = 0; i < vertexNum; i++) {
            wAP_i[i] = -1;
        }
    }

    // Can't figure out how to do parallel.  I could use a mutex.
    {
        const auto read_t = sycl::access::mode::read;
        const auto write_t = sycl::access::mode::write;
        const auto read_write_t = sycl::access::mode::read_write;

        auto match_i = match.get_access<read_t>();
        auto auxMatch_i = auxMatch.get_access<read_t>();
        auto dist_i = dist.get_access<read_t>();
        auto start_i = start.get_access<read_t>();
        auto wAP_i = winningAugmentingPath.get_access<write_t>();
        for (int i = 0; i < vertexNum; i++) {
            if (wAP_i[start_i[i]] != -1)
                continue;

            // Case 1 : trivial augmenting path (end of tree (unmatched) 
            // odd depth vertex.
            if (dist_i[i] % 2 == 1 &&
                match_i[i] < 4 &&
                wAP_i[start_i[i]] == -1)
            {
                wAP_i[start_i[i]] = i;
            // Odd level aug-path
            // (start_i[i] != start_i[auxMatch_i[i]])
            // prevents blossoms from claiming a stake
            } else if (dist_i[i] % 2 == 1 &&
                        match_i[i] >= 4 &&
                        auxMatch_i[i] >= 4 &&
                        start_i[i] != start_i[auxMatch_i[i]] &&
                        wAP_i[start_i[i]] == -1 &&
                        wAP_i[start_i[auxMatch_i[i]]] == -1){
                wAP_i[start_i[i]] = i;
                wAP_i[start_i[auxMatch_i[i]]] = auxMatch_i[i];
            // Even level aug-path
            // (start_i[i] != start_i[auxMatch_i[i]])
            // prevents blossoms from claiming a stake
            // i < match_i[i]  ensures only 1 vertex from
            // the match tries to claim the SV.
            } else if (dist_i[i] % 2 == 0 &&
                        auxMatch_i[i] >= 4 &&
                        start_i[i] != start_i[auxMatch_i[i]] &&
                        wAP_i[start_i[i]] == -1 &&
                        wAP_i[start_i[auxMatch_i[i]]] == -1){
                wAP_i[start_i[i]] = i;
                wAP_i[start_i[auxMatch_i[i]]] = auxMatch_i[i];
            }
        }
    }

    // The challenge is claiming up two starting vertices per
    // augmenting path and ensuring no two augmenting paths
    // claim common start vertices.

    // This is done in two phases (stake and capture)
    // Stake claims the starting vertex of the smaller
    // of the two matched vertices in the secondary match.

    // Capture claims the larger if the smaller's stake was successful.

    // Since only successful stakes capture the second vertex,
    // the final value written to capture is ensured to be a valid
    // augmenting path.

    // The paths are guarunteed disjoint by the definition of matching.

    // Stake claim / Push smaller of two matched vertices
    // Command Group creation

    /*
    auto cg4 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto match_i = match.get_access<read_t>(h);
    auto auxMatch_i = auxMatch.get_access<read_t>(h);
    auto dist_i = dist.get_access<read_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto wAP_i = winningAugmentingPath.get_access<write_t>(h);

    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) {  
                            // Case 1 : trivial augmenting path (end of tree (unmatched) 
                            // odd depth vertex.
                            if (dist_i[i] % 2 == 1 &&
                                match_i[i] < 4)
                            {
                                wAP_i[start_i[i]] = i;
                            // Odd level aug-path
                            // (start_i[i] != start_i[auxMatch_i[i]])
                            // prevents blossoms from claiming a stake
                            } else if (dist_i[i] % 2 == 1 &&
                                        match_i[i] >= 4 &&
                                        i < match_i[i] &&
                                        auxMatch_i[i] >= 4 &&
                                        start_i[i] != start_i[auxMatch_i[i]]){
                                wAP_i[start_i[i]] = i;
                            // Even level aug-path
                            // (start_i[i] != start_i[auxMatch_i[i]])
                            // prevents blossoms from claiming a stake
                            // i < match_i[i]  ensures only 1 vertex from
                            // the match tries to claim the SV.
                            } else if (dist_i[i] % 2 == 0 &&
                                        auxMatch_i[i] >= 4 &&
                                        i < match_i[i] &&
                                        start_i[i] != start_i[auxMatch_i[i]]){
                                wAP_i[start_i[i]] = i;
                            }
    });
    };
    q.submit(cg4);  


    // Capture
    // Command Group creation
    auto cg5 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto match_i = match.get_access<read_t>(h);
    auto auxMatch_i = auxMatch.get_access<read_t>(h);
    auto dist_i = dist.get_access<read_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto wAP_i = winningAugmentingPath.get_access<write_t>(h);

    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) {  
                            // I won this starting vertex.
                            if (wAP_i[start_i[i]] == i && 
                                auxMatch_i[i] >= 4 &&
                                auxMatch_i[i] != 4+i){
                                // I claim the other starting vertex.
                                wAP_i[start_i[auxMatch_i[i]-4]] = auxMatch_i[i]-4;
                            }
    });
    };
    q.submit(cg5); 

    // Stake claim / Push smaller of two matched vertices
    // Command Group creation
    auto cg5 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto match_i = match.get_access<read_t>(h);
    auto auxMatch_i = auxMatch.get_access<read_t>(h);
    auto dist_i = dist.get_access<read_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto wAP_i = winningAugmentingPath.get_access<read_write_t>(h);

    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) {  
                            if (dist_i[i] % 2 == 1 &&
                                        match_i[i] >= 4 &&
                                        i > match_i[i] &&
                                        auxMatch_i[i] >= 4 &&
                                        start_i[i] != start_i[auxMatch_i[i]] &&
                                        wAP_i[start_i[auxMatch_i[i]]] == i){
                                wAP_i[start_i[i]] = i;
                            // Even level aug-path
                            // (start_i[i] != start_i[auxMatch_i[i]])
                            // prevents blossoms from claiming a stake
                            // i < match_i[i]  ensures only 1 vertex from
                            // the match tries to claim the SV.
                            } else if (dist_i[i] % 2 == 0 &&
                                        auxMatch_i[i] >= 4 &&
                                        i > match_i[i] &&
                                        start_i[i] != start_i[auxMatch_i[i]] &&
                                        wAP_i[start_i[auxMatch_i[i]]] == i){
                                wAP_i[start_i[i]] = i;
                            }
    });
    };
    q.submit(cg5);  

    // Augment paths
    // Command Group creation
    auto cg6 = [&](sycl::handler &h) {    
    const auto read_t = sycl::access::mode::read;
    const auto write_t = sycl::access::mode::write;
    const auto read_write_t = sycl::access::mode::read_write;

    auto match_i = match.get_access<read_write_t>(h);
    auto auxMatch_i = auxMatch.get_access<read_t>(h);
    auto dist_i = dist.get_access<read_t>(h);
    auto start_i = start.get_access<read_t>(h);
    auto wAP_i = winningAugmentingPath.get_access<read_t>(h);

    h.parallel_for(VertexSize,
                    [=](sycl::id<1> i) {  
                            // Case 1 : trivial augmenting path (end of tree (unmatched) 
                            // odd depth vertex.
                            if (dist_i[i] % 2 == 1 &&
                                match_i[i] < 4 &&
                                wAP_i[start_i[i]] == i)
                            {
                               ;
                            // Case 2 : Non-trivial augmenting path
                            } else if (wAP_i[start_i[auxMatch_i[i]-4]] == auxMatch_i[i]-4 &&
                                        wAP_i[start_i[auxMatch_i[i]-4]] == auxMatch_i[i]-4){
                            }
    });
    };
    q.submit(cg6);  
    */

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

    sycl::buffer<int> checkMatch{VertexSize};

    {
        const auto write_t = sycl::access::mode::write;

        auto cm_i = checkMatch.get_access<write_t>();

        for (int i = 0; i < vertexNum; i++) {
            cm_i[i] = 0;
        }
    }
    constexpr const size_t TripletonSz = 3;
    const sycl::range Tripleton{TripletonSz};
    sycl::buffer<unsigned int> colsum {Tripleton};
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
        std::cout << "red count : " << cs[0] << std::endl;
        std::cout << "blue count : " << cs[1] << std::endl;
        std::cout << "dead count : " << cs[2] << std::endl;
        std::cout << "new matched count : " << vertexNum-(cs[0]+cs[1]+cs[2]) << std::endl;
        for (int i = 0; i < vertexNum; i++) {
            if(cm_i[i] > 1){
                validMatch = false;
                printf("Error %d is matched %d times\n", i, cm_i[i]);
            }
        }  
        matchCount = vertexNum-(cs[0]+cs[1]+cs[2]);
    }
    if(validMatch){
        printf("Match 3 is valid\n");
    }

    return;
}

#endif