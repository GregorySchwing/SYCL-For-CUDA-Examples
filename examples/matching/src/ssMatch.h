int edmonds(sycl::queue &myQueue, 
            sycl::buffer<uint32_t> &rows, 
            sycl::buffer<uint32_t> &cols, 
            const size_t vertexNum){


    const sycl::range VertexSize{vertexNum};

    sycl::buffer<int> match{VertexSize};
    sycl::buffer<int> q{VertexSize};
    sycl::buffer<int> father{VertexSize};
    sycl::buffer<int> base{VertexSize};

    sycl::buffer<bool> inq{VertexSize};
    sycl::buffer<bool> inb{VertexSize};
    sycl::buffer<bool> inp{VertexSize};

}