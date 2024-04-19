#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

// dim3 grid_dim(B, nh);  // batch_size x num_heads
// dim3 block_dim(Bc);  // Bc threads per block
__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y;  // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* Qi = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    float* S = &sram[tile_size * 3];

    for (int j = 0; j < Tc; j++) {

        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x];
        }
        __syncthreads();  // such that the inner loop can use the correct Kj, Vj

        for (int i = 0; i < Tr; i++)  {

            // Load Qi to SRAM, l and m to registers
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x];
            }
            float row_m_prev = m[lm_offset + (Br * i) + tx];
            float row_l_prev = l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
                }
                sum *= softmax_scale;
                S[(Bc * tx) + y] = sum;

                if (sum > row_m)
                    row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m);
                row_l += S[(Bc * tx) + y];
            }

            // Compute new m and l
            float row_m_new = max(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0;  // Pij * Vj
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
                }
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1 / row_l_new) \
                    * ((row_l_prev * __expf(row_m_prev - row_m_new) * O[qkv_offset + (tile_size * i) + (tx * d) + x]) \
                    + (__expf(row_m - row_m_new) * pv));
            }
            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();  // otherwise, thread can use the wrong Kj, Vj in inner loop
    }
}

__global__
void myForward_kernel(const float* Q, const float* K, const float* V, const int nh, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float *m, float* O) {
    // 每个block负责Br个Q/O, Bc个K V
    int bidx = blockIdx.x; int bidy = blockIdx.y; // b,nh
    int tid = threadIdx.x;
    int qkv_offset = bidx * nh * N * d + bidy * N * d;
    //每个block循环Tr次和Tc次
    extern __shared__ float smem[]; 
    // Q: Br * d
    // K: Bc * d
    float* Qi = smem;  // Br * d
    float* Kj = &smem[Br * d];  // Bc * d
    float* Vj = &smem[Br * d + Bc * d]; // Bc * d
    float* S = &smem[Br * d + 2 * Bc * d];  // Br * Bc

    // 每个thread负责q的一行，K、V的一列
    // 总共有Br个thread
    for(int j = 0; j<Tc; ++j){
        // load Kj Vj
        for(int k = 0; k<d; ++k){
            Kj[tid * d + k] = K[qkv_offset + j * (Bc*d) + tid * d + k]; //tid: 0-Br
            Vj[tid * d + k] = V[qkv_offset + j * (Bc*d) + tid * d + k]; //tid: 0-Br
        }
        __syncthreads();
        // load Qi
        for(int i = 0; i< Tr; ++i){
            for(int k = 0; k<d; ++k){
                Qi[tid * d + k] = Q[qkv_offset + i * (Br*d) + tid * d + k]; //tid: 0-Br
            }
            // 每个线程负责计算O的一行
            // 每个线程要遍历Kj Vj，以及Q的一行
            // Kj Vj等价于一个tilling
            
            // 找到tilling中的Q@K_T的最大值m_local
            // 并计算tilling  exp(-(Q@K_T-m_local))   
            // 原公式是：exp(-(Q@K_T-m_new))  =====> exp(-(Q@K_T-m_local   + m_local - m_new))

            // 每一个线程的逻辑
            // 这一个tilling的m
            float m_local = -INFINITY;
            float m_pre = m[bidx * nh * N + bidy * N + i * (Br) + tid];
            float l_pre = l[bidx * nh * N + bidy * N + i * (Br) + tid];
            for(int bc = 0; bc<Bc; ++bc){
                float sum = 0.;
                //计算O一行中的一个元素
                for(int k = 0; k<d; ++k){
                    sum += Qi[tid*d + k] * Kj[bc*d + k]; // Kj: Bc, d 虽然是乘转置，但K_T访问顺序依旧是顺序的
                }
                sum *= softmax_scale;
                m_local = fmaxf(sum, m_local);
                S[tid*Bc + bc] = sum; // O: 现在是Q@K_T
            }

            //这一个tilling的l
            float l_local = 0;
            for(int bc = 0; bc<Bc; ++bc){
                S[tid*Bc + bc] = expf(S[tid*Bc + bc]-m_local);
                l_local += S[tid*Bc + bc]; // O: 现在是exp(xj-m_local)
            }

            float m_new = fmaxf(m_pre, m_local);
            float l_new = l_pre * (expf(m_pre - m_new)) + l_local * (expf(m_local-m_new));

            for(int k = 0; k<d; ++k){
                float logits_v = 0.;
                for(int bc = 0; bc<Bc; ++bc){
                    logits_v += S[tid * Bc + bc] * expf(m_local-m_new) * Vj[bc * d + k];
                }
                O[qkv_offset + i*(Br*d) + tid * d + k] = 1/l_new * (
                    l_pre * expf(m_pre-m_new) * O[qkv_offset + i*(Br*d) + tid * d + k] + 
                    + logits_v
                );
            }
            m[bidx * nh * N + bidy * N + i * (Br) + tid] = m_new;
            l[bidx * nh * N + bidy * N + i * (Br) + tid] = l_new;
        }
        __syncthreads();
    }

}

__global__
void myForward_kernel2(const float* Q, const float* K, const float* V, const int nh, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale,
                    float* l, float* O_gmem) {
    // 每个block负责Br个Q/O, Bc个K V
    int bidx = blockIdx.x; int bidy = blockIdx.y; // b,nh
    int tid = threadIdx.x;
    int qkv_offset = bidx * nh * N * d + bidy * N * d;
    //每个block循环Tr次和Tc次
    extern __shared__ float smem[]; 
    // Q: Br * d
    // K: Bc * d
    float* Qi = smem;  // Br * d
    float* Kj = &smem[Br * d];  // Bc * d
    float* Vj = &smem[Br * d + Bc * d]; // Bc * d
    float* S = &smem[Br * d + 2 * Bc * d];  // Br * Bc
    float* O =  &smem[Br * d + 2 * Bc * d + Br * Bc];  // Br, d
    // float* l =  &smem[Br * d + 2 * Bc * d + Br * Bc + Br * d];  // Br
    // float* m =  &smem[Br * d + 2 * Bc * d + Br * Bc + Br * d + Br];  // Br
    
    // 每个thread负责q的一行，K、V的一列
    // 总共有Br个thread
    for(int i = 0; i< Tr; ++i){
        for(int k = 0; k<d; ++k){
            Qi[tid * d + k] = Q[qkv_offset + i * (Br*d) + tid * d + k]; //tid: 0-Br
        }
        // 每个线程的逻辑
        // float m_pre = m[tid];
        // float l_pre = l[tid];

        float m_pre = -INFINITY;
        float l_pre = l[qkv_offset + tid];
        for(int j = 0; j<Tc; ++j){
            // load Kj Vj
            for(int k = 0; k<d; ++k){
                Kj[tid * d + k] = K[qkv_offset + j * (Bc*d) + tid * d + k]; //tid: 0-Br
                Vj[tid * d + k] = V[qkv_offset + j * (Bc*d) + tid * d + k]; //tid: 0-Br
            }
            __syncthreads();
            
            float m_local = -INFINITY;
            // 计算Q@K_T的一行  Bc和元素
            for(int bc = 0; bc<Bc; ++bc){
                float sum = 0.;
                //计算O一行中的一个元素
                for(int k = 0; k<d; ++k){
                    sum += Qi[tid*d + k] * Kj[bc*d + k]; // Kj: Bc, d 虽然是乘转置，但K_T访问顺序依旧是顺序的
                }
                sum *= softmax_scale;
                m_local = fmaxf(sum, m_local);
                S[tid*Bc + bc] = sum; // S: 现在是Q@K_T
            }

            //这一个tilling的l
            float l_local = 0;
            for(int bc = 0; bc<Bc; ++bc){
                S[tid*Bc + bc] = expf(S[tid*Bc + bc]-m_local);
                l_local += S[tid*Bc + bc]; // S: 现在是exp(xj-m_local)
            }

            float m_new = fmaxf(m_pre, m_local);
            float l_new = l_pre * (expf(m_pre - m_new)) + l_local * (expf(m_local-m_new));

            // o：Br, Bc
            for(int k = 0; k<d; ++k){
                float logits_v = 0.;
                for(int bc = 0; bc<Bc; ++bc){
                    logits_v += S[tid * Bc + bc] * expf(m_local-m_new) * Vj[bc * d + k];
                }
                
                O[tid*d+k] = 1/l_new * (
                    l_pre * expf(m_pre-m_new) * O[tid*d+k]
                    + logits_v
                );
            }
            // m[tid] = m_new;
            // l[tid] = l_new;
            m_pre = m_new;
            l_pre = l_new;
        }
        
        for(int k = 0; k<d; ++k){
            O_gmem[qkv_offset + i*(Bc*d) + tid*d + k] = O[tid*d+k];
        }
        __syncthreads();
        l[qkv_offset + tid] = l_pre;
    }
}



torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    // Q/K/V: b, h, L
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);
    // 
    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});
    auto m = torch::full({B, nh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (3 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh);  // batch_size x num_heads
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}

torch::Tensor myForward(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    // Q: b, n, N, d
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);
    
    const float softmax_scale = 1.0 / sqrt(d);
    int Br = 32, Bc = 32;
    int Tr = ceil(N/Br);
    int Tc = ceil(N/Bc);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});  // O(n)
    auto m = torch::full({B, nh, N}, -INFINITY); // O(n)
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    dim3 block(Br);
    dim3 grid(B, nh);
    float sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %f \\n", max_sram_size, sram_size);

    myForward_kernel<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        nh, N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}


torch::Tensor myForward2(torch::Tensor Q, torch::Tensor K, torch::Tensor V){
    // Q: b, n, N, d
    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);
    
    const float softmax_scale = 1.0 / sqrt(d);
    int Br = 32, Bc = 32;
    int Tr = ceil(N/Br);
    int Tc = ceil(N/Bc);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nh, N});  // O(n)
    torch::Device device(torch::kCUDA);
    l = l.to(device);

    dim3 block(Br);
    dim3 grid(B, nh);
    float sram_size = (Br * d + 2 * Bc * d + Br * Bc + Br * d) * sizeof(float);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %f \n", max_sram_size, sram_size);
    printf("%f\n", (sram_size / sizeof(float)));
    printf("%f\n", (sram_size / sizeof(float)) - (Br * d + 2 * Bc * d + Br * Bc));
    //                                            Br * d + 2 * Bc * d + Br * Bc
    myForward_kernel2<<<grid, block, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        nh, N, d, Tc, Tr, Bc, Br, softmax_scale,
        l.data_ptr<float>(),
        O.data_ptr<float>()
    );
    return O;
}