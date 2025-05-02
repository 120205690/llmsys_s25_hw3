/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute the partial gradients by looping across inp rows
  // 2. Store the partial gradients in the shared memory arrays
  // 3. Compute the reduce sum of the shared memory arrays with g.shfl_down
  //      -> More hints about `g.shfl_down`:
  //      -> https://developer.nvidia.com/blog/cooperative-groups/#:~:text=Using%20thread_block_tile%3A%3Ashfl_down()%20to%20simplify%20our%20warp%2Dlevel%20reduction%20does%20benefit%20our%20code%3A%20it%20simplifies%20it%20and%20eliminates%20the%20need%20for%20shared%20memory
  //      -> The highlighted line gives you a conceptual understanding of what the g.shfl_down is doing. Usually, the threads inside a block need to load everything to shared memory and work together to reduce the result (like what you have implemented in the hw1 for reduce function). 
  //      -> Now g.shfl_down helps you do so without consuming any shared memory. g.shfl_down makes it more efficient.
  // 4. Assign the final result to the correct position in the global output

  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);
  float beta_sum = 0.0f;
  float gamma_sum = 0.0f;
  
  int global_col = blockDim.x * blockIdx.x + threadIdx.x;
  
  // Step 1
  //betagrad is simply equal to the reduce_sum of the y_grad or out_grad
  /*The row threads in a block have to span the entire set of rows in the input matrix (=rows here). Therefore each row thread must cover rows/blockDim.y rows which it does by iterating over them. Simultaneously, the warp is parallel to the rows and allows coalesced memory access by reading successive memory locations of the input array at the same time.
  Calculate the colsum for each row and reduce them for all rows in a block.
  Also the shared memory array stores the final results for the entire block since it is blockdim in size */
  /* things to keep in mind: boundary checks, casting to float for calculation since T type may not match shared memory, division precision by instead multiplying by the reciprocal and also a smoothening factor of epsilon.
  Also, the threads which run outside the boundaries may also have to initialize their shared memory locs to zero and this has to be done synchronously*/


  if(global_col<width){
    for(int rowidx = threadIdx.y; rowidx<rows; rowidx+=blockDim.y){
      int input_ptr = rowidx*width + global_col;
      //boundary check
      
        //load input and mean and variance and any other variable that needs to be loaded using this
        beta_sum += static_cast<float>(out_grad[input_ptr]);
        //now x_hat must be calculated for gamma_sum
        //either var and means are given or beta and gamma. modify formula for calulating x_hat accordingly
        float inp_val = static_cast<float>(inp[input_ptr]);
        float x_hat = 0.0f;
        if(vars && means){
          x_hat = (inp_val - static_cast<float>(means[rowidx])) * rsqrt(static_cast<float>(vars[rowidx]) + LN_EPSILON);
        }
        else if(gamma && betta){
          //inp not represents the output since means is null
          x_hat = (inp_val - static_cast<float>(betta[global_col]))/static_cast<float>(gamma[global_col]);
        }
        else{
          //throw error
          assert(false && "Error: Entered forbidden condition case");
        }
        gamma_sum = out_grad[input_ptr] * x_hat;
    }
  }
  // Step 2
  //store beta_sum and gamma_sum in the shared arrays
  betta_buffer[threadIdx.y][threadIdx.x] = beta_sum;
  gamma_buffer[threadIdx.y][threadIdx.x] = gamma_sum;
  b.sync();
  // Step 3
  /* Perform a reduction_sum in each col to end up with a hidden_size array. This reduction function accesses values and variables in other threads (offset by the loop variable offset) and adds them together till only one thread contains the entire reduced value. This is something performed explicitly by just one thread hence thread control flow has been added. */
  gamma_sum = 0.0f;
  beta_sum = 0.0f;
  if(threadIdx.y ==0 && global_col<width){
    for(int offset = TILE_DIM/2; offset>0; offset/=2){
      gamma_sum += g.shfl_down(gamma_sum, offset);
      beta_sum += g.shfl_down(beta_sum, offset);

    } 
    // Step 4
    gamma_grad[global_col] = static_cast<T>(gamma_sum);
    betta_grad[global_col] = static_cast<T>(beta_sum);
  }
  
  b.sync();// ensures other threads weight while one thread performs the reduction


  // assert(false && "Not Implemented");
  /// END ASSIGN3_2
}