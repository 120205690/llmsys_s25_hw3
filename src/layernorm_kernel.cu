#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
hidden_size here equals atleast one-fourth of the actual hidden_size since 4 data elements are loaded at once
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size/(some constant)
loops (some constant) number of times to cover the entire hidden_size vector
@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute x and x^2 with reinterpret_cast by casting to float4 for speedup
  // 2. Compute reduce sum with blockReduce and add epsilon with LN_EPSILON
  // 3. Compute layernorm result with reinterpret_cast by casting to float4 for speedup
  
  // Step 1
  // if(blockIdx.x > hidden_size){
  //   return;
  // }
  float l_sum = 0;
  float l_sum_sqr = 0;
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    l_sum += val.x + val.y + val.z + val.w;
    l_sum_sqr += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }
  const int reductions_per_thread = 1;
  // Step 2
  
  blockReduce<ReduceType::kSum, reductions_per_thread>(&l_sum);
  blockReduce<ReduceType::kSum, reductions_per_thread>(&l_sum_sqr);
  __shared__ float mean, inv_std, var;
  
  // float var;
  if(threadIdx.x == 0){
    mean = l_sum/(hidden_size*4);
    var = l_sum_sqr/(hidden_size*4) - mean*mean;
    inv_std = rsqrt(var + LN_EPSILON);
  }
  // Step 3
  __syncthreads();
  //cast scale and bias for float4 coalesced reads
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  float4 *ln_out_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;

  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 scale_val = scale_f4[idx];
    float4 bias_val = bias_f4[idx];
    float4 ln_val;
    ln_val.x = scale_val.x * (val.x - mean)*inv_std + bias_val.x;
    ln_val.y = scale_val.y * (val.y - mean)*inv_std + bias_val.y;
    ln_val.z = scale_val.z * (val.z - mean)*inv_std + bias_val.z;
    ln_val.w = scale_val.w * (val.w - mean)*inv_std + bias_val.w;
    ln_out_f4[idx] = ln_val;
  }

  //write results out
  if(threadIdx.x == 0){
    vars[blockIdx.x] = var;
    if(means != nullptr){
      means[blockIdx.x] = mean;
    }
  }
  /// END ASSIGN3_2
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

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
  /*The row threads in a block have to span the entire set of rows or batches in the input matrix
   (=rows here). Therefore each row thread must cover rows/blockDim.y rows which it does by iterating
    over them. Simultaneously, the warp is parallel to the rows and allows coalesced memory access by 
    reading successive memory locations of the input array at the same time.
  Calculate the colsum for each row and reduce them for all rows in a block.
  Also the shared memory array stores the final results for the entire block since it is blockdim in size */
  
  /* things to keep in mind: boundary checks, casting to float for calculation since T type may not match
   shared memory, division precision by instead multiplying by the reciprocal and also a smoothening 
   factor of epsilon.
  Also, the threads which run outside the boundaries may also have to initialize their shared memory locs
   to zero and this has to be done synchronously*/


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
          //inp now represents the output since means is null
          x_hat = (inp_val - static_cast<float>(betta[global_col]))/static_cast<float>(gamma[global_col]);
        }
        else{
          //throw error
          assert(false && "Error: Entered forbidden condition case");
        }
        gamma_sum += static_cast<float>(out_grad[input_ptr]) * x_hat;
    }
  }
  // Step 2
  /*store beta_sum and gamma_sum in the shared arrays. Here's a cool trick. Reductions happen best across
   a warp which is currently row-wise. But the reduction is to be carried across the columns of the array.
   Therefore, give each horizontal warp the data for a particular vertical col. Carry out the reduction 
   such that now the first elem of each warp has all the data i.e. threadX=0. Copy this over into the 
   global array. But the threads now holding the data have been switched therefore, their global_col too is
   different (since col 1 holds all the data after the reduction). Therefore calc the new global_col and 
   store the data back into global_memory. 
   */
  betta_buffer[threadIdx.y][threadIdx.x] = beta_sum;
  gamma_buffer[threadIdx.y][threadIdx.x] = gamma_sum;
  b.sync();

  beta_sum = betta_buffer[threadIdx.x][threadIdx.y];
  gamma_sum = gamma_buffer[threadIdx.x][threadIdx.y];
  b.sync();

  // Step 3
  /* Perform a reduction_sum in each col to end up with a hidden_size array. This reduction function accesses values and variables in other threads (offset by the loop variable offset) and adds them together till only one thread contains the entire reduced value. This is something performed explicitly by just one thread hence thread control flow has been added. */
  // gamma_sum = 0.0f;
  // beta_sum = 0.0f;
  
  for(int offset = TILE_DIM/2; offset>0; offset/=2){
    beta_sum += g.shfl_down(beta_sum, offset);
    gamma_sum += g.shfl_down(gamma_sum, offset);
  } 
  b.sync();
  global_col = blockDim.x * blockIdx.x + threadIdx.y;

  if(threadIdx.x == 0 && global_col < width){
  // Step 4
  gamma_grad[global_col] = static_cast<T>(gamma_sum);
  betta_grad[global_col] = static_cast<T>(beta_sum);
  }
  // b.sync();
  /// END ASSIGN3_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size/(some constant)

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
beta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *beta, const T *vars,
                               const T *means, int hidden_size) {
  
  /// BEGIN ASSIGN3_2
  /// TODO
  // Hints:
  // 1. Compute dxhat=dy*gamma with reinterpret_cast by casting to float4 for speedup
  // 2. Compute xhat with reinterpret_cast by casting to float4 for speedup
  // 3. Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  // 4. Compute final gradient
  /* The gradient equation gives the gradient element i for batch b of the input matrix.
  j represents a sum over the entire hidden dim. Therefore, dot products between a batch of y and gamma
  and x_hat are taken and stored. This is essentially a reduction which is to be subtracted from the
  product of the i_th element of the gradient y (of a particular batch b) and gamma vector.
  Each batch of y of size hidden_size is also stored in consecutive memory locations so warping makes 
  sense. 
  As an aside, always store matrices in warp sized dimensions to make it explicitly easier to load as a 
  warp to perform warp reduce over all hidden_size elements. 
  If each row corresponds to a hidden_size vector, with block of dim 32 x 32,
  then each row can reduce an entire vector. But now blockreduce would have to performed for every row
  making the entire gpu wait. In beta_grad, reduction was performed over the batch_dim where an iteration
  was performed over an entire col. Similarly, if each col is now a hidden_dim, the exact same
  paradigm can be now applied wherein 32 rows iterate to cover the entire hidden_dim and only one reduction
  is carried out. The no. of cols equals the batch_size. Each block now operates independently.

  Easier still: just have each block do the hidden_size ops for each batch. Launch batch_size no. of blocks.
  Same as earlier, since same number of threads, but using 
  */
  // Step 1
  //Note hidden_size is now the actual hidden_size/4
  int tx = threadIdx.x; 
  int blockdim = blockDim.x;
  
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + blockIdx.x * hidden_size;  
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + blockIdx.x * hidden_size;  

  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);  
  const float4 *beta_f4 = reinterpret_cast<const float4 *>(beta);
  float vars_val = rsqrt(*(static_cast<const float *>(vars + blockIdx.x)) + LN_EPSILON);
  float mean_val = *(static_cast<const float *>(means + blockIdx.x));

  float y_gamma = 0.0f; float y_gamma_xhat = 0.0f;
  
  //Loop over row and calc the 2 dot product terms which need to be reduced
  for(int i = tx; i < hidden_size; i += blockdim){
    float4 inp_val  = inp_f4[i];
    float4 out_grad_val  = out_grad_f4[i];
    float4 gamma_val = gamma_f4[i];
    float4 beta_val = beta_f4[i];

    y_gamma += out_grad_val.x * gamma_val.x + out_grad_val.y * gamma_val.y + out_grad_val.z * gamma_val.z + out_grad_val.w * gamma_val.w;
    if(means && vars){
      y_gamma_xhat += out_grad_val.x * gamma_val.x * (inp_val.x - mean_val)*vars_val + out_grad_val.y * gamma_val.y * (inp_val.y - mean_val)*vars_val + out_grad_val.z * gamma_val.z * (inp_val.z - mean_val)*vars_val + out_grad_val.w * gamma_val.w * (inp_val.w - mean_val)*vars_val;
    }
    else{
      y_gamma_xhat += out_grad_val.x * gamma_val.x * (inp_val.x - beta_val.x)/gamma_val.x + out_grad_val.y * gamma_val.y * (inp_val.y - beta_val.y)/gamma_val.y + out_grad_val.z * gamma_val.z * (inp_val.z - beta_val.z)/gamma_val.z + out_grad_val.w * gamma_val.w * (inp_val.w - beta_val.w)/gamma_val.w;
    }
  }

  // Step 2
  const int reductions_per_thread = 1;
  blockReduce<ReduceType::kSum, reductions_per_thread>(&y_gamma);
  blockReduce<ReduceType::kSum, reductions_per_thread>(&y_gamma_xhat);
  
  // Step 3
  float4 x_hat_val;

  for(int i = tx; i < hidden_size; i += blockdim){
    float4 inp_val  = inp_f4[i];
    float4 out_grad_val  = out_grad_f4[i];
    float4 gamma_val = gamma_f4[i];
    float4 beta_val = beta_f4[i];

    if(vars && means){
      x_hat_val.x = (inp_val.x - mean_val) * vars_val;
      x_hat_val.y = (inp_val.y - mean_val) * vars_val;
      x_hat_val.z = (inp_val.z - mean_val) * vars_val;
      x_hat_val.w = (inp_val.w - mean_val) * vars_val;
    }
    else if(gamma && beta){
      //inp now represents the output since means is null
      x_hat_val.x = (inp_val.x - beta_val.x)/gamma_val.x;
      x_hat_val.y = (inp_val.y - beta_val.y)/gamma_val.y;
      x_hat_val.z = (inp_val.z - beta_val.z)/gamma_val.z;
      x_hat_val.w = (inp_val.w - beta_val.w)/gamma_val.w;
    }
    float4 term1, term3;
    term1.x = out_grad_val.x * gamma_val.x * vars_val;
    term1.y = out_grad_val.y * gamma_val.y * vars_val;
    term1.z = out_grad_val.z * gamma_val.z * vars_val;
    term1.w = out_grad_val.w * gamma_val.w * vars_val;


    float term2 = -vars_val/(hidden_size*4);

    term3.x = y_gamma + x_hat_val.x * y_gamma_xhat;
    term3.y = y_gamma + x_hat_val.y * y_gamma_xhat;
    term3.z = y_gamma + x_hat_val.z * y_gamma_xhat;
    term3.w = y_gamma + x_hat_val.w * y_gamma_xhat;


    float4 x_grad_val;
    x_grad_val.x = term1.x + term2 * term3.x;
    x_grad_val.y = term1.y + term2 * term3.y;
    x_grad_val.z = term1.z + term2 * term3.z;
    x_grad_val.w = term1.w + term2 * term3.w;

    // Step 4 
    inp_grad_f4[i] = x_grad_val;
  }
  /// END ASSIGN3_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
