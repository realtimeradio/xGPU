#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define NTIME_GATHER 4

__constant__ unsigned int lut[16] = {0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255};

CUBE_KERNEL(static swizzleKernel, unsigned int *in, unsigned int *out, int nfreq, int ninput)
{
  CUBE_START;
  // We define threads such that each thread deals with
  // 4 consecutive pols and 4 consecutive times
  // Call with ninput / 4 threads per block
  // Call with block dimensions (nfreq, NTIME/NTIME_GATHER)
  
  // grab 4 times, 4 pols:
  //int *in = in +
  //           threadIdx.x + // pol selection by thread. Each 32-bits is 4 pols
  //           ninput*blockIdx.x/4 + // channel selection
  //           ninput*nfreq*blockIdx.y; // time selection

  //char lut[16] = {0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1};
  //unsigned int lut[16] = {0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255};
  unsigned int *in_offset = (unsigned int *)in + threadIdx.x + ninput/4*blockIdx.x + ninput/4*nfreq*NTIME_GATHER*blockIdx.y;

  unsigned int t[4];
  t[0] = in_offset[0];                   // pols threadId -> threadId+3; time 0
  t[1] = in_offset[(nfreq*ninput)>>2];   //time 1
  t[2] = in_offset[(2*nfreq*ninput)>>2]; //time 2
  t[3] = in_offset[(3*nfreq*ninput)>>2]; //time 3

  // t is 4[times] x 4[pols] x 2[complexity] x 4 bits
  // output buffer is NTIME * nfreq * ninput * complexity x 4 x 1byte
  // char *out =  out +
  //              NTIME_GATHER * NTIME_GATHER * 2 * threadIdx.x + // Write 4 pols, 2 complexities, 4 times per thread
  //              NTIME_GATHER * ninput * 2 * blockIdx.x + // chans
  //              nfreq * ninput * NTIME_GATHER * 2 * blockIdx.y; // times

  //char *out_offset =  out + NTIME_GATHER*NTIME_GATHER*2*threadIdx.x + NTIME_GATHER*ninput*2*blockIdx.x + nfreq*ninput*NTIME_GATHER*2*blockIdx.y;

  //  #pragma unroll
  //  for (int i=0; i<4; i++) {
  //    out_offset[8*i+0]   = lut[(t[0] >> (8*i + 4)) & 0xf]; // imag, time0
  //    out_offset[8*i+1]   = lut[(t[1] >> (8*i + 4)) & 0xf]; // imag, time1
  //    out_offset[8*i+2]   = lut[(t[2] >> (8*i + 4)) & 0xf]; // imag, time2
  //    out_offset[8*i+3]   = lut[(t[3] >> (8*i + 4)) & 0xf]; // imag, time3
  //    out_offset[8*i+4]   = lut[(t[0] >> (8*i)) & 0xf]; // real, time0
  //    out_offset[8*i+5]   = lut[(t[1] >> (8*i)) & 0xf]; // real, time1
  //    out_offset[8*i+6]   = lut[(t[2] >> (8*i)) & 0xf]; // real, time2
  //    out_offset[8*i+7]   = lut[(t[3] >> (8*i)) & 0xf]; // real, time3
  //  }

  unsigned int *out_offset =  (unsigned int *)out + NTIME_GATHER*2*threadIdx.x + ninput*2*blockIdx.x + nfreq*ninput*2*blockIdx.y;

  #pragma unroll (4)
  for (int i=0; i<4; i++) {
    out_offset[2*i]     = ((lut[(t[0] >> (8*i + 4)) & 0xf])<<24) +
                          ((lut[(t[1] >> (8*i + 4)) & 0xf])<<16) +
                          ((lut[(t[2] >> (8*i + 4)) & 0xf])<<8) +
                          ((lut[(t[3] >> (8*i + 4)) & 0xf])<<0);
    out_offset[2*i+1]   = ((lut[(t[0] >> (8*i)) & 0xf])<<24) +
                          ((lut[(t[1] >> (8*i)) & 0xf])<<16) +
                          ((lut[(t[2] >> (8*i)) & 0xf])<<8) +
                          ((lut[(t[3] >> (8*i)) & 0xf])<<0);
  }
  CUBE_END;
}
