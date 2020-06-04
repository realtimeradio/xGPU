#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

#define NTIME_GATHER 4
#define POLBLOCKSIZE 4
#define CHANBLOCKSIZE 8

__constant__ unsigned int lut[16] = {0, 1, 2, 3, 4, 5, 6, 7, 248, 249, 250, 251, 252, 253, 254, 255};
//__constant__ unsigned int mask[16] = {0x0000000f, 0x000000f0, 0x00000f00, 0x0000f000,
//                                    0x000f0000, 0x00f00000, 0x0f000000, 0xf0000000};

CUBE_KERNEL(static swizzleKernel, unsigned int *in, unsigned int *out, int nfreq, int ninput)
{
  CUBE_START;
  // We define threads such that each thread deals with
  // 4*POLBLOCKSIZE consecutive pols and 4 consecutive times
  // Call with (ninput / (4*POLBLOCKSIZE) x  CHANBLOCKSIZE) threads per block
  // Call with block dimensions (nfreq/CHANBLOCKSIZE, NTIME/NTIME_GATHER)
  
  // grab 4 times, 4*POLBLOCKSIZE pols:
  //int *in = in +
  //           POLBLOCKSIZE*threadIdx.x + // pol selection by thread. Each 32-bits is 4 pols
  //           ninput*(threadIdx.y + CHANBLOCKSIZE*blockIdx.x)/4 + // channel selection
  //           ninput*nfreq*blockIdx.y; // time selection

  //unsigned int *in_offset = (unsigned int *)in + POLBLOCKSIZE*threadIdx.x + ninput/4*blockIdx.x + ninput/4*nfreq*NTIME_GATHER*blockIdx.y;
  unsigned int *in_offset = (unsigned int *)in + POLBLOCKSIZE*threadIdx.x + ninput/4*(threadIdx.y + CHANBLOCKSIZE*blockIdx.x) + ninput/4*nfreq*NTIME_GATHER*blockIdx.y;

  unsigned int x[NTIME_GATHER][POLBLOCKSIZE]; //4 times x 4*POLBLOCKSIZE pols (as POLBLOCKSIZE x 4-pols-per-word)
  int pol, i, t;
  for (t=0; t<NTIME_GATHER; t++) {
    # pragma unroll (8)
    for (pol=0; pol<POLBLOCKSIZE; pol++) {
      x[t][pol] = in_offset[((t*nfreq*ninput)>>2) + pol];
    }
  }

  // output buffer is NTIME * nfreq * ninput * complexity x 4[times] x 1byte
  // u32 *out =  out +
  //             POLBLOCKSIZE * NTIME_GATHER * 2 * threadIdx.x + // Write 4*POLBLOCKSIZE pols, 2 complexities, 4 times per thread (4 chars per u32)
  //              NTIME_GATHER * ninput * 2 * (threadIdx.y CHANBLOCKSIZE*blockIdx.x) + // chans
  //              nfreq * ninput * NTIME_GATHER * 2 * blockIdx.y; // times

  //unsigned int *out_offset =  (unsigned int *)out + POLBLOCKSIZE*NTIME_GATHER*2*threadIdx.x + ninput*2*blockIdx.x + nfreq*ninput*2*blockIdx.y;
  unsigned int *out_offset =  (unsigned int *)out + POLBLOCKSIZE*NTIME_GATHER*2*threadIdx.x + (CHANBLOCKSIZE*blockIdx.x + threadIdx.y)*ninput*2 + nfreq*ninput*2*blockIdx.y;

  for (pol=0; pol<POLBLOCKSIZE; pol++) {
    for (i=0; i<4; i++) {
      out_offset[8*pol + 2*i]   = ((lut[(x[0][pol] >> (8*i + 4)) & 0xf]) << 24) +
                                  ((lut[(x[1][pol] >> (8*i + 4)) & 0xf]) << 16) +
                                  ((lut[(x[2][pol] >> (8*i + 4)) & 0xf]) << 8 ) +
                                  ((lut[(x[3][pol] >> (8*i + 4)) & 0xf]) << 0 );
      out_offset[8*pol + 2*i+1] = ((lut[(x[0][pol] >> (8*i    )) & 0xf]) << 24) +
                                  ((lut[(x[1][pol] >> (8*i    )) & 0xf]) << 16) +
                                  ((lut[(x[2][pol] >> (8*i    )) & 0xf]) << 8 ) +
                                  ((lut[(x[3][pol] >> (8*i    )) & 0xf]) << 0 );
    }
    //#pragma unroll
    //for (i=0; i<4; i++) {
    //  out_offset[8*pol + 2*i]   = (lut[(x[0][pol] & mask[2*i+1])] << (24 - 8*i + 4)) +
    //                              (lut[(x[1][pol] & mask[2*i+1])] << (16 - 8*i + 4)) +
    //                              (lut[(x[2][pol] & mask[2*i+1])] << (8  - 8*i + 4)) +
    //                              (lut[(x[3][pol] & mask[2*i+1])] << (0  - 8*i + 4)) ;
    //  out_offset[8*pol + 2*i+1] = (lut[(x[0][pol] & mask[2*i])]   << (24 - 8*i    )) +
    //                              (lut[(x[1][pol] & mask[2*i])]   << (16 - 8*i    )) +
    //                              (lut[(x[2][pol] & mask[2*i])]   << (8  - 8*i    )) +
    //                              (lut[(x[3][pol] & mask[2*i])]   << (0  - 8*i    )) ;
    //}
  }
  CUBE_END;
}
