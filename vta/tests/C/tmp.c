#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdint.h>


void main(){
    int32_t M = 4;
    int32_t N = 4;
    int32_t K = 2;

    int32_t *gA  =(int32_t *)malloc(M*N*sizeof(int32_t));
    for (int32_t i=0; i<M*N; i++){
        gA[i] = i;
    }

    int32_t *gB  =(int32_t *)malloc(M*N*sizeof(int32_t));
    for (int32_t i=0; i<M*N; i++){
        gB[i] = i;
    }
    
    //Temp Sum
    int32_t *tS  =(int32_t *)malloc(M*N*sizeof(int32_t));
    for (int32_t i=0; i<M*N; i++){
        tS[i] = 0;
    }

    //VTA Parameters
    int32_t BATCH = 1;
    int32_t BLOCK_IN = 16; 
    int32_t BLOCK_OUT = 16;

    for (int32_t m=0; m<M; m++){
        //N columns
        for (int32_t i0=0; i0<N; i0+=BLOCK_OUT){
            // Block_out numbers
            for(int32_t i1=0; i1<BLOCK_OUT; i1++){
                // single row x sigle column = one output number
                for(int32_t j0=0; j0<K; j0+=BLOCK_IN){
                    // pick 16 numbers once
                    for(int32_t j1=0; j1<BLOCK_IN; j1++){
                        tS[m*N+i0+i1] += gA[m*K+j0+j1] * gB[i0*K+j0*BLOCK_OUT+i1*BLOCK_IN+j1];
                    }
                }
            }
        }
    }
    
    return 0;
}