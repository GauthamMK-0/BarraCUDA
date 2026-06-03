/* test_struct.cu -- input kernels for the control-flow structuriser tests.
 * One of each shape the relooper has to put back together: a bare if, an
 * if/else, a counted for (two phis, induction and accumulator), and a while
 * carrying a break and a continue past a nested if. If the structuriser can
 * draw these without a goto between them, it has earned its keep. */

__global__ void st_if(int *o, int n) {
    int i = threadIdx.x;
    if (i < n) o[i] = 1;
}

__global__ void st_ifelse(int *o, int n) {
    int i = threadIdx.x;
    if (i < n) o[i] = 1; else o[i] = 2;
}

__global__ void st_for(float *o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int k = 0; k < n; k++) acc += o[k];
    o[i] = acc;
}

__global__ void st_whilebreak(int *o, int n) {
    int k = 0;
    while (k < n) {
        if (o[k] < 0) break;
        if (o[k] == 0) { k++; continue; }
        o[k] = o[k] * 2;
        k++;
    }
    o[0] = k;
}
