import numpy as np
import math
import matplotlib.pyplot as plt

batch_size = 32
channel = 128 * 2
env_BATCH = 1
env_BLOCK_IN = 16
env_BLOCK_OUT = 16

Q = 10
int16_MAX = 2 ** Q - 1
res_type = np.int32

def dft(Q, int16_MAX, res_type):
    np.random.seed(1)
    data_orig = np.random.uniform(-1, 1, size=4096)
    data_orig = data_orig.astype(np.complex128)

    # fft_orig = np.fft.fft(data_orig)
    # print("fft_orig=",fft_orig[:16])
    # print("MAX_fft_orig=",max(fft_orig))
    # print("MIN_fft_orig=",min(fft_orig))

    # bit reverse  
    def split_array(arr, depth=0):
        if depth == 5:  
            return [arr]
        odd_elements = arr[::2]
        even_elements = arr[1::2]
        sub_arrays = split_array(odd_elements, depth + 1) + split_array(even_elements, depth + 1)
        return sub_arrays
    split_arrays = split_array(data_orig)
    data_orig = [element for subarray in split_arrays for element in subarray]

    print("data_orig=",data_orig[:16])

    W_N_128 = np.exp(-1j * 2 * np.pi / 128) 
    weight_orig = np.empty((128, 128), dtype=complex)
    for i in range(128):
        for j in range(128):
            weight_orig[i,j] = W_N_128 ** (i*j)

    res_ref = np.zeros(4096).astype(np.complex128)
    for k in range(4096//128):
        p = k * 128
        # fft_1 = np.fft.fft(data_orig[p: p+128])
        # print("fft_1 = ",fft_1[:16])
        res_ref[p: p+128] = data_orig[p: p+128] @ weight_orig.T
    print("res_ref = ",res_ref[:16])

    # float 
    data_act = np.zeros(4096*2).astype(np.float128)
    for i in range(4096):
        data_act[2*i] = data_orig[i].real
        data_act[2*i+1] = data_orig[i].imag

    weight_act = np.empty((128*2, 128*2), dtype=np.float128)
    for i in range(128):
        for j in range(128):
            # (a+bj)(c+dj) = (ac-bd) + (ad+bc)j
            weight_act[2*i,2*j] = weight_orig[i,j].real
            weight_act[2*i,2*j+1] = -weight_orig[i,j].imag ## -imag
            weight_act[2*i+1,2*j] = weight_orig[i,j].imag ## imag
            weight_act[2*i+1,2*j+1] = weight_orig[i,j].real ## real

    res_ref_flot = np.zeros(4096*2).astype(np.float128)
    for k in range(4096//128):
        p = k * 256
        # fft_1 = np.fft.fft(data[p: p+128])
        res_ref_flot[p: p+256] = data_act[p: p+256] @ weight_act.T    
    print("res_ref_flot = ",res_ref_flot[:16])
    # print("data_flot = ",data_act[:16])
    # print("weight_flot = ",weight_act[:16])

    #float to int
    data_int = np.zeros(4096*2).astype(np.int16)
    for i in range(4096):
        data_int[2*i] = math.floor(0.5+data_orig[i].real*2**Q)
        data_int[2*i+1] = math.floor(0.5+data_orig[i].imag*2**Q)

    weight_int = np.empty((128*2, 128*2), dtype=np.int16)
    for i in range(128):
        for j in range(128):
            # (a+bj)(c+dj) = (ac-bd) + (ad+bc)j
            weight_int[2*i,2*j] = math.floor(0.5+weight_orig[i,j].real* int16_MAX) ## real
            weight_int[2*i,2*j+1] = -math.floor(0.5+weight_orig[i,j].imag* int16_MAX) ## -imag
            weight_int[2*i+1,2*j] = math.floor(0.5+weight_orig[i,j].imag* int16_MAX) ## imag
            weight_int[2*i+1,2*j+1] = math.floor(0.5+weight_orig[i,j].real* int16_MAX) ## real
    
    # print("data_int = ",data_int[:16])
    # print("weight_int = ",weight_int[:1])

    # data_a = 0
    # for i in range(256):
    #     data_a += (data_int[i].astype(np.int32)*weight_int[0,i].astype(np.int32))
    # print("data_a = ",data_a.astype(np.int64))

    res_ref_int = np.zeros(4096*2).astype(res_type)
    for k in range(4096//128):
        p = k * 256
        res_ref_int[p: p+256] = data_int[p: p+256].astype(res_type) @ weight_int.T.astype(res_type)
    print("type of res_ref_int = ",res_type)
    print("res_ref_int = ",res_ref_int[:16])
    res_ref_int = np.right_shift(res_ref_int, Q)
    res_ref_int = np.clip(res_ref_int, -(1 << 15) + 1, (1 << 15) - 1).astype(np.int16)
    # res_ref_int = res_ref_int.astype(np.int16)
    print("res_ref_int = ",res_ref_int[:16])

    # for VTA
    data_int = np.array(data_int).reshape(batch_size, channel)
    #data_shape : (batch_size // env.BATCH, channel // env.BLOCK_IN, env.BATCH, env.BLOCK_IN)
    data_packed = data_int.reshape(
        batch_size // env_BATCH, env_BATCH, channel // env_BLOCK_IN, env_BLOCK_IN
    ).transpose((0, 2, 1, 3))

    # weight_shape : (channel // env.BLOCK_OUT, channel // env.BLOCK_IN, env.BLOCK_OUT, env.BLOCK_IN)
    # weight_orig = np.transpose(weight_orig)
    weight_int = np.array(weight_int).reshape(channel, channel)
    weight_packed = weight_int.reshape(
        channel // env_BLOCK_OUT, env_BLOCK_OUT, channel // env_BLOCK_IN, env_BLOCK_IN
    ).transpose((0, 2, 1, 3))

    ## data.h
    with open ("data.h", "w") as f:
        inp = data_packed.reshape(32*256)
        # f.write("complx inp[] = {\n")
        # for i in range(32*128-1):
        #     f.write('{:.1f}+{:.1f}j, '.format(inp[i].real,inp[i].imag))
        # f.write('{:.1f}+{:.1f}j\n '.format(inp[32*128-1].real,inp[32*128-1].imag))
        # f.write('};\n')
        f.write("int16_t inp[] = {\n")
        for i in range(32*256-1):
            f.write('{:d}, '.format(inp[i]))
        f.write('{:d}\n '.format(inp[32*256-1]))
        f.write('};\n')

        wgt = weight_packed.reshape(256*256)
        # f.write("complx wgt[] = {\n")
        # for i in range(256*256-1):
        #     f.write('{:.1f}+{:.1f}j,'.format(wgt[i].real,wgt[i].imag))
        # f.write('{:.1f}+{:.1f}j\n'.format(wgt[128*128-1].real,wgt[128*128-1].imag))
        # f.write('};\n')
        f.write("int16_t wgt[] = {\n")
        for i in range(256*256-1):
            f.write('{:d}, '.format(wgt[i]))
        f.write('{:d}\n'.format(wgt[256*256-1]))
        f.write('};\n')


        ref = res_ref_int.reshape(32*256)
        f.write("int16_t res[] = {\n")
        for i in range(32*256-1):
            f.write('{:d}, '.format(ref[i]))
        f.write('{:d}\n'.format(ref[32*256-1]))
        f.write('};\n')

    # error
    res_ref_int = res_ref_int.astype(np.float32)
    res_ref_int /= (1 << Q)
    err_flot_int = np.abs(res_ref_int - res_ref_flot)
    max_error = np.max(err_flot_int)

    return max_error

## 量化与精度
def quant():
    error_32 = np.zeros(8)
    error_64 = np.zeros(8)
    for i in range(8):
        Q = 8+i
        int16_MAX = 2 ** Q - 1
        res_type = np.int32
        print("Q= ",Q)
        error_32[i] = dft(Q,int16_MAX,res_type)
        res_type = np.int64
        print("Q= ",Q)
        error_64[i] = dft(Q,int16_MAX,res_type)
    print( "error_32 = ", error_32)
    print( "error_64 = ", error_32)

    Q_values = [8 + i for i in range(8)]
    plt.figure(figsize=(10, 5))
    plt.plot(Q_values, error_32, label='int_32', marker='o')
    plt.plot(Q_values, error_64, label='int_64', marker='x')
    plt.title('Error Values for Q')
    plt.xlabel('Q')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('output.png') 

dft(Q,int16_MAX,res_type)
