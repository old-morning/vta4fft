## FFT_BENCHMARK
## INPUT_type: complex128
## 

import numpy as np


## dft+fft*8
def mixfft_8(data):
    W_N_8 = np.exp(-1j * 2 * np.pi / 8)
    W_N_4 = np.exp(-1j * 2 * np.pi / 4)
    
    # fft in py
    fft_result = np.fft.fft(data)
    ref_res = np.abs(fft_result)
    # print(ref_res[:8])

    #bit reverse
    print("data: ",data)
    data_even = data[1::2]
    data_odd = data[::2]
    data = np.concatenate((data_odd, data_even))
    print("data_reverse: ",data)
    
    # dft
    widdle_x = np.empty((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            widdle_x[i,j] =  W_N_4 ** (i*j)
    print(widdle_x)
    for k in range(8//4):
        p = k * 4
        data[p: p+4] = data[p: p+4] @ widdle_x.T
        print("p =",p)
        print(data[p: p+4])

    # fft
    k = np.arange(4)
    widdle =  W_N_8 ** (k)
    print("widdle =",widdle )
    tmp = data[4:8] * widdle
    data[4:8] = data[0:4] - tmp
    data[0:4] = data[0:4] + tmp
    
    print("data = ",data)
    print("fft_result = ",fft_result)

    ## test
    res = np.abs(data)
    for i in range(8):
        if abs(ref_res[i]-res[i]) > 1e-5:
            print("error",i,ref_res[i],res[i])


## dft+fft*32
def mixfft_32(data):
    W_N_32 = np.exp(-1j * 2 * np.pi / 32)
    W_N_8 = np.exp(-1j * 2 * np.pi / 8)
    
    # fft in py
    fft_result = np.fft.fft(data)
    ref_res = np.abs(fft_result)

    # bit reverse
    print("data: ",data)
    data_a = data[::4]
    data_b = data[1::4]
    data_c = data[2::4]
    data_d = data[3::4]
    data = np.concatenate((data_a, data_c, data_b, data_d))
    print("data_reverse: ",data)

    # dft
    widdle_x = np.empty((8, 8), dtype=complex)
    for i in range(8):
        for j in range(8):
            widdle_x[i,j] =  W_N_8 ** (i*j)
    print(widdle_x)
    for k in range(32//8):
        p = k * 8
        fft_1 = np.fft.fft(data[p: p+8])
        data[p: p+8] = data[p: p+8] @ widdle_x.T
        print("fft_1= ",fft_1)
        print("p =",p)
        print(data[p: p+8])

    # fft
    k = np.arange(8)
    widdle =  W_N_32 ** (2*k)
    tmp = data[8:16] * widdle
    data[8:16] = data[0:8] - tmp
    data[0:8] = data[0:8] + tmp

    tmp = data[24:32] * widdle
    data[24:32] = data[16:24] - tmp
    data[16:24] = data[16:24] + tmp

    k = np.arange(16)
    widdle =  W_N_32 ** (k)
    tmp = data[16:32] * widdle
    data[16:32] = data[0:16] - tmp
    data[0:16] += tmp

    ## test
    res = np.abs(data)
    for i in range(32):
        if abs(ref_res[i]-res[i]) > 1e-5:
            print("error",i,ref_res[i],res[i])


## dft+fft *4096
def mixfft_4096(data):
    W_N_4096 = np.exp(-1j * 2 * np.pi / 4096)
    W_N_128 = np.exp(-1j * 2 * np.pi / 128) 
    
    # fft in py
    fft_result = np.fft.fft(data)
    ref_res = np.abs(fft_result)    
    
    # bit reverse  
    def split_array(arr, depth=0):
        if depth == 5:  
            return [arr]
        odd_elements = arr[::2]
        even_elements = arr[1::2]
        sub_arrays = split_array(odd_elements, depth + 1) + split_array(even_elements, depth + 1)
        return sub_arrays
    
    
    split_arrays = split_array(data)
    data = [element for subarray in split_arrays for element in subarray]
    # print(f"Total Subarrays: {len(split_arrays)}")
    
    # dft
    widdle_x = np.empty((128, 128), dtype=complex)
    for i in range(128):
        for j in range(128):
            widdle_x[i,j] =  W_N_128 ** (i*j)
    print(widdle_x[:8])
    
    for k in range(4096//128):
        p = k * 128
        # fft_1 = np.fft.fft(data[p: p+128])
        # data_a = 0
        # for i in range(128):
        #     data_a += data[p+i] * (W_N_128 **(i*0))
        data[p: p+128] = data[p: p+128] @ widdle_x.T    
    print("data=", data[:30])
    
    # fft
    for st in range(4,-1,-1):
        btfly_num = 2 ** st
        btfly_len = 4096 // btfly_num
        k = np.arange(btfly_len / 2)
        widdle =  W_N_4096 ** ((2 ** st) * k)        
        for i in range(btfly_num):
            p = i * btfly_len
            tmp = data[p + btfly_len // 2: p + btfly_len] * widdle
            data[p + btfly_len // 2: p + btfly_len] = data[p + btfly_len // 2: p + btfly_len] - tmp
            data[p : p + btfly_len // 2] = data[p : p + btfly_len // 2] + tmp    

    #test
    res = np.abs(data)
    for i in range(32,100):
        if abs(ref_res[i]-res[i]) > 1e-5:
            print("error",i,ref_res[i],res[i])


## date generator
np.random.seed(1)

data = np.random.randint(-128, 128, size=4096)
data = data.astype(np.complex128)
print(data[:8])

# mixfft_32(data)
mixfft_4096(data)