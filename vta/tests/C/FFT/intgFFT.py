import numpy as np
import math

## /*make the data
N = 4096
dc = 1
harm_value = [0, 0.2, 0.1, 0, 0, 0, 0]
data = []
for i in range(N):
    for j in range(len(harm_value)):
        if j ==0:
            data.append(dc)
        else:
            data[i] += harm_value[j] * np.sin(j*2*math.pi * i /N)
data = np.array(data)
print("data = ", data)

## random data
np.random.seed(1)
data_orig = np.random.uniform(-1, 1, size=4096)
data_orig = data_orig.astype(np.complex128)

## 
data_orig = data

fft_orig = np.fft.fft(data_orig)
print("fft_orig=",fft_orig[:16])


## 4x DFT
# [1x4096] == 256x[1x16]
# 256*input = [1x16]
# weight = [16*16]
def split_array(arr, depth=0):
    if depth == 8:  
        return [arr]
    odd_elements = arr[::2]
    even_elements = arr[1::2]
    sub_arrays = split_array(odd_elements, depth + 1) + split_array(even_elements, depth + 1)
    return sub_arrays
split_arrays = split_array(data_orig)
data_arrays = [element for subarray in split_arrays for element in subarray]


W_N_16 = np.exp(-1j * 2 * np.pi / 16) 
weight_orig = np.empty((16, 16), dtype=complex)
for i in range(16):
    for j in range(16):
        weight_orig[i,j] = W_N_16 ** (i*j)

res_ref = np.zeros(4096).astype(np.complex128)
fft_1 = np.zeros(4096).astype(np.complex128)
for k in range(4096//16):
    p = k * 16
    fft_1[p:p+16] = np.fft.fft(data_arrays[p: p+16])
    res_ref[p: p+16] = data_arrays[p: p+16] @ weight_orig.T

## the first step have the right output 
# print("res_ref = ",res_ref[1:16])
# print("fft_1 = ",fft_1[:16])


## the second step: 4x FFT SIMD
print("==========the second step: 4x FFT SIMD============")
# /*-----check the output*/
def split_array(arr, depth=0):
    if depth == 4:  
        return [arr]
    odd_elements = arr[::2]
    even_elements = arr[1::2]
    sub_arrays = split_array(odd_elements, depth + 1) + split_array(even_elements, depth + 1)
    return sub_arrays
split_arrays = split_array(data_orig)
data_arraysx8 = [element for subarray in split_arrays for element in subarray]

for k in range(4096//256):
    p = k * 256
    fft_1[p:p+256] = np.fft.fft(data_arraysx8[p: p+256])
#---------*/

# /* index =1 ; input[[0,16,256],[256,16,512],...,[3840,16,4096]]
# /* weight = 256*[]_{1*16}
W_N_4096 = np.exp(-1j * 2 * np.pi / 4096) 
weightx16 = np.array([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15])*2**4 ##这里需要推理一下，往回倒了4级
indices = np.repeat(np.arange(256), 16).reshape(256, 16)
print("indices",indices)
print("indices* weightx16 ",indices * weightx16)
weight_orig =np.power(W_N_4096, indices * weightx16)

# /*reshape the weight[256*16] to [256*1*16]
weight_resized = weight_orig.reshape((weight_orig.shape[0], 1, weight_orig.shape[1]))
print("weight_orig",weight_orig[1])
# print("weight_resized", weight_resized[0])

# /*reshape the data
data_orig = res_ref
data_inp = np.zeros((16, 16, 16), dtype=np.complex128)
# /*data_inp_reshape
for start_index in range(16):
    for row in range(16):
        current_start = start_index + row * 256
        current_end = current_start + 16*16
        data_inp[start_index, row, :] = data_orig[current_start:current_end:16]
# print("data_inp",data_inp)
print("data_inp.shape",data_inp[0].shape)
# print("data_inp",data_inp[0])

# /*目前是把data_inp load 16次，但理论上应该可以将load不变data不变，改变weight（weight的排布变一下，这个可以之后再说）
# print("weight_resized[0]",weight_resized[0])
# print("data_inp[0].T",data_inp[0].T)
opt = weight_resized[0] @ data_inp[0].T
# print("opt = ",opt)

opt = np.zeros((256,16), dtype=np.complex128)
for i in range(256):
    opt[i] = weight_resized[i] @ data_inp[i%16].T
# print("opt[0] = ",opt[0])

opt_resized = opt.reshape((opt.shape[0], 1, opt.shape[1]))

## /*---- for the data check 
fft_1 = fft_1.reshape(16,-1)
fft_1 = fft_1.T #[0 1 ... 256,257 ..., ...]->[0 256 ..., 1 257 ..., ...]
fft_1 = fft_1.reshape((opt.shape[0], 1, opt.shape[1]))
print("fft_1.shape",fft_1.shape)
print("fft_1 = ",fft_1[0][0][1])
for i in range(2):
    for j in range(16):
        if abs(fft_1[i][0][j] -opt_resized[i][0][j]) < 1e-5:
            print("pass",i,opt_resized[i][0][j],fft_1[i][0][j])    
## ------*/

## the third step: 4x intgration FFT
# /*input 256*[1*16]
# /*weight [256*16*16]
print("==========the third step: 4x intgration FFT============")

W_N_4096 = np.exp(-1j * 2 * np.pi / 4096) 
inpx16 = np.array([0,256,512,768,1024,1280,1536,1792,2048,2304,2560,2816,3072,3328,3584,3840]).reshape(16,-1)
weightx16 = np.array([0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]).reshape(1,-1)
weight_orig = np.zeros((256, 16, 16),dtype=np.complex128)
for i in range(256):
    inp = inpx16 + i
    indices = inp @ weightx16
    weight_orig[i] =np.power(W_N_4096, indices)

# print("inpx16",inpx16)
print("weightx16.shape",weightx16.shape)
print("indices.shape",indices.shape)

print("weight_orig.shape",weight_orig.shape)
print("opt_resized.shape",opt_resized.shape)

opt = np.zeros((256,16), dtype=np.complex128)
for i in range(256):
    opt[i,:] = opt_resized[i,:] @ weight_orig[i].T

opt = opt.T.flatten()
# print("opt.shape",opt.shape)

flag = 0
for i in range(400):
    if abs(opt[i]-fft_orig[i]) > 1e-3:
        flag = 1
        print("error",i,opt[i],fft_orig[i])
if flag == 0:
    print("the same!")


## 能不能把权重拿出来算完乘累加再乘，这样就不用存这么多权重数了

