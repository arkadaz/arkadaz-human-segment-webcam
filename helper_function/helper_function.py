import cv2
import subprocess
import numpy as np

# cp_add_arrs = cp.RawKernel(r'''
# extern "C" __global__
# void add_arrs(int* x1, int* x2, int* y, int N){
#   int i = blockDim.x * blockIdx.x + threadIdx.x;
#   int j = blockDim.y * blockIdx.y + threadIdx.y;
  
#   if(i < N && j < N){
#     y[j + i*N] = x1[j + i*N] + x2[j + i*N];
#   }
# }
# ''', 'add_arrs')

def preprocess(image, w, h):
    img = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)/255.0
    img -= (0.485, 0.456, 0.406)
    img /= (0.229, 0.224, 0.225)
    return img.transpose([2, 0, 1]).reshape(1, 3, 224, 224).astype(np.float32)


def check_device():
    check_device = "CPU"
    try:
        subprocess.check_call("nvidia-smi")
        print("GPU")
        check_device = "GPU"
    except:
        print("CPU")
    return check_device