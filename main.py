import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
import numpy as np
import onnxruntime as ort
from numba import cuda, prange
from math import ceil
import subprocess
import numba
from helper.helper_function import preprocess, check_device


# CUDA kernel to multiply each channel of an image with a corresponding mask
@cuda.jit
def mask_image(ori, mask):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= ori.shape[0]:
        return
    if y >= ori.shape[1]:
        return
    for i in prange(ori.shape[-1]):
        ori[x, y, i] *= mask[x, y]


# CUDA kernel to remove noise from an image
@cuda.jit
def remove_noise(ori):
    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if x >= ori.shape[0]:
        return
    if y >= ori.shape[1]:
        return
    if ori[x, y] >= 2:
        ori[x, y] = 1
    else:
        ori[x, y] = 0


def main():
    # Check the available device (GPU or CPU)
    device = check_device()
    device = "CPU"
    W = 1280
    H = 720
    FPS = 30

    W_bg_remove = 224
    H_bg_remove = 224
    threads_per_block_2d_full_scale = 16, 16
    blocks_per_grid_2d_full_scale = ceil(H / 16), ceil(W / 16)

    # Read from webcam
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    webcam.set(3, W)
    webcam.set(4, H)

    # Initialize ONNX Runtime session for human segmentation
    EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        sess = ort.InferenceSession(
            "human_segment_int8.onnx", providers=EP_list, sess_options=sess_options
        )
        output_name = sess.get_outputs()[0].name
        input_name = sess.get_inputs()[0].name
        fake_image = np.ones([1, 3, 224, 224], dtype=np.float32)
        prediction = sess.run([output_name], {input_name: fake_image})[0]
    except:
        # If CUDAExecutionProvider fails, use CPUExecutionProvider
        sess = ort.InferenceSession(
            "human_segment_int8.onnx",
            providers=["CPUExecutionProvider"],
            sess_options=sess_options,
        )

    output_name = sess.get_outputs()[0].name
    input_name = sess.get_inputs()[0].name

    # Start virtual camera
    with pyvirtualcam.Camera(W, H, FPS, fmt=PixelFormat.BGR) as cam:
        print("Virtual camera device: " + cam.device)

        if webcam.isOpened():
            success = True
            while success:
                success, frame = webcam.read()
                if success:
                    # Preprocess the frame for human segmentation
                    frame_flip = cv2.flip(frame, 1)
                    image_data = preprocess(frame_flip, W_bg_remove, H_bg_remove)

                    # Run inference on the preprocessed frame
                    prediction = sess.run([output_name], {input_name: image_data})[0]
                    prediction = np.argmax(prediction[0], axis=0) * 255

                    # Post-process the segmentation mask
                    frame_sent = cv2.resize(
                        np.uint8(np.squeeze(prediction)),
                        (W, H),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    contours, hierarchy = cv2.findContours(
                        frame_sent, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
                    )

                    # Create a mask based on the largest contour
                    try:
                        contour = max(contours, key=cv2.contourArea)
                        frame_mask = np.zeros_like(frame_sent)
                        cv2.fillPoly(frame_mask, pts=[contour], color=(255, 255, 255))
                    except:
                        frame_mask = np.zeros_like(frame_sent)

                    # Apply the mask to the frame
                    if device == "GPU":
                        frame_mask = numba.cuda.to_device(
                            np.ascontiguousarray(frame_mask)
                        )
                        frame_flip = numba.cuda.to_device(
                            np.ascontiguousarray(frame_flip)
                        )
                        remove_noise[
                            blocks_per_grid_2d_full_scale,
                            threads_per_block_2d_full_scale,
                        ](frame_mask)
                        mask_image[
                            blocks_per_grid_2d_full_scale,
                            threads_per_block_2d_full_scale,
                        ](frame_flip, frame_mask)
                        frame_flip = frame_flip.copy_to_host()
                    else:
                        frame_mask[frame_mask >= 2] = 1
                        frame_mask[frame_mask != 1] = 0
                        for i in range(3):
                            frame_flip[:, :, i] = frame_flip[:, :, i] * frame_mask

                    # Display the processed frame
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key to exit
                        break
                    cam.send(frame_flip)
                    cam.sleep_until_next_frame()


if __name__ == "__main__":
    main()
