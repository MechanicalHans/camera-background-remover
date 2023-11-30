#!/usr/bin/env python3

import argparse as ap # https://docs.python.org/3/library/argparse.html
import cv2 as cv # https://docs.opencv.org/
import pyvirtualcam as vc # https://letmaik.github.io/pyvirtualcam/
import mediapipe as mp # https://developers.google.com/mediapipe/api/solutions
import numpy as np # https://numpy.org/doc/

def get_command_line_arguments():
    parser = ap.ArgumentParser(description="Apply transparency to the background objects in a camera feed.")
    parser.add_argument('input', type=str, help="input camera path")
    parser.add_argument('output', type=str, help="output camera path")
    parser.add_argument('width', type=int, help="frame width")
    parser.add_argument('height', type=int, help="frame height")
    parser.add_argument('rate', type=int, help="target frame rate")
    parser.add_argument('codec', type=str, nargs='?', default='MJPG', help="input camera codec")
    return parser.parse_args()

def main():
    args = get_command_line_arguments()
    input = get_input_device(args.input, args.width, args.height, args.rate, args.codec)
    output = get_output_device(args.output, args.width, args.height, args.rate)
    segmentor = get_segmentor()

    while True:
        success, frame = input.read()
        if not success: break
        # cv2 uses BGR format but MediaPipe uses RGB format
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = segmentor.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2RGBA)
        frame[:, :, 3] = (256*(mask.segmentation_mask)).astype(np.uint8)
        output.send(frame)
        output.sleep_until_next_frame()

def get_input_device(path, width, height, rate, codec):
    physical = cv.VideoCapture()
    if not physical.open(path): raise ValueError("invalid physical camera")
    try: fourcc = cv.VideoWriter_fourcc(*codec)
    except TypeError: raise ValueError("invalid codec")
    if not physical.set(cv.CAP_PROP_FOURCC, fourcc): raise ValueError("invalid codec")
    if not physical.set(cv.CAP_PROP_FRAME_WIDTH, width): raise ValueError("invalid frame width")
    if not physical.set(cv.CAP_PROP_FRAME_HEIGHT, height): raise ValueError("invalid frame height")
    if not physical.set(cv.CAP_PROP_FPS, rate): raise ValueError("invalid frame rate")
    return physical

def get_output_device(path, width, height, rate):
    return vc.Camera(width, height, rate, device=path, fmt=vc.PixelFormat.RGB)

def get_segmentor():
    return mp.solutions.selfie_segmentation.SelfieSegmentation()

if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: pass
