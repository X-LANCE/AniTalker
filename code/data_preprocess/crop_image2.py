# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:36:01 2021

@author: Xinya
"""

import os
import glob
import time
import numpy as np
import csv
import cv2
import dlib

from skimage import transform as tf

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('code/data_preprocess/shape_predictor_68_face_landmarks.dat')


import imageio



def save(path, frames, format):
    if format == '.mp4':
        imageio.mimsave(path, frames)
    elif format == '.png':
        if not os.path.exists(path):


            os.makedirs(path)
        for j, frame in enumerate(frames):
            cv2.imwrite(path+'/'+str(j)+'.png',frame)
    #        imageio.imsave(os.path.join(path, str(j) + '.png'), frames[j])
    else:
        print ("Unknown format %s" % format)
        exit()

def crop_image(image_path, out_path):
    template = np.load('code/data_preprocess/M003_template.npy')
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)  #detect human face
    if len(rects) != 1:
        return 0
    for (j, rect) in enumerate(rects):
        shape = predictor(gray, rect) #detect 68 points
        shape = shape_to_np(shape)

    pts2 = np.float32(template[:47,:])
    # pts2 = np.float32(template[17:35,:])
    # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
    pts1 = np.float32(shape[:47,:]) #eye and nose
    # pts1 = np.float32(landmark[17:35,:])
    tform = tf.SimilarityTransform()
    tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
    
    dst = tf.warp(image, tform, output_shape=(256, 256))

    dst = np.array(dst * 255, dtype=np.uint8)
    
    
    cv2.imwrite(out_path,dst)

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def crop_image_tem(video_path, out_path):
    image_all = []
    videoCapture = cv2.VideoCapture(video_path)
    success, frame = videoCapture.read()
    n = 0
    while success :
        image_all.append(frame)
        n = n + 1
        success, frame = videoCapture.read()
        
    if len(image_all)!=0 :
        template = np.load('./M003_template.npy')
        image=image_all[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)  #detect human face
        if len(rects) != 1:
            return 0
        for (j, rect) in enumerate(rects):
            shape = predictor(gray, rect) #detect 68 points
            shape = shape_to_np(shape)

        pts2 = np.float32(template[:47,:])
        # pts2 = np.float32(template[17:35,:])
        # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
        pts1 = np.float32(shape[:47,:]) #eye and nose
        # pts1 = np.float32(landmark[17:35,:])
        tform = tf.SimilarityTransform()
        tform.estimate( pts2, pts1) #Set the transformation matrix with the explicit parameters.
        out = []
        for i in range(len(image_all)):
            image = image_all[i]
            dst = tf.warp(image, tform, output_shape=(256, 256))

            dst = np.array(dst * 255, dtype=np.uint8)
            out.append(dst)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        save(out_path,out,'.png')

def proc_audio(src_mouth_path, dst_audio_path):
    audio_command = 'ffmpeg -i \"{}\" -loglevel error -y -f wav -acodec pcm_s16le ' \
                    '-ar 16000 \"{}\"'.format(src_mouth_path, dst_audio_path)
    os.system(audio_command)



if __name__ == "__main__":


    image_path ='source_image.png'
    save_path = 'crop_image.png'
    crop_image(image_path, save_path)

