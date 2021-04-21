from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader

from TextureLoader import load_texture
# ******************************
from ObjLoader import ObjLoader
# ******************************

from PIL import Image

import pyrr
import glfw
import numpy as np
import cv2
import time

from argparse import ArgumentParser

import torch
from torch import nn
from torchvision import transforms, models
from torch.autograd import Variable
import sys
sys.path.insert(0, "network")

from handnet_mask import HandNetInitial
from handnet_s import HandNet

URL = "http://192.168.1.59:8080/video"

vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec2 a_texture;
layout(location = 2) in vec3 a_normal;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
out vec2 v_texture;
void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    v_texture = a_texture;
}
"""

fragment_src = """
# version 330

in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;
void main()
{
    out_color = texture(s_texture, v_texture);
}
"""

# parameters
window_width = 320
window_height = 320
img_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# frame info
ts = []
frame_count = 0

# model
deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False, progress=True, num_classes=2)
class HandSegModel(nn.Module):
    def __init__(self):
        super(HandSegModel,self).__init__()
        self.dl = deeplab
        
    def forward(self, x):
        y = self.dl(x)['out']
        return y
        
# crop image to size 320x320
def crop_image(img):
    return img[320:640, 0:0+320]

# convert the opencv frame color to RGB
def opencv_to_pil(opencv_img):
    return Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))

# convert the opencv frame color to BGR
def pil_to_opencv(pil_img, channel=3):
    opencv_image = np.array(pil_img)
    if channel == 3:
        # Convert RGB to BGR
        opencv_image = opencv_image[:, :, ::-1].copy()
    return opencv_image

'''
def window_resize(window, win_width, win_height):
    glViewport(0, 0, win_width, win_height)
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, float(win_width) / float(win_height), 0.1, 100)
    proj_loc = glGetUniformLocation(shader, "projection")
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
'''

# initialize glfw (the window properties)
def initialize_glfw(win_width, win_height, pos_x, pos_y):
    if not glfw.init():
        raise Exception("glfw can not be initialized!")
    window = glfw.create_window(win_width, win_height, "OpenGL window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("glfw window can not be created!")
    glfw.set_window_pos(window, pos_x, pos_y)
    # glfw.set_window_size_callback(window, window_resize)
    # glfw.set_key_callback(window, key_input_clb)
    glfw.make_context_current(window)
    return window

# initialize shader data
def initialize_shader_data(vertex_shader, fragment_shader):
    shader = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER), compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    return shader

def pre_assignment(shader):
    projection = pyrr.matrix44.create_perspective_projection_matrix(45, window_width / window_height, 0.1, 100)
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 8]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
    proj_location = glGetUniformLocation(shader, "projection")
    view_location = glGetUniformLocation(shader, "view")
    glUniformMatrix4fv(proj_location, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_location, 1, GL_FALSE, view)

# initialize opengl
# shader, *sending projection and view matrix (do once in the program)
def initialize_opengl():
    shader = initialize_shader_data(vertex_src, fragment_src)
    glUseProgram(shader)
    glClearColor(0, 0, 1, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    pre_assignment(shader)
    return shader # for multi-shader, use array.

# load the OBJ file and bind the data to opengl
def load_object_data(filename):
    # OBJ loader to load the indice data from the OBJ file
    object_indices, object_buffer = ObjLoader.load_model(filename)
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)
    # Vertex Array Object
    glBindVertexArray(VAO)
    # Vertex Buffer Object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, object_buffer.nbytes, object_buffer, GL_STATIC_DRAW)
    # vertices
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(0))
    # textures
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(12))
    # normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, object_buffer.itemsize * 8, ctypes.c_void_p(20))
    glEnableVertexAttribArray(2)
    return object_indices, VAO

# apply the texture and return the texture id
def applying_object_texture(filename):
    # load monkey texture 
    object_texture = glGenTextures(1)
    # return the texture id
    return load_texture(filename, object_texture)

# translation matrix
def position(position):
    return pyrr.matrix44.create_from_translation(pyrr.Vector3(position))

# draw object in the opengl scene
def draw_object(shader, VAO, texture, model, indices):
    glBindVertexArray(VAO)
    glBindTexture(GL_TEXTURE_2D, texture)
    model_location = glGetUniformLocation(shader, "model")
    glUniformMatrix4fv(model_location, 1, GL_FALSE, model)
    glDrawArrays(GL_TRIANGLES, 0, len(indices))
    glBindVertexArray(0)

def initialize_webcam():
    video_capture = cv2.VideoCapture(URL)
    video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 60)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    video_capture.set(cv2.CAP_PROP_FPS, 60)
    return video_capture

def load_camera_matrix(path):
    from camera_calibration import load_coefficients
    return load_coefficients(path)

def display_current_frame(window, video_capture):
    ret, frame = video_capture.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()
    resizedFrame = cv2.resize(frame, (window_width, window_height))
    global frame_count, ts
    frame_count += 1
    
    t = time.time()

    hand = np.asarray(resizedFrame)
    
    gray = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2GRAY)

    #-----------GL frame processing----------
    frame_width, frame_height = glfw.get_framebuffer_size(window)
    pixels = glReadPixels(0, 0, window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE)

    glBuffer = cv2.cvtColor(np.flipud(np.frombuffer(pixels, np.uint8).reshape((frame_height, frame_width, 4))), cv2.COLOR_RGB2BGR)
    glBuffer2gray = cv2.cvtColor(glBuffer, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(glBuffer2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    cameraFrame = cv2.bitwise_and(resizedFrame, resizedFrame, mask=mask_inv)
    newGlBuffer = cv2.bitwise_and(glBuffer, glBuffer, mask=mask)
    #-----------------------------------------
    dst = cv2.add(cameraFrame, newGlBuffer)

    # cameraFrame = crop_image(cameraFrame)
    # dst = crop_image(dst)
    # hand = crop_image(hand)
    if args.debug:
        '''
        cv2.imshow('object_mask', cameraFrame)
        cv2.imshow('background removed object', newGlBuffer)
        cv2.imshow('object', dst)
        cv2.imshow('hand', hand)
        '''
        cv2.imshow('object_mask, background removed object, object, hand', np.hstack((cameraFrame, newGlBuffer, dst, hand)))
    
    final_result = call_network(dst, cameraFrame, hand)
    
    dt = time.time()-t
    ts += [dt]
    FPS = int(1/(np.mean(ts[-10:])+1e-6))   # compute the mean fps.
    print('\r', '%d'%FPS, end=' ')
    
    if final_result is not None:
        final_result =  np.asarray(final_result, dtype=np.uint8)
        cv2.putText(final_result, "FPS: %d"%FPS, (40, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)
        cv2.imshow('final_result', final_result)

def call_network(object_frame, object_mask, hand):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net = HandNetInitial().to(device)
    net.load_state_dict(torch.load('44.pth.tar')['state_dict'])  
    net.eval()

    handseg_net = HandNet().to(device)
    handseg_net.load_state_dict(torch.load('hand_seg.tar')['state_dict'])
    
    if(object_frame is not None and object_mask is not None):
        pil_hand_frame = opencv_to_pil(hand)
        pil_object_frame = opencv_to_pil(object_mask)
        hand_var = Variable(img_transform(pil_hand_frame).unsqueeze(0)).to(device)
        object_var = Variable(img_transform(pil_object_frame).unsqueeze(0)).to(device)

        with torch.no_grad():
            res = handseg_net(hand_var)
        confidence = res[0].data.squeeze(0).cpu().numpy()
        
        hand_mask = np.argmax(confidence, axis=0)
        hand_mask = np.uint8(hand_mask)
        
        hand_mask[np.where(hand_mask == 1)] = 255 # background removing
        
        hand[np.where(hand_mask != 255)] = np.array((119, 178, 78)).astype(np.uint8)
        
        if args.debug:
            cv2.imshow('hand_segmentation', hand)

        pil_hand_frame = opencv_to_pil(hand)
        hand_var = Variable(img_transform(pil_hand_frame).unsqueeze(0)).to(device)

        with torch.no_grad():
            res, _ = net(hand_var, object_var, object_var, torch.tensor([1]))
        confidence = res[0].data.squeeze(0).cpu().numpy()

        ## mask ##
        mask = np.argmax(confidence, axis=0)
        mask = np.uint8(mask)
        mask[np.where(mask == 1)] = 128  # object
        mask[np.where(mask == 2)] = 255  # hand

        mask = cv2.medianBlur(mask, 7)
        ret, mask = cv2.threshold(mask, 129, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        object_frame[np.where(np.logical_and(mask==255, hand_mask==255))] = \
            hand[np.where(np.logical_and(mask==255, hand_mask==255))]

        return object_frame

def main():
    init_time = time.time()

    window = initialize_glfw(window_width, window_height, 300, 400)
    shader = initialize_opengl()

    # ** if the file not contain mtl.
    # data from OBJ
    object_indices, object_VAO = load_object_data("meshes/monkey.obj")
    object_texture = applying_object_texture("meshes/monkey.jpg")
    # ** need to improve the mtl reader.
    # initial object position
    object_position = position([0,0,-2.5])
    model_matrix = pyrr.matrix44.multiply(1.0, object_position)
    
    video_capture = initialize_webcam()

    # the camera matrix is specified for different camera used,
    # S8+ camera be used in the test.
    camera_matrix, dist_matrix = load_camera_matrix("./camera.yml")

    print("initialization time: ", time.time() - init_time)

    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClearColor(0.0,0.0,0.0,1.0)

        draw_object(shader, object_VAO, object_texture, model_matrix, object_indices)
        display_current_frame(window, video_capture)
        # display_camera(video_capture, )

        glfw.swap_buffers(window)
        glfw.poll_events()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--cam", type=int, help="", default=0)
    parser.add_argument("--cpu", action="store_true", help="", default=False)
    parser.add_argument("--debug", action="store_true",  help="", default=False)
    parser.add_argument("--connect", action="store_true",  help="", default=False)
    parser.add_argument("--fps", action="store_true",  help="", default=False)
    args = parser.parse_args()
    main()
