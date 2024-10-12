import os
import ctypes
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import cv2
import threading
import time
import pyautogui
import configparser
from datetime import datetime
from PIL import Image
import numpy as np

pyautogui.FAILSAFE = False

# Read configuration values from config.txt
config = configparser.ConfigParser(allow_no_value=True)
config.read('config.txt')

upscale_factor = config.getfloat('DEFAULT', 'upscale_factor')
upscaling_type = config.getint('DEFAULT', 'upscaling_type')
gamma = config.getfloat('DEFAULT', 'gamma')

# Validate config values
if upscale_factor < 1:
    print("upscale_factor must be 1 or greater.")
    time.sleep(1)
    exit()

if upscaling_type not in [1, 2, 3]:
    print("upscaling_type must be 1 (linear), 2 (vitaScaler single pass), or 3 (vitaScaler double pass).")
    time.sleep(1)
    exit()

print(f"Upscale Factor: {upscale_factor}")
print(f"Upscaling Type: {upscaling_type}")
print(f"Gamma: {gamma}")
print(f"Applying upscaling factor of {upscale_factor} and gamma correction {gamma}.")
if upscaling_type == 1:
    print("Using linear upscaler.")
elif upscaling_type == 2:
    print("Using vitaScaler single pass.")
elif upscaling_type == 3:
    print("Using vitaScaler double pass.")

# Vertex data for a full-screen quad
vertices = np.array([
    -1.0, -1.0, 0.0, 0.0, 1.0,
     1.0, -1.0, 0.0, 1.0, 1.0,
     1.0,  1.0, 0.0, 1.0, 0.0,
    -1.0,  1.0, 0.0, 0.0, 0.0
], dtype=np.float32)

indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

vertex_shader_code = """#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

fragment_shader_code = """#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D image;

void main()
{
    vec4 Color = texture(image, TexCoord);
    float temp = Color.r;
    Color.r = Color.b;
    Color.b = temp;
    FragColor = Color;
}
"""

def load_shader(vertex_code, fragment_code):
    vertex_shader = compileShader(vertex_code, GL_VERTEX_SHADER)
    fragment_shader = compileShader(fragment_code, GL_FRAGMENT_SHADER)
    return compileProgram(vertex_shader, fragment_shader)

def setup_quad():
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    ebo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * vertices.itemsize, ctypes.c_void_p(3 * vertices.itemsize))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao

def create_texture():
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id

def load_texture(texture_id, frame):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, frame)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

def upscale_image(frame, scale_factor=2):
    height, width = frame.shape[:2]
    return cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)

def detect_edges(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, 100, 200)
    _, binary_mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
    return binary_mask

def variable_blur(frame, edges, ksize=3):
    blurred_frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    mask = cv2.bitwise_not(edges)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blurred_with_edges = np.where(mask == 255, frame, blurred_frame)
    return blurred_with_edges

def adjust_gamma(image, gamma=0.82):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def process_frame(frame):
    global upscale_factor, gamma, upscaling_type
    if upscaling_type == 2:
        frame = upscale_image(frame, upscale_factor)
        edges = detect_edges(frame)
        frame = variable_blur(frame, edges, ksize=5)
        frame = adjust_gamma(frame, gamma)
        return frame
    if upscaling_type == 3:
        edges = detect_edges(frame)
        frame = variable_blur(frame, edges, ksize=3)
        frame = upscale_image(frame, upscale_factor)
        edges = detect_edges(frame)
        frame = variable_blur(frame, edges, ksize=5)
        frame = adjust_gamma(frame, gamma)
        return frame

def render_frame(frame, shader_program, vao, texture_id):
    processed_frame = process_frame(frame) if upscaling_type != 1 else adjust_gamma(frame, gamma)
    load_texture(texture_id, processed_frame)

    glUseProgram(shader_program)
    glBindVertexArray(vao)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(glGetUniformLocation(shader_program, "image"), 0)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)

def list_available_devices():
    available_devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_devices.append(i)
            cap.release()
    return available_devices

def display_psvita_stream(device_index):
    global halt_flag
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open the selected device.")
        return

    pygame.init()

    # Get the screen resolution automatically
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h

    # Set the Pygame window to fullscreen mode and at full resolution
    screen = pygame.display.set_mode((screen_width, screen_height), pygame.OPENGL | pygame.DOUBLEBUF | pygame.FULLSCREEN)

    # Initialize OpenGL settings here
    shader_program = load_shader(vertex_shader_code, fragment_shader_code)
    vao = setup_quad()
    texture_id = create_texture()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            continue

        # Clear the OpenGL buffer
        glClear(GL_COLOR_BUFFER_BIT)

        # Render the frame and display
        render_frame(frame, shader_program, vao, texture_id)
        pygame.display.flip()

    cap.release()
    pygame.quit()

# Main
if __name__ == "__main__":
    available_devices = list_available_devices()
    if not available_devices:
        print("No video capture devices found.")
    else:
        print(f"Available devices: {available_devices}")
        try:
            device_idx = int(input("input a device index:\t"))
        except:
            device_idx = 0
        display_psvita_stream(available_devices[device_idx])