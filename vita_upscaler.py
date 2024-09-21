import os
import ctypes
import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import cv2
import threading
import time
import keyboard
import pyautogui
from win32api import GetSystemMetrics
import os
import cv2, random
from datetime import datetime
from PIL import Image
import numpy as np
pyautogui.FAILSAFE = False

def screenshot_capture(raw_frame, scaled_frame):
    # Create the screenshots directory if it doesn't exist
    screenshot_dir = "vitaScreenshots"
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)

    # Generate a unique identifier using the current date and time
    unique_identifier = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the raw frame (non-scaled) using OpenCV
    raw_filename = os.path.join(screenshot_dir, f"{unique_identifier}_raw.png")
    cv2.imwrite(raw_filename, raw_frame)
    print(f"Raw frame saved as {raw_filename}")

    # Save the upscaled frame
    scaled_filename = os.path.join(screenshot_dir, f"{unique_identifier}_upscaled.png")
    cv2.imwrite(scaled_filename, scaled_frame)
    print(f"Upscaled frame saved as {scaled_filename}")

# Global variable to control the stream
halt_flag = False

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
    return cv2.resize(frame, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)

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
    height, width = frame.shape[:2]
    edges = upscale_image(detect_edges(frame))
    frame = variable_blur(upscale_image(frame), edges, ksize=5)
    frame = adjust_gamma(frame)
    return frame

def render_frame(frame, shader_program, vao, texture_id):
    global processed_frame
    processed_frame = process_frame(frame)  # Process the frame here
    load_texture(texture_id, processed_frame)  # Update the texture with the processed frame

    glUseProgram(shader_program)
    glBindVertexArray(vao)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(glGetUniformLocation(shader_program, "image"), 0)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)

def load_image_as_texture(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load splash image: {image_path}")
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB format for OpenGL
    return image

def load_texture_from_image(texture_id, image):
    image = image[:, :, [2, 1, 0]] 
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)


def display_splash_screen(shader_program, vao, texture_id, splash_image, window):
    load_texture_from_image(texture_id, splash_image)

    glClear(GL_COLOR_BUFFER_BIT)
    glUseProgram(shader_program)
    glBindVertexArray(vao)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(glGetUniformLocation(shader_program, "image"), 0)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)
    glfw.swap_buffers(window)

def display_psvita_stream(device_index):
    global halt_flag
    global processed_frame
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open the selected device.")
        return

    if not glfw.init():
        return

    monitor = glfw.get_primary_monitor()
    window = glfw.create_window(GetSystemMetrics(0), GetSystemMetrics(1), "OpenGL Window", monitor, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    shader_program = load_shader(vertex_shader_code, fragment_shader_code)
    vao = setup_quad()
    texture_id = create_texture()  # Create a texture once

    # Load the splash image
    splash_image = load_image_as_texture("splash.png")
    if splash_image is None:
        glfw.terminate()
        return

    while not halt_flag:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Displaying splash screen.")
            if keyboard.is_pressed("escape"):
                halt_flag = True
            display_splash_screen(shader_program, vao, texture_id, splash_image, window)

            # Check for reconnection every second
            time.sleep(0.05)
            cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                continue  # Stay in the splash screen until the device is reconnected
            continue  # Continue to next iteration if the device is reconnected
        else:
            #render and display upscaled image
            glClear(GL_COLOR_BUFFER_BIT)
            render_frame(frame, shader_program, vao, texture_id)
            glfw.swap_buffers(window)
            glfw.poll_events()
            if keyboard.is_pressed("F12"):
                screenshot_capture(frame, processed_frame)
            if keyboard.is_pressed("escape"):
                halt_flag = True

    glDeleteTextures(1, [texture_id])  # Clean up texture
    cap.release()
    glfw.terminate()

def keep_awake():
    global halt_flag
    while not halt_flag:
        pyautogui.press('left')
        time.sleep(2.5)

def list_available_devices():
    available_devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_devices.append(i)
            cap.release()
    return available_devices

if __name__ == "__main__":
    available_devices = list_available_devices()
    if not available_devices:
        print("No video capture devices found.")
        time.sleep(2)
        quit()

    print("Available devices:")
    for idx, device in enumerate(available_devices):
        print(f"{idx + 1}: Device {device}")

    selected_device_index = int(input("Select a device by number: ")) - 1
    if selected_device_index < 0 or selected_device_index >= len(available_devices):
        print("Invalid selection.")
        quit()

    selected_device = available_devices[selected_device_index]

    # Start the keep awake thread
    thread = threading.Thread(target=keep_awake)
    thread.start()

    # Start the video stream
    display_psvita_stream(selected_device)
