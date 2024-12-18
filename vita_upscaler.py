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
global upscale_factor, upscaling_type, gamma, enable_taa
#upscale_factor = 2

import configparser
config = configparser.ConfigParser(allow_no_value=True)
config.read('config.txt')

# Get the configuration values from the DEFAULT section
upscale_factor = config.getfloat('DEFAULT', 'upscale_factor')
enable_taa  = config.getboolean('DEFAULT', 'taa')
upscaling_type = config.getint('DEFAULT', 'upscaling_type')
gamma = config.getfloat('DEFAULT', 'gamma')

# Global variables to store previous frames and alpha values
previous_frame_1 = None
previous_frame_2 = None
previous_frame_3 = None

try:
    taa_alpha_1 = config.getfloat('DEFAULT', 'taa_alpha_1')
    taa_alpha_2 = config.getfloat('DEFAULT', 'taa_alpha_2')
    taa_alpha_3 = config.getfloat('DEFAULT', 'taa_alpha_3')
except:
    print("Incorrect taa values")
    taa_alpha_1 = 0.24
    taa_alpha_2 = 0.18
    taa_alpha_3 = 0.12
print(taa_alpha_1, taa_alpha_2, taa_alpha_3)

def check_ocl():
    try:
        # Returns True if OpenCL is present
        ocl = cv2.ocl.haveOpenCL()
        # Prints whether OpenCL is present
        print("OpenCL Supported?: ", end='')
        print(ocl)
        print()
        return ocl

    except cv2.error as e:
        print('Error:')
        
ocl = check_ocl()
# Enables use of OpenCL by OpenCV if present
if ocl == True:
    print('Now enabling OpenCL support')
    cv2.ocl.setUseOpenCL(True)
    print("Has OpenCL been Enabled?: ", end='')
    print(cv2.ocl.useOpenCL())

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
print("Taa: ", enable_taa)
print(f"Applying upscaling factor of {upscale_factor} and gamma correction {gamma}.")
if upscaling_type == 1:
    print("Using linear upscaler.")
elif upscaling_type == 2:
    print("Using vitaScaler single pass.")
elif upscaling_type == 3:
    print("Using vitaScaler double pass.")


def screenshot_capture(raw_frame, scaled_frame):
    raw_frame = cv2.resize(raw_frame, (GetSystemMetrics(0), GetSystemMetrics(1)), interpolation=cv2.INTER_NEAREST)
    scaled_frame = cv2.resize(scaled_frame, (GetSystemMetrics(0), GetSystemMetrics(1)), interpolation=cv2.INTER_LINEAR)
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
    """
    Upscale an image using OpenCL-accelerated cv2.UMat.
    """
    height, width = frame.get().shape[:2]  # Get dimensions from the numpy representation
    upscaled = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
    return upscaled

def detect_edges(frame):
    """
    Detect edges using OpenCL-accelerated cv2.Canny.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    edges = cv2.Canny(gray_frame, 100, 200)  # Perform edge detection
    _, binary_mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)  # Threshold the edges
    return binary_mask

def variable_blur(frame, edges, ksize=3):
    """
    Apply variable Gaussian blur based on edges using OpenCL acceleration.
    """
    blurred_frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)  # Blur the frame
    mask = cv2.bitwise_not(edges)  # Invert the edges mask
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
    blurred_with_edges = cv2.addWeighted(frame, 1.0, blurred_frame, 0.0, 0, mask)
    return blurred_with_edges

def precompute_gamma_lut(gamma=0.82):
    invGamma = 1.0 / gamma
    return np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")

# Global gamma LUT
gamma_lut = precompute_gamma_lut(gamma)

def adjust_gamma(frame, gamma=0.82):
    """
    Adjust gamma using the precomputed LUT.
    """
    return cv2.LUT(frame, gamma_lut)

def apply_taa(frame):
    """
    Apply Temporal Anti-Aliasing (TAA) using weighted averaging of pixels.
    """
    try:
        global previous_frame_1, previous_frame_2, previous_frame_3
        global taa_alpha_1, taa_alpha_2, taa_alpha_3

        # Initialize previous frames if they're not yet set
        if previous_frame_1 is None:
            previous_frame_1 = cv2.UMat(frame)
            return frame
        if previous_frame_2 is None:
            previous_frame_2 = cv2.UMat(frame)
            return frame
        if previous_frame_3 is None:
            previous_frame_3 = cv2.UMat(frame)
            return frame

        # Calculate the total weight for normalization
        total_weight = 1.0 + taa_alpha_1 + taa_alpha_2 + taa_alpha_3

        # Compute the weighted average of the frames
        blended_frame = cv2.addWeighted(frame, 1.0 / total_weight, previous_frame_1, taa_alpha_1 / total_weight, 0)
        blended_frame = cv2.addWeighted(blended_frame, 1.0, previous_frame_2, taa_alpha_2 / total_weight, 0)
        blended_frame = cv2.addWeighted(blended_frame, 1.0, previous_frame_3, taa_alpha_3 / total_weight, 0)

        # Update previous frames for the next iteration
        previous_frame_1 = cv2.UMat(previous_frame_2.get())
        previous_frame_2 = cv2.UMat(previous_frame_3.get())
        previous_frame_3 = cv2.UMat(frame)

        return blended_frame

    except Exception as e:
        print("TAA Error:", e)
        return frame

def process_frame(frame):
    """
    Process a frame using the OpenCL-accelerated pipeline with optimizations.
    """
    global upscale_factor, gamma, upscaling_type, enable_taa, gamma_lut

    # Convert frame to UMat at the start
    frame_umat = cv2.UMat(frame)

    # Apply upscaling and edge detection
    if upscaling_type == 2:
        frame_umat = upscale_image(frame_umat, upscale_factor)
        edges = detect_edges(frame_umat)
        frame_umat = variable_blur(frame_umat, edges, ksize=5)
    elif upscaling_type == 3:
        edges = detect_edges(frame_umat)
        frame_umat = variable_blur(frame_umat, edges, ksize=3)
        frame_umat = upscale_image(frame_umat, upscale_factor)
        edges = detect_edges(frame_umat)
        frame_umat = variable_blur(frame_umat, edges, ksize=5)

    # Apply gamma correction
    frame_umat = adjust_gamma(frame_umat, gamma)

    # Apply Temporal Anti-Aliasing (if enabled)
    if enable_taa:
        frame_umat = apply_taa(frame_umat)

    # Convert back to numpy array only at the end
    return frame_umat.get()


def render_frame(frame, shader_program, vao, texture_id):
    global processed_frame, upscaling_type, gamma
    if upscaling_type != 1:
        processed_frame = process_frame(frame)  # Process the frame here
    else:
        processed_frame = adjust_gamma(frame, gamma)
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
    global upscale_factor
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

