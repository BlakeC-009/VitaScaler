import os
import ctypes

script_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_dir, 'lib-vc2022')
os.environ['PATH'] = dll_path + ';' + os.environ['PATH']
glfw_library = ctypes.CDLL(os.path.join(dll_path, 'glfw3.dll'))

from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
import numpy as np
from PIL import Image
from win32api import GetSystemMetrics
import cv2, threading, time, pyautogui
import keyboard

print("Width =", GetSystemMetrics(0))
print("Height =", GetSystemMetrics(1))

halt_flag = False

# Vertex data for a full-screen quad
vertices = np.array([
    # Positions     # Texture Coords
    -1.0, -1.0, 0.0,  0.0, 1.0,  # Flip Y coordinate
     1.0, -1.0, 0.0,  1.0, 1.0,  # Flip Y coordinate
     1.0,  1.0, 0.0,  1.0, 0.0,  # Flip Y coordinate
    -1.0,  1.0, 0.0,  0.0, 0.0   # Flip Y coordinate
], dtype=np.float32)

indices = np.array([
    0, 1, 2,
    2, 3, 0
], dtype=np.uint32)

vertex_shader_code = """#version 330 core

layout (location = 0) in vec3 aPos;       // Vertex position
layout (location = 1) in vec2 aTexCoord;  // Texture coordinate

out vec2 TexCoord;  // Pass texture coordinates to fragment shader

void main()
{
    gl_Position = vec4(aPos, 1.0);  // Output vertex position
    TexCoord = aTexCoord;           // Pass texture coordinates to fragment shader
}
"""
fragment_shader_code = """#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D image;      // The input image texture
uniform vec2 texelSize;       // Size of one texel in texture coordinates (1.0 / texture size)

// Gaussian blur kernel
const float kernel[9] = float[](
    1.0 / 16, 2.0 / 16, 1.0 / 16,
    2.0 / 16, 4.0 / 16, 2.0 / 16,
    1.0 / 16, 2.0 / 16, 1.0 / 16
);

// Sobel kernel offsets for edge detection (horizontal and vertical)
const vec2 offsets[9] = vec2[](
    vec2(-1, -1), vec2(0, -1), vec2(1, -1),
    vec2(-1,  0), vec2(0,  0), vec2(1,  0),
    vec2(-1,  1), vec2(0,  1), vec2(1,  1)
);

// Sobel X and Y kernels for edge detection
const float sobelX[9] = float[](
    -1.0, 0.0, 1.0,
    -2.0, 0.0, 2.0,
    -1.0, 0.0, 1.0
);

const float sobelY[9] = float[](
    -1.0, -2.0, -1.0,
     0.0,  0.0,  0.0,
     1.0,  2.0,  1.0
);

void main()
{
    vec2 texCoord = TexCoord;
    
    // Apply Gaussian blur first
    vec4 blurredColor = vec4(0.0);
    for (int i = 0; i < 9; i++) {
        blurredColor += texture(image, texCoord + offsets[i] * texelSize) * kernel[i];
    }

    // Edge detection using Sobel filter on the blurred image
    float edgeX = 0.0;
    float edgeY = 0.0;
    for (int i = 0; i < 9; i++) {
        // Correctly use the 'image' texture, not the color value 'blurredColor'
        vec4 sample = texture(image, texCoord + offsets[i] * texelSize); 
        edgeX += sample.r * sobelX[i];
        edgeY += sample.r * sobelY[i];
    }
    
    float edge = sqrt(edgeX * edgeX + edgeY * edgeY);
    
    // Create a smoother binary mask for edges
    float edgeMask = smoothstep(0.01, 0.5, edge);  // Adjust thresholds as needed

    // Combine blurred and original image based on the smoother edge mask
    vec4 finalColor = mix(texture(image, texCoord), blurredColor, edgeMask);

    // Clamp values to [0, 1]
    finalColor.rgb = clamp(finalColor.rgb, 0.0, 1.0);
    
    // Swap red and blue channels
    float temp = finalColor.r;
    finalColor.r = finalColor.b;
    finalColor.b = temp;
    
    FragColor = finalColor;
}
"""
def load_shader(vertex_code, fragment_code):
    vertex_shader = compileShader(vertex_code, GL_VERTEX_SHADER)
    if not vertex_shader:
        raise RuntimeError("Vertex Shader failed to compile.")
    
    fragment_shader = compileShader(fragment_code, GL_FRAGMENT_SHADER)
    if not fragment_shader:
        raise RuntimeError("Fragment Shader failed to compile.")

    return compileProgram(vertex_shader, fragment_shader)

def setup_shaders():
    return load_shader(vertex_shader_code, fragment_shader_code)

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

def load_texture(frame):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Ensure the texture format and type are correct
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, frame)
    glGenerateMipmap(GL_TEXTURE_2D)
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture_id

def delete_texture(texture_id):
    glDeleteTextures([texture_id])

def render_frame(frame, shader_program, vao):
    texture_id = load_texture(frame)
    glUseProgram(shader_program)
    glBindVertexArray(vao)
    
    # Bind the texture
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glUniform1i(glGetUniformLocation(shader_program, "image"), 0)
    
    # Set uniforms
    glUniform1f(glGetUniformLocation(shader_program, "brightness"), 0.01)  # Example value
    glUniform1f(glGetUniformLocation(shader_program, "gamma"), 0.8)     # Example value
    
    # Render the quad
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)
    
    glBindVertexArray(0)
    glBindTexture(GL_TEXTURE_2D, 0)
    delete_texture(texture_id)  # Clean up the texture after use

def main():
    global video_mode
    if not glfw.init():
        return

    # Get the primary monitor and its video mode for fullscreen resolution
    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)

    # Set window hints to match the monitor's current video mode
    glfw.window_hint(glfw.RED_BITS, video_mode.red_bits)
    glfw.window_hint(glfw.GREEN_BITS, video_mode.green_bits)
    glfw.window_hint(glfw.BLUE_BITS, video_mode.blue_bits)
    glfw.window_hint(glfw.REFRESH_RATE, video_mode.refresh_rate)

    # Create a fullscreen window using the monitor and its current video mode
    window = glfw.create_window(video_mode.width, video_mode.height, "OpenGL Fullscreen Window", monitor, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    shader_program = setup_shaders()
    glUniform1f(glGetUniformLocation(shader_program, "sharpenStrength"), 0.5)  # Example value for sharpening strength
    glUniform2f(glGetUniformLocation(shader_program, "texelSize"), 1.0 / frame.shape[1], 1.0 / frame.shape[0])
    vao = setup_quad()
    
    while not glfw.window_should_close(window):
        # Get frame data (e.g., from video capture)
        frame = np.zeros((240, 180, 3), dtype=np.uint8)  # Placeholder for actual frame
        
        # Render the frame
        glClear(GL_COLOR_BUFFER_BIT)
        render_frame(frame, shader_program, vao)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


    
def keep_awake():
    global halt_flag
    while not halt_flag:
        pyautogui.press('left')
        time.sleep(2.5)
    print("done")

def display_psvita_stream(device_index):
    global halt_flag, video_mode
    time.sleep(1)
    cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open the selected device.")
        return

    if not glfw.init():
        return
    
    monitor = glfw.get_primary_monitor()
    video_mode = glfw.get_video_mode(monitor)

    # Disable window decorations and resizing for fullscreen
    glfw.window_hint(glfw.DECORATED, False)
    glfw.window_hint(glfw.RESIZABLE, False)

    window = glfw.create_window(GetSystemMetrics(0), GetSystemMetrics(1), "OpenGL Window", monitor, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    
    shader_program = setup_shaders()
    vao = setup_quad()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        # Render the frame
        glClear(GL_COLOR_BUFFER_BIT)
        render_frame(frame, shader_program, vao)
        
        glfw.swap_buffers(window)
        glfw.poll_events()
        
        if keyboard.is_pressed("escape"):
            cv2.destroyAllWindows()
            halt_flag = True
            glfw.terminate()
            break

    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()

def list_available_devices():
    available_devices = []
    for i in range(10):  # Checking for the first 10 devices; adjust as needed
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_devices.append(i)
                cap.release()
        except:
            pass
    return available_devices

if __name__ == "__main__":
    available_devices = list_available_devices()
    print("finished searching for devices")
    print(chr(27) + "[2J")
    if not available_devices:
        print("No video capture devices found please ensure the micro usb cable is inserted into the Ps Vita and try again.")
        time.sleep(3.5)
        quit()

    print("Available devices:")
    for idx, device in enumerate(available_devices):
        print(f"{idx + 1}: Device {device}")

    selected_device_index = int(input("Select a device by number: ")) - 1
    if selected_device_index < 0 or selected_device_index >= len(available_devices):
        print("Invalid selection.")
        time.sleep(1)
        quit()
    print("Starting Stream, Press ESC Key To Exit...")
    time.sleep(0.8)
    input("press enter to start: ")

    selected_device = available_devices[selected_device_index]

    thread = threading.Thread(target=keep_awake)
    thread.start()
    pyautogui.FAILSAFE = False

    while True:
        if halt_flag:
            thread.join()
            time.sleep(0.5)
            quit()
        display_psvita_stream(selected_device)
