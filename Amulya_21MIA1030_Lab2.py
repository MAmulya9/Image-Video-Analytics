#Lab Task 1: Setup and Basic Extraction
#Objective:
#Install the necessary tools and libraries, and extract frame information from a video.
#Steps:
#1. Install ffmpeg and ffmpeg-python:
#  Install the ffmpeg tool and the ffmpeg-python library.
#2. Extract Frame Information:
#  Extract frame information from a sample video.

import ffmpeg

def extract_frame_info(video_path):
    try:
        # Probe the video to get detailed information
        probe = ffmpeg.probe(video_path)

        # Extract video stream information
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        if video_stream is None:
            print('No video stream found in the file.')
        else:
            width = video_stream['width']
            height = video_stream['height']
            codec = video_stream['codec_name']
            frame_rate = eval(video_stream['r_frame_rate'])
            duration = float(video_stream['duration'])

            print(f'Video Width: {width}')
            print(f'Video Height: {height}')
            print(f'Codec: {codec}')
            print(f'Frame Rate: {frame_rate}')
            print(f'Duration: {duration} seconds')

    except ffmpeg.Error as e:
        print(f'Error: {e.stderr.decode()}')

if __name__ == "__main__":
    video_path = "C:\\Users\\91630\\Downloads\\v_SkyDiving_g01_c01.mp4"
    extract_frame_info(video_path)

#------------------------------------------------------------------------------------------------------------

#Lab Task 2: Frame Type Analysis
#Objective:
#Analyze the extracted frame information to understand the distribution of I, P, and B frames in a video.
#Steps:
#1. Modify the Script:
#  Count the number of I, P, and B frames.
#  Calculate the percentage of each frame type in the video.
#2. Analyze Frame Distribution:
#  Plot the distribution of frame types using a library like matplotlib.
#  Plot a pie chart or bar graph showing the distribution of frame types using matplotlib.

import matplotlib.pyplot as plt

def extract_frame_info(video_path):
    try:
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='frame=pkt_pts_time,pict_type', format='json')
        frames = probe['frames']
        frame_info = [(frame.get('pkt_pts_time', 'N/A'), frame.get('pict_type', 'N/A')) for frame in frames if 'pict_type' in frame]
        return frame_info
    except ffmpeg.Error as e:
        print(f"Error: {e}")
        return []

def analyze_frame_distribution(frame_info):
    frame_counts = {'I': 0, 'P': 0, 'B': 0}
    for _, frame_type in frame_info:
        if frame_type in frame_counts:
            frame_counts[frame_type] += 1
    
    total_frames = sum(frame_counts.values())
    frame_percentages = {frame_type: (count / total_frames) * 100 for frame_type, count in frame_counts.items()}

    return frame_counts, frame_percentages

def plot_frame_distribution(frame_counts, frame_percentages):
    labels = frame_counts.keys()
    counts = frame_counts.values()
    percentages = [f'{frame_percentages[frame_type]:.2f}%' for frame_type in labels]

    # Plotting bar graph
    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts, color=['blue', 'orange', 'green'])
    plt.xlabel('Frame Type')
    plt.ylabel('Count')
    plt.title('Distribution of Frame Types')
    for i, (count, percentage) in enumerate(zip(counts, percentages)):
        plt.text(i, count + 0.5, percentage, ha='center', va='bottom')
    plt.show()

    # Plotting pie chart
    plt.figure(figsize=(10, 5))
    plt.pie(counts, labels=labels, autopct='%1.2f%%', colors=['blue', 'orange', 'green'])
    plt.title('Percentage Distribution of Frame Types')
    plt.show()


video_path = "C:\\Users\\91630\\Downloads\\v_SkyDiving_g01_c01.mp4"
frame_info = extract_frame_info(video_path)
frame_counts, frame_percentages = analyze_frame_distribution(frame_info)

print(f"Frame counts: {frame_counts}")
print(f"Frame percentages: {frame_percentages}")

plot_frame_distribution(frame_counts, frame_percentages)

#--------------------------------------------------------------------------------------------------

#Lab Task 3: Visualizing Frames
#Objective:
#Extract actual frames from the video and display them using Python.
#Steps:
#1. Extract Frames:
#  Use ffmpeg to extract individual I, P, and B frames from the video.
#  Save these frames as image files.
#2. Display Frames:
#  Use a library like PIL (Pillow) or opencv-python to display the extracted frames.
#   1. Save I, P, and B frames as separate image files using ffmpeg.
#   2. Use PIL or opencv-python to load and display these frames in a Python script.
#   3. Compare the visual quality of I, P, and B frames.

import os
from PIL import Image

# Function to extract specific frame types (I, P, B) and save them as images
def extract_frames_by_type(video_path, output_folder, frame_type):
    os.makedirs(output_folder, exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(f'{output_folder}/frame_%04d.png', vf=f'select=eq(pict_type\\,{frame_type})', vsync='vfr')
        .run()
    )
    print(f"{frame_type} frames extracted and saved to {output_folder}")

# Function to display frames using PIL (Pillow)
def display_frames(folder_path):
    image_files = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.png')]
    for image_file in image_files:
        img = Image.open(image_file)
        img.show()

# Path to your video file
video_path = "C:\\Users\\91630\\Downloads\\v_SkyDiving_g01_c01.mp4"

# Output directories for I, P, and B frames
output_folders = {
    'I': "C:/Users/91630/Downloads/I_frames",
    'P': "C:/Users/91630/Downloads/P_frames",
    'B': "C:/Users/91630/Downloads/B_frames"
}

# Extract and display I, P, and B frames
for frame_type, folder in output_folders.items():
    extract_frames_by_type(video_path, folder, frame_type)
    display_frames(folder)
    
import cv2
from skimage.metrics import structural_similarity as ssim

def load_image(frame_path):
    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image at {frame_path} could not be found.")
    return img

def calculate_ssim(image1, image2):
    return ssim(image1, image2)

def compare_frames(i_frame_path, p_frame_path, b_frame_path):
    try:
        
        i_frame = load_image(i_frame_path)
        p_frame = load_image(p_frame_path)
        b_frame = load_image(b_frame_path)
        
        
        ssim_i_p = calculate_ssim(i_frame, p_frame)
        ssim_i_b = calculate_ssim(i_frame, b_frame)
        ssim_p_b = calculate_ssim(p_frame, b_frame)
        
        print(f"SSIM between I-frame and P-frame: {ssim_i_p:.4f}")
        print(f"SSIM between I-frame and B-frame: {ssim_i_b:.4f}")
        print(f"SSIM between P-frame and B-frame: {ssim_p_b:.4f}")
        
    except FileNotFoundError as e:
        print(e)

# Define file paths for comparison
i_frame_path = "C:\\Users\\91630\\Downloads\\I_frames\\frame_0001.png"
p_frame_path = "C:\\Users\\91630\\Downloads\\P_frames\\frame_0035.png"
b_frame_path = "C:\\Users\\91630\\Downloads\\B_frames\\frame_0001.png"

# Compare frames
compare_frames(i_frame_path,p_frame_path,b_frame_path)
#-------------------------------------------------------------------------------------------------------------------------

#Lab Task 4: Frame Compression Analysis
#Objective:
#Analyze the compression efficiency of I, P, and B frames.
#Steps:
#1. Calculate Frame Sizes:
#   Calculate the file sizes of extracted I, P, and B frames.
#   Compare the average file sizes of each frame type.
#2. Compression Efficiency:
#   Discuss the role of each frame type in video compression.
#   Analyze why P and B frames are generally smaller than I frames.

def calculate_file_size(file_path):
    return os.path.getsize(file_path)

# Function to calculate the average size of frames in a folder
def calculate_average_frame_size(folder_path):
    frame_sizes = []
    for frame_file in os.listdir(folder_path):
        if frame_file.endswith('.png'):  # Assuming frames are saved as .png
            frame_path = os.path.join(folder_path, frame_file)
            frame_size = calculate_file_size(frame_path)
            frame_sizes.append(frame_size)
    average_size = sum(frame_sizes) / len(frame_sizes) if frame_sizes else 0
    return average_size, frame_sizes

# Paths to the extracted frames (replace these with your paths)
i_frames = "C:/Users/91630/Downloads/I_frames"
p_frames = "C:/Users/91630/Downloads/P_frames"
b_frames= "C:/Users/91630/Downloads/B_frames"

# Calculate average frame sizes
i_avg_size, i_frame_sizes = calculate_average_frame_size(i_frames)
p_avg_size, p_frame_sizes = calculate_average_frame_size(p_frames)
b_avg_size, b_frame_sizes = calculate_average_frame_size(b_frames)

# Print results
print(f"Average I-Frame Size: {i_avg_size / 1024:.2f} KB")
print(f"Average P-Frame Size: {p_avg_size / 1024:.2f} KB")
print(f"Average P-Frame Size: {b_avg_size / 1024:.2f} KB")

#------------------------------------------------------------------------------------------

#Lab Task 5: Advanced Frame Extraction
#Objective:
#Extract frames from a video and reconstruct a part of the video using only I frames.
#Steps:
#1. Extract and Save I Frames:
#  Extract I frames from the video and save them as separate image files.
#. Reconstruct Video:
#  Use the extracted I frames to reconstruct a portion of the video.
#  Create a new video using these I frames with a reduced frame rate


def extract_i_frames(video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Probe to get the list of frames and their types
        probe = ffmpeg.probe(video_path, v='error', select_streams='v:0', show_entries='frame=pict_type', format='json')
        frames = probe['frames']
        
        # Collect indices of I frames
        i_frame_indices = [i for i, frame in enumerate(frames) if frame.get('pict_type') == 'I']
        
        # Extract and save each I frame
        for index in i_frame_indices:
            frame_filename = os.path.join(output_dir, f"I_frame_{index:04d}.jpg")
            (
                ffmpeg
                .input(video_path)
                .filter('select', f'eq(n,{index})')  # Use raw string or escape comma
                .output(frame_filename, vframes=1)
                .run(capture_stdout=True, capture_stderr=True)
            )
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr.decode()}")


video_path = "C:\\Users\\91630\\Downloads\\Road.mp4" 
output_dir = 'C:\\Users\\91630\\Downloads\\extracted_i_frames'

# Extract and save I frames
extract_i_frames(video_path, output_dir)

import subprocess

def reconstruct_video_from_i_frames(i_frames_folder, output_video_path, frame_rate=30):
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError("FFmpeg is not installed or not found in the system path.")
    if not os.path.isdir(i_frames_folder):
        raise FileNotFoundError(f"The directory {i_frames_folder} does not exist.")
    i_frames = [f for f in os.listdir(i_frames_folder) if os.path.isfile(os.path.join(i_frames_folder, f))]
    if not i_frames:
        raise FileNotFoundError(f"No files found in the directory {i_frames_folder}.")
    i_frames.sort()
    with open('frames_list.txt', 'w') as file:
        for frame in i_frames:
            file.write(f"file '{os.path.join(i_frames_folder, frame)}'\n")
    command = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'frames_list.txt',
        '-framerate', str(frame_rate),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]

    subprocess.run(command, check=True)
    os.remove('frames_list.txt')
i_frames_folder = "C:\\Users\\91630\\Downloads\\extracted_i_frames"
output_video_path = 'C:\\Users\\91630\\Downloads\\output_reconstructed_video.mp4'

reconstruct_video_from_i_frames(i_frames_folder,output_video_path,frame_rate=15)











