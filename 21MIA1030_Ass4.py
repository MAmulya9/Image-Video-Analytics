import cv2
import numpy as np
import os

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
   
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
   
    cap.release()
    return frames

def convert_to_hsv(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

def noise_detection(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    noise_mask = cv2.absdiff(frame, blurred)
    return noise_mask

def histogram_comparison(frame1, frame2):
    hist1 = cv2.calcHist([frame1], [0, 1], None, [256, 256], [0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1], None, [256, 256], [0, 256, 0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
   
   
    similarity_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity_score

def sobel_edge_detection(frame):
    sobel_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(edges)

def find_least_similarity_cuts(similarity_scores, frames, top_n=5):
    # Get the indices of the least similarity scores (lowest values indicate scene cuts)
    least_similar_indices = np.argsort(similarity_scores)[:top_n]
    
    print(f"Top {top_n} scene cut frames (by least similarity):")
    for i, idx in enumerate(least_similar_indices):
        print(f"Scene cut {i+1}: between frame {idx} and frame {idx+1} with similarity score: {similarity_scores[idx]}")

        # Display the two consecutive frames where the scene cut occurs
        cv2.imshow(f'Scene Cut {i+1}: Frame {idx}', frames[idx])
        cv2.imshow(f'Scene Cut {i+1}: Frame {idx+1}', frames[idx+1])
        cv2.waitKey(1000)  # Wait for a key press to display next set of frames
        
    return least_similar_indices

def process_video(video_path, hsv_directory, noise_directory, edge_directory, top_n_cuts=5):
    if not os.path.exists(hsv_directory):
        os.makedirs(hsv_directory)
    if not os.path.exists(noise_directory):
         os.makedirs(noise_directory)
    if not os.path.exists(edge_directory):
         os.makedirs(edge_directory)
   
    frames = load_video(video_path)
    hsv_frames = []
    noise_frames = []
    edge_frames = []
    similarity_scores = []
   
    for idx, frame in enumerate(frames):
        # Step 2: Convert each frame into HSV
        hsv_frame = convert_to_hsv(frame)
        hsv_frames.append(hsv_frame)
        cv2.imwrite(os.path.join(hsv_directory, f'hsv_frame_{idx}.png'), hsv_frame)

        # Step 3: Detect noise
        noise_frame = noise_detection(frame)
        noise_frames.append(noise_frame)
        cv2.imwrite(os.path.join(noise_directory, f'noise_frame_{idx}.png'), noise_frame)

        # Step 4: Perform histogram comparison with the previous frame
        if idx > 0:
            similarity = histogram_comparison(frames[idx-1], frames[idx])
            similarity_scores.append(similarity)
            print(f'Similarity between frame {idx-1} and frame {idx}: {similarity}')

        # Step 5: Perform edge detection
        edge_frame = sobel_edge_detection(frame)
        edge_frames.append(edge_frame)
        cv2.imwrite(os.path.join(edge_directory, f'edge_frame_{idx}.png'), edge_frame)

    # Step 6: Find least similarity cuts and display the frames
    if len(similarity_scores) > 0:
        least_similar_indices = find_least_similarity_cuts(similarity_scores, frames, top_n_cuts)
        
        # Optionally, save the frames at the scene cut points for analysis
        for i, idx in enumerate(least_similar_indices):
            cv2.imwrite(os.path.join(hsv_directory, f'scene_cut_frame_{i+1}_frame_{idx}.png'), frames[idx])
            cv2.imwrite(os.path.join(hsv_directory, f'scene_cut_frame_{i+1}_frame_{idx+1}.png'), frames[idx+1])


video_path = r"C:\Users\91630\Downloads\Frnds.mp4"
hsv_directory = r"C:\Users\91630\Downloads\hsv_frames"
noise_directory = r"C:\Users\91630\Downloads\noise_frames"
edge_directory = r"C:\Users\91630\Downloads\edge_frames"
process_video(video_path, hsv_directory, noise_directory, edge_directory, top_n_cuts=5)
