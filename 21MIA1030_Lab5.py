import cv2

# Load the video
video_path = r"C:\Users\91630\Downloads\74-135732548_small.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit(1)

# Get the frame rate of the video for timestamp calculations
fps = cap.get(cv2.CAP_PROP_FPS)

# Parameters for motion detection
threshold_value = 20  # Adjust based on sensitivity needed
min_area = 300  # Minimum area to be considered as motion (adjustable)

# Read the first frame
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# To store event timestamps
events = []

while cap.isOpened():
    # Read the next frame
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)

    # Apply a binary threshold to the difference image
    _, thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)

    # Display the thresholded image for debugging
    cv2.imshow('Frame Difference', thresh)

    # Find contours of the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a flag to mark motion
    motion_detected = False

    # Loop through the contours to identify significant motion
    for contour in contours:
        # Ignore small contours based on area
        if cv2.contourArea(contour) < min_area:
            continue

        # Draw bounding box around significant motion
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Mark that motion is detected
        motion_detected = True

    # If significant motion is detected, store the event
    if motion_detected:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Convert milliseconds to seconds
        events.append(timestamp)
        print(f"Motion detected at {timestamp:.2f}s")
        cv2.putText(curr_frame, f"Event detected at {timestamp:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display the frame with motion detection
    cv2.imshow('Motion Detection', curr_frame)

    # Use a longer wait time if frames disappear too quickly
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    # Update the previous frame to the current one for the next iteration
    prev_frame_gray = curr_frame_gray

# Release resources
cap.release()
cv2.destroyAllWindows()

# Re-open video to save the output
cap = cv2.VideoCapture(video_path)
output_path = r"C:\Users\91630\Downloads\output_with_events.mp4"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Reprocess to save the annotated frames
ret, prev_frame = cap.read()
prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    # Convert the current frame to grayscale
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between current and previous frame
    frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)
    _, thresh = cv2.threshold(frame_diff, threshold_value, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(curr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    if motion_detected:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        cv2.putText(curr_frame, f"Event detected at {timestamp:.2f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    out.write(curr_frame)
    prev_frame_gray = curr_frame_gray

# Release resources
cap.release()
out.release()
print(f"Events detected at times: {events}")



# Load the image
image_path = r"C:\Users\91630\Downloads\crowd.jpg"  # Replace with the path of the uploaded image
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Define the emotions list for labeling
emotions = ["Happy", "Sad", "Thinking", "Surprised", "Angry", "Excited"]

# Function to detect emotions based on facial features
def detect_emotion(face_region):
    height, width = face_region.shape

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(face_region, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

    # Detect mouth within the lower half of the face region
    mouth_region = face_region[int(height * 0.5):, :]
    mouths = mouth_cascade.detectMultiScale(mouth_region, scaleFactor=1.5, minNeighbors=15, minSize=(30, 30))

    # Initialize variables to store emotion features
    mouth_detected = len(mouths) > 0
    eye_count = len(eyes)

    # Analyze mouth aspect ratio for better smile detection
    smile_threshold = 0.5  # Adjust this value to make smile detection more sensitive
    if mouth_detected:
        for (mx, my, mw, mh) in mouths:
            mouth_aspect_ratio = mh / mw  # Ratio of mouth height to width
            if mouth_aspect_ratio < smile_threshold:
                return "Happy"
            elif mouth_aspect_ratio > smile_threshold:
                return "Excited"  # New case for excitement

    # Use conditions to classify emotions based on detected features
    if not mouth_detected and eye_count >= 2:
        if eye_count == 2:
            return "Thinking"  # Renamed from "Neutral"
        elif eye_count > 2:
            return "Surprised"  # Case for wide-open eyes (more eye regions detected)
    elif not mouth_detected and eye_count < 2:
        return "Sad"
    elif eye_count >= 2 and not mouth_detected:
        return "Angry"  # New case for angry (wide-open eyes with no smile)
    else:
        return "Thinking"  # Default to thinking

# Store detected emotions for each face
detected_emotions = []

# Loop through each detected face
for (x, y, w, h) in faces:
    # Draw a rectangle around the face
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Extract the face region for further analysis
    face_region = gray[y:y+h, x:x+w]
    
    # Detect the emotion using the custom function
    emotion = detect_emotion(face_region)
    
    # Store the emotion and draw text
    detected_emotions.append(emotion)
    cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Calculate overall sentiment
overall_sentiment = max(set(detected_emotions), key=detected_emotions.count)

# Display overall sentiment
cv2.putText(image, f"Crowd Sentiment: {overall_sentiment}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# Save and display the result image
output_path = r"C:\Users\91630\Downloads\crowd_sentiment.jpg"
cv2.imwrite(output_path, image)
cv2.imshow("Emotion Analysis", image)
cv2.waitKey(1000)
cv2.destroyAllWindows()

print(f"Individual emotions detected: {detected_emotions}")
print(f"Overall crowd sentiment: {overall_sentiment}")



import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern
import zipfile

import zipfile

# Path to the zip file
zip_file_path = r"C:\Users\91630\Downloads\Face dataset1.zip"

# Path to the folder where you want to extract the files
extracted_folder = r"C:\Users\91630\Downloads\extracted folder"

# Create the extraction folder if it doesn't exist
os.makedirs(extracted_folder, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder)

# List the contents of the extracted folder to verify
print("Files extracted to:", extracted_folder)
for root, dirs, files in os.walk(extracted_folder):
    for file in files:
        print(os.path.join(root, file))

subfolder_path = os.path.join(extracted_folder, 'Face dataset1')
os.listdir(subfolder_path)

haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# Helper function to detect and crop faces with adjusted parameters
def detect_and_crop_face(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:  # If no frontal face detected, try profile
        faces = profile_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30))
    
    # If face is detected, crop the face
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face, faces
    return None, faces

# Helper function to extract geometric features (distances between landmarks)
def extract_geometric_features(face_image):
    h, w, _ = face_image.shape
    
    # Hypothetical positions of key features (to be replaced with actual landmark detection in a full implementation)
    eye_left = (int(0.3 * w), int(0.4 * h))
    eye_right = (int(0.7 * w), int(0.4 * h))
    nose = (int(0.5 * w), int(0.55 * h))
    mouth = (int(0.5 * w), int(0.75 * h))
    
    # Calculate geometric distances
    eye_distance = np.linalg.norm(np.array(eye_left) - np.array(eye_right))
    nose_to_mouth = np.linalg.norm(np.array(nose) - np.array(mouth))
    face_width = w
    face_height = h
    
    # Add new features: width to height ratio
    width_to_height_ratio = face_width / face_height
    
    return {
        'eye_distance': eye_distance,
        'nose_to_mouth': nose_to_mouth,
        'width_to_height_ratio': width_to_height_ratio
    }

# Helper function to extract texture features using Local Binary Patterns (LBP)
def extract_texture_features(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_face, P=8, R=1, method='uniform')
    
    # Calculate histogram of LBP as a texture descriptor
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 11),
                             range=(0, 10))
    # Normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    
    # Add variance of the LBP histogram to capture texture smoothness
    lbp_variance = np.var(hist)
    
    return hist, lbp_variance

# Helper function to classify gender based on geometric and texture features
def classify_gender(geometric_features, texture_features, lbp_variance):
    eye_distance = geometric_features['eye_distance']
    nose_to_mouth = geometric_features['nose_to_mouth']
    width_to_height_ratio = geometric_features['width_to_height_ratio']
    
    # Define a threshold for the eye distance to nose-to-mouth ratio
    eye_to_mouth_ratio = eye_distance / (nose_to_mouth + 1e-6)
    
    # Base classification using geometric ratios
    if eye_to_mouth_ratio > 1.2:
        gender = 'Male'
    else:
        gender = 'Female'
    
    # Adjust based on face shape (width-to-height ratio)
    if width_to_height_ratio < 0.85:
        gender = 'Female'
    elif width_to_height_ratio > 1.15:
        gender = 'Male'
    
    # Refine based on texture analysis:
    smooth_texture_score = np.sum(texture_features[:3])  # Sum up smooth texture patterns
    
    # Adjust thresholds and use variance for refinement:
    if smooth_texture_score > 0.45 and lbp_variance < 0.025:
        # Smoother texture tends to indicate female, but adjust threshold for smoother male faces
        gender = 'Female'
    elif smooth_texture_score < 0.4 or lbp_variance > 0.03:
        # More variance or rougher texture tends to indicate male
        gender = 'Male'
    
    return gender


sample_image_path = os.path.join(subfolder_path, '063489.jpg.jpg')

# Detect and crop the face from the sample image
cropped_face, face_rect = detect_and_crop_face(sample_image_path)

# Display the original image with face detection and the cropped face (if detected)
if cropped_face is not None:
    # Show original image with face rectangle
    img = cv2.imread(sample_image_path)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Face')
    
    # Show cropped face
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Face')
    
    plt.show()
else:
    print("No face detected in the sample image.")
# Extract features from the cropped face (if face was detected)
if cropped_face is not None:
    geometric_features = extract_geometric_features(cropped_face)
    texture_features, lbp_variance = extract_texture_features(cropped_face)

    # Classify gender using the extracted features
    predicted_gender = classify_gender(geometric_features, texture_features, lbp_variance)

    # Display the result
    print(f"Predicted Gender: {predicted_gender}")
else:
    print("No face detected to classify.")

