import numpy as np
import cv2
from matplotlib import pyplot as plt

def to_grayscale(image):
    grayscale = np.mean(image, axis=2).astype(np.uint8)
    return grayscale

def thresholding(image, threshold_value):
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold_value] = 255
    return binary_image

def find_contours(binary_image):
    contours = []
    visited = np.zeros_like(binary_image)
    height, width = binary_image.shape

    def bfs(x, y):
        q = [(x, y)]
        contour = []
        while q:
            cx, cy = q.pop(0)
            if visited[cx, cy] == 0 and binary_image[cx, cy] == 255:
                visited[cx, cy] = 1
                contour.append((cx, cy))
                for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1),
                               (cx + 1, cy + 1), (cx - 1, cy - 1), (cx + 1, cy - 1), (cx - 1, cy + 1)]:
                    if 0 <= nx < height and 0 <= ny < width and visited[nx, ny] == 0 and binary_image[nx, ny] == 255:
                        q.append((nx, ny))
        return contour

    for x in range(height):
        for y in range(width):
            if binary_image[x, y] == 255 and visited[x, y] == 0:
                contour = bfs(x, y)
                contours.append(contour)
    
    return contours


def draw_contours(image, contours):
    for contour in contours:
        for (x, y) in contour:
            image[x, y] = [0, 255, 0]  
    return image

from skimage.feature import graycomatrix, graycoprops
def extract_features(img, gray, binary, contours):
    features = []
    if contours is None or len(contours) < 3:
        print("Invalid contour detected. Skipping...")
        return None 
    for i, contour in enumerate(contours):
        # Shape features
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        centroid = np.mean(contour, axis=0)[0]
        
        # Create a mask for this object
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        
        # Edge features
        edges = cv2.Canny(gray, 100, 200)
        object_edges = cv2.bitwise_and(edges, edges, mask=mask)
        edge_pixels = cv2.countNonZero(object_edges)
        
        # Texture features
        object_gray = cv2.bitwise_and(gray, gray, mask=mask)
        glcm = graycomatrix(object_gray, [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        
        # Color features (if color image)
        if len(img.shape) == 3:
            object_color = cv2.bitwise_and(img, img, mask=mask)
            color_hist = cv2.calcHist([object_color], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_hist = cv2.normalize(color_hist, color_hist).flatten()
        else:
            color_hist = None
        
        features.append({
            'id': i + 1,
            'area': area,
            'perimeter': perimeter,
            'bounding_box': (x, y, w, h),
            'centroid': tuple(centroid),
            'edge_pixels': edge_pixels,
            'contrast': contrast,
            'color_histogram': color_hist
        })
    
    return features

from sklearn.preprocessing import MinMaxScaler

# Normalize features
def normalize_features(features_list):
    # Convert features into a 2D array (each row is an object's feature vector)
    feature_matrix = np.array([
        [f['area'], f['perimeter'], f['bounding_box'][2] - f['bounding_box'][0],  # width
         f['bounding_box'][3] - f['bounding_box'][1],  # height
         f['texture_contrast'], f['texture_homogeneity']] for f in features_list
    ])
    
    # MinMax scaling (0-1 range normalization)
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    return normalized_features

from scipy.spatial.distance import euclidean

# Compare two objects using Euclidean distance on their feature vectors
def compare_objects(obj_features_1, obj_features_2):
    dist = euclidean(obj_features_1, obj_features_2)
    return dist

image = cv2.imread(r"C:\Users\91630\Downloads\images.jpeg")  
grayscale_image = to_grayscale(image)

threshold_value = 128
binary_image = thresholding(grayscale_image, threshold_value)
plt.figure(figsize=(10, 5))

# Show original image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Show grayscale image
plt.subplot(1, 3, 2)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale Image')

# Show binary (thresholded) image
plt.subplot(1, 3, 3)
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded Image')

plt.show()

binary_img = thresholding(image, 128)  # Apply thresholding
# Find contours
contours = find_contours(binary_image)

# Load original image in color for visualization
original_image = cv2.imread(r"C:\Users\91630\Downloads\images.jpeg")

# Draw contours on the original image
contours = draw_contours(original_image.copy(), contours)

# Display the original image with contours
plt.imshow(cv2.cvtColor(contours, cv2.COLOR_BGR2RGB))
plt.title('Image with Detected Contours')
plt.show()

features_list = []
for contour in contours:
    if contour.size >= 3:  # Ensure contour has enough points
        features = extract_features(original_image, grayscale_image, binary_image, contour)

# Example output of features for each object
for i, features in enumerate(features_list):
    print(f"Object {i+1} features: {features}")

labels = [0,1] 
normalized_features = normalize_features(features_list)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, labels, test_size=0.2, random_state=42)

# k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = knn.score(X_test, y_test)
print(f'Classification accuracy: {accuracy * 100:.2f}%')

# Function to draw the bounding box and class label on the image
def visualize_classification(image, contours, labels):
    for contour, label in zip(contours, labels):
        min_x, min_y, max_x, max_y = calculate_bounding_box(contour)
        cv2.rectangle(image, (min_y, min_x), (max_y, max_x), (0, 255, 0), 2)  # Bounding box
        cv2.putText(image, f"Class: {label}", (min_y, min_x - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Object Classification')
    plt.show()

visualize_classification(original_image, contours, y_pred)

from sklearn.cluster import KMeans

# Apply k-means clustering to group similar objects
kmeans = KMeans(n_clusters=2, random_state=42)  
kmeans.fit(normalized_features)

# Predict the cluster for each object
cluster_labels = kmeans.predict(normalized_features)

# Visualize clustering results
visualize_classification(original_image, contours, cluster_labels)

def visualize_labeled_objects(img, labeled_objects, ref_roi):
    result_img = img.copy()
    
    rx, ry, rw, rh = ref_roi
    cv2.rectangle(result_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 255), 2)
    cv2.putText(result_img, "Reference", (rx, ry - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    
    for obj in labeled_objects:
        x, y, w, h = obj['bounding_box']
        color = (0, 255, 0) if obj['label'] == 'Reference Object' else (255, 0, 0)
        cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_img, f"{obj['label']} ({obj['similarity']:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_img

def create_labeled_dataset(image_path, labeled_objects, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save labeled objects data as JSON
    json_path = os.path.join(output_dir, 'labeled_objects.json')
    with open(json_path, 'w') as f:
        json.dump(labeled_objects, f, indent=2)
    
    # Save image with labeled objects
    img = cv2.imread(image_path)
    for obj in labeled_objects:
        x, y, w, h = obj['bounding_box']
        color = (0, 255, 0) if obj['label'] == 'Reference Object' else (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, obj['label'], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    image_path = os.path.join(output_dir, 'labeled_image.jpg')
    cv2.imwrite(image_path, img)
    
    print(f"Labeled dataset created in {output_dir}")
    print(f"- JSON data: {json_path}")
    print(f"- Labeled image: {image_path}")

# Main execution
if __name__ == "__main__":
    image_path = r"C:\Users\91630\Downloads\images.jpeg"
    output_dir = r"D:\Amulya\Amulya VIT\Sem-7\I&V Analytics"
    threshold = 127  
    similarity_threshold = 0.8  
    
    try:
        # Label objects
        img, labeled_objects, ref_roi = label_objects(image_path, similarity_threshold, threshold)
        
        # Visualize results
        result_img = visualize_labeled_objects(img, labeled_objects, ref_roi)
        
        # Display results
        display_images({
            "Original Image": img,
            "Labeled Objects": result_img
        })
        
        # Create labeled dataset
        create_labeled_dataset(image_path, labeled_objects, output_dir)
        
    except Exception as e:
        print(f"An error occurred: {e}")

