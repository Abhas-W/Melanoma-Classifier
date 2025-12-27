import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

## dataset:
## https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images?resource=download

# one-hot vectors
# [1,0] = benign
# [0,1] = malignant

img_size = 50 # 50x50 pixels

# locations of img files 
benign_training_folder = "train/benign/"
malignant_training_folder = "train/malignant/"

benign_testing_folder = "test/benign/"
malignant_testing_folder = "test/malignant/"

# Function to load image data 

def img_data(folder_path, label, img_size):
    data = []

    for filename in os.listdir(folder_path):
        try:
            path = os.path.join(folder_path, filename)

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size,img_size))
            img_array = np.array(img)
            data.append([img_array, np.array(label)])

        except:
            pass
    return np.array(data, dtype="object")

# Load image data

benign_training_data = img_data(
    benign_training_folder,
    label=[1, 0],
    img_size=img_size
)

malignant_training_data = img_data(
    malignant_training_folder,
    label=[0, 1],
    img_size=img_size
)

benign_testing_data = img_data(
    benign_testing_folder,
    label=[1, 0],
    img_size=img_size
)

malignant_testing_data = img_data(
    malignant_testing_folder,
    label=[0, 1],
    img_size=img_size
)

# Balancing training data
benign_training_data = benign_training_data[0:len(malignant_training_data)]

print()
print()
print(f"Benign training count: {len(benign_training_data)}")
print(f"Malignant training count: {len(malignant_training_data)}")
print()
print(f"Benign testing count: {len(benign_testing_data)}")
print(f"Malignant testing count: {len(malignant_testing_data)}")


# Combine and shuffle training and testing data
#training_data = benign_training_data + malignant_training_data
training_data = np.concatenate((benign_training_data, malignant_training_data), axis=0)
np.random.shuffle(training_data)
np.save("melanoma_training_data.npy", training_data)


testing_data = np.concatenate((benign_testing_data,malignant_testing_data), axis=0)
np.random.shuffle(testing_data)
np.save("melanoma_testing_data.npy", testing_data)

# print(training_data)
# print()
# print(malignant_training_data)

# print(benign_testing_data)
# print(malignant_testing_data)
# print(testing_data)

# for row in training_data:
#     print(row[0])
#     print(row[1])
#     print()
#     input()
