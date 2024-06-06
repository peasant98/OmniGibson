import json
import random

# Load the existing JSON file
with open('transforms.json', 'r') as file:
    data = json.load(file)

data["camera_angle_x"] = 1.1046714208162038
# Extract frames
frames = data["frames"]
# put the idx in the frame
for idx, frame in enumerate(frames):
    frame["idx"] = idx

# Shuffle frames to ensure randomness
random.shuffle(frames)

# Calculate the split index
split_index = int(0.8 * len(frames))

# Split the frames into training and testing sets
train_frames = frames[:split_index]
test_frames = frames[split_index:]

# Create new data structures for train and test
train_data = data.copy()
train_data["frames"] = train_frames

test_data = data.copy()
test_data["frames"] = test_frames

# Save the training data to transforms_train.json
with open('transforms_train.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

# Save the testing data to transforms_test.json
with open('transforms_test.json', 'w') as test_file:
    json.dump(test_data, test_file, indent=4)

print("Files created: transforms_train.json and transforms_test.json")
