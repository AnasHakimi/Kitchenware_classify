import requests
import os

# Use an existing image from the dataset or the root folder
# I saw myspoon1.png in the root MATLAB folder earlier
image_path = '../myspoon1.png'
url = 'http://localhost:8000/predict'

if not os.path.exists(image_path):
    print(f"Error: Test image not found at {image_path}")
    exit(1)

try:
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)
        
    if response.status_code == 200:
        print("Success! Response:")
        print(response.json())
    else:
        print(f"Failed with status code {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"An error occurred: {e}")
