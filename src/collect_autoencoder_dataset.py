import time
import os
from PIL import Image
from env import SlitherIOEnv
from selenium.common.exceptions import NoSuchWindowException
import keyboard

# Create environment
env = SlitherIOEnv()

# Directory to save the frames
output_dir = 'data/encoder_dataset'
os.makedirs(output_dir, exist_ok=True)

# Wait for page to load properly and for full-screen message to leave
time.sleep(3)

try:
    # Loop to keep observing the environment and save the frames
    while True:
        observation = env.grab_observation() # Get current frame from environment
        timestamp = time.time() # Timestamp for unique filename
        image = Image.fromarray(observation) # Convert to PIL image
        image_path = os.path.join(output_dir, f'{timestamp}.png') # Path to save the image
        image.save(image_path) # Save the image
        print(f'Saved frame to {image_path}')
        time.sleep(1) # Wait for a second

except (KeyboardInterrupt, NoSuchWindowException):
    print('Stopped collecting frames.')

finally:
    env.close()
