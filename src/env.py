import gym
import dxcam
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import ctypes
import cv2
import torchvision.transforms as transforms
import torch
from torch.utils.tensorboard import SummaryWriter
import base64
from urllib.parse import unquote
import json
from pathlib import Path

# Initialize the TensorBoard writer
writer = SummaryWriter('runs/slitherio_experiment')

# Path to the directory containing the unpacked NTL mod extension
extension_path = r"C:\Users\Joe\OneDrive\Documents\Programming\Chrome Extensions\ntlmod-pub"

# Path to directory containing the settings file
settings_path = "slither-settings-aibot.json"


def extract_settings(file_path):
    # Get the file extension
    file_extension = Path(file_path).suffix.lower()

    # Read and process .ntlmod file
    if file_extension == '.ntlmod':
        with open(file_path, 'rb') as file:
            content = file.read()
        decoded_content = base64.b64decode(content)
        json_content = unquote(decoded_content.decode('utf-8'))
        settings_dict = json.loads(json_content)

    # Read and process .json file
    elif file_extension == '.json':
        with open(file_path, 'r') as file:
            settings_dict = json.load(file)

    else:
        raise ValueError("Unsupported file extension. Only .ntlmod and .json files are supported.")

    return settings_dict


def preprocess_image(image, desired_height=256):
    # Calculate the proportional width
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height
    new_width = int(desired_height * aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, desired_height))

    # Convert to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    return grayscale_image

def middle_click(browser, element):
    browser.execute_script("""
        var event = new MouseEvent('mousedown', {
            'bubbles': true,
            'cancelable': true,
            'button': 1
        });
        arguments[0].dispatchEvent(event);
    """, element)

class SlitherIOEnv(gym.Env):
    def __init__(self):
        # Create DXCamera instance
        self.camera = dxcam.create()

        # Launch Chrome + NTL mod extension using Selenium
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_argument(f'load-extension={extension_path}')
        self.browser = webdriver.Chrome(options=chrome_options)

        # initialise ui elements that will be used
        self.play_button = None
        self.score_element = None

        # Reset the environment (also navigates to slither.io and starts the game)
        self.reset()

        # Maximize the window and make it full screen
        self.browser.maximize_window()
        self.browser.fullscreen_window()

        # Initialise browser window coordinates to None
        self.left, self.top, self.right, self.bottom = self.get_window_coordinates()

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=np.array([-180, 0, 0]), high=np.array([180, 100, 1]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.bottom - self.top, self.right - self.left, 3),
                                                dtype=np.uint8)

    def hide_extra_elements(self):
        # Define the JavaScript code to hide elements
        js_code = '''
        var elements = document.querySelectorAll('.nsi:not(canvas):not(#mybox):not(#mmap):not(#mmap>*):not(#fps-hud), #bchat, #eemenu');
        for (var i = 0; i < elements.length; i++) {
            elements[i].style.display = 'none';
        }
        '''
        # Execute the JavaScript code
        self.browser.execute_script(js_code)

    def reset(self):
        self.browser.get('http://slither.io')
        self.apply_settings(extract_settings(settings_path))
        # Wait for the play button to appear on the page, up to 10 seconds
        self.play_button = WebDriverWait(self.browser, 5).until(
            EC.element_to_be_clickable((By.ID, "connect-btn"))
        )
        self.play_button.click()
        self.score_element = WebDriverWait(self.browser, 5).until(
            EC.presence_of_element_located((By.ID, "_mylength"))
        )
        # Send 'k' key press twice to the webpage
        body = self.browser.find_element_by_tag_name('body')
        body.send_keys('k')
        body.send_keys('k')
        self.hide_extra_elements()
        middle_click(self.browser, body)

    def apply_settings(self, settings):
        # Convert the settings dictionary to a JSON string
        settings_json = json.dumps(settings)

        # JavaScript code to apply settings without Ua function
        js_code = f"""
        var settings = {settings_json};
        for (var key in settings) {{
            localStorage.setItem(key, settings[key]);
        }}
        location.reload(true);
        """

        # Execute the JavaScript code
        self.browser.execute_script(js_code)

    def get_window_coordinates(self):
        window_rect = self.browser.get_window_rect()
        sf = 1.5
        return (
            int(window_rect['x'] * sf),
            int(window_rect['y'] * sf),
            int((window_rect['x'] + window_rect['width']) * sf),
            int((window_rect['y'] + window_rect['height']) * sf)
        )

    def perform_action(self, action):
        angle, offset, click = action
        x_offset = offset * np.cos(np.deg2rad(angle))
        y_offset = offset * np.sin(np.deg2rad(angle))
        x = (self.right + self.left) // 2 + x_offset
        y = (self.bottom + self.top) // 2 + y_offset
        ctypes.windll.user32.SetCursorPos(int(x), int(y))

        if click > 0.5:
            ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
            ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)

    def grab_observation(self):
        # get browser window coordinates
        self.left, self.top, self.right, self.bottom = self.get_window_coordinates()
        # Try to grab the frame, up to 10 attempts
        observation = None
        attempts = 0
        while observation is None and attempts < 10:
            observation = self.camera.grab(region=(self.left, self.top, self.right, self.bottom))
            attempts += 1

        if observation is None:
            raise RuntimeError("Failed to grab observation after 10 attempts")

        # Preprocess the observation
        processed_observation = preprocess_image(observation)

        return processed_observation

    def step(self, action):
        self.perform_action(action)
        observation = self.grab_observation()

        # Extract the score from the game by finding the element with id "_mylength"
        try:
            reward = int(self.score_element.text)
        except:
            # Handle any exceptions that may occur (e.g., if the element is not found or the text cannot be converted to an int)
            reward = 0

        done = False
        return observation, reward, done, {}

    def render(self, mode='human', step_count=0):
        observation = self.grab_observation()

        if mode == 'human':
            cv2.imshow('Slither.io', observation)
            cv2.waitKey(1)

        elif mode == 'tensorboard':
            observation_tensor = transforms.ToTensor()(observation)
            writer.add_image('Slither.io Observation', observation_tensor, step_count)

    def close(self):
        self.browser.quit()
        self.camera.release()
