import cv2, numpy as np
import os
from PIL import Image
import pandas as pd

for root, directory, files in os.walk('bed_dataset/'):
        for file in files:
            filepath = root + os.sep + file
            print(filepath)
            img = cv2.imread(filepath)
            

        
