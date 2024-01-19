from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
import time

import numpy as np
import scanning
import generator
import cv2

Builder.load_file('app.kv')


class CameraClick(BoxLayout):
    def capture(self):
        # camera = self.ids['camera']
        # texture = camera.texture
        # image = texture.pixels

        # image = np.frombuffer(image, dtype=np.uint8)
        # image = image.reshape(texture.height, texture.width, 4) # 4 for RGBA
        # image = image[::-1] # Reverse the image
        # image = image[::1] # Flip the image

        image = cv2.imread("test_1.jpg")
        
        scanned = scanning.scan(image)
        cv2.imwrite("scanned.jpg", scanned)
        print("Captured")


class TestCamera(App):

    def build(self):
        return CameraClick()


TestCamera().run()

###### 

regions = [generator.River(), generator.Forest(), generator.Road(), generator.Bush()] # Sample regions, holding the instances 

region_map = generator.generate_region_map() #TODO: Generate map

before = cv2.imread("blank.jpg")
region = generator.generate_region()

print(region.prompt())

correctly_generated = False

while not correctly_generated:

    current = None # #TODO Take picture
    current = scanning.scan(current) # scan picture
    diff = scanning.difference(before, current)

    try:
        pixels, _ =  scanning.check_region(diff, region)

        scanning.save_to_map(region_map, pixels, ) # This can also throw error, #TODO Test to see if another try except is needed

        correctly_generated = True

    except Exception as e:
        print(e)
        #TODO Prompt user to redraw region


before = current
