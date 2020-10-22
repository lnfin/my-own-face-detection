#https://www.kaggle.com/andrewmvd/face-mask-detection

import xml.etree.ElementTree as ET
import os
import shutil

for file in os.listdir("annotations"):
    tree = ET.parse(f"annotations/{file}")
    root = tree.getroot()
    for el in root.findall("filename"):
        filename = el.text
        out = filename[12:]
        if "with_mask" in open("annotations/" + file).read():
            shutil.copyfile("images/" + filename, "with_mask/" + str(out))
        else:
            shutil.copyfile("images/" + filename, "without_mask/" + str(out))
