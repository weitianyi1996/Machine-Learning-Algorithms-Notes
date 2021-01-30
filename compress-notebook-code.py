# this file is used to compress large jupyter notebook with images into small size file

import os
from ipynbcompress import compress


# specify input and output jupyter notebook location
filename = '/Users/weitianyi/Desktop/ML-Notes/Coursera-ML-Notes.ipynb'
out = '/Users/weitianyi/Desktop/ML-Notes/compressed-Coursera-ML-Notes.ipynb'

compress(filename, output_filename=out, img_width=800, img_format='jpeg')
compress(filename)

print("done")