import os
from ipynbcompress import compress

filename = '/Users/weitianyi/Desktop/ML-Notes/Coursera-ML-Notes.ipynb'
out = '/Users/weitianyi/Desktop/ML-Notes/compressed-Coursera-ML-Notes.ipynb'

compress(filename, output_filename=out, img_width=800, img_format='jpeg')
compress(filename)

print("done")