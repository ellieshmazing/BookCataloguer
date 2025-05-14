This folder contains all of the code and images for the spine segmentation portion of our assignment.

/Images/Raw/ contains the two example input images.
/Images/Full/ contains all intermediate and output images for the full shelf image.
/Images/SingleShelf/ contains all intermediate and output images for the single shelf image.

/src/ contains the code for book segmentation.
/src/bookSegmentation.py is the main file for accomplishing book segmentation. It takes two inputs: a path to a source image and a path to a destination folder. To run the pipeline, use the format "python bookSegmentation.py [input image path] [output folder path]".

bookSegmentation.py calls functions in bookSeeding.py and bookSuperpixelization.py.
bookSeeding.py holds all functions related to detecting and culling text regions to serve as origin points for region-growing.
bookSuperpixelization.py holds all functions related to generating the Superpixel image, and then using that information for extracting each book.

preprocessing.py did not end up being used in our pipeline, but I spent a lot of time on it and felt bad deleting it lol.