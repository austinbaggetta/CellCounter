### Welcome to Cell Counter!

The goal of this jupyter notebook is to provide an easy and adaptable interface to aid in automatic cell detection. To create the conda environment to use cell counter, first navigate to the folder you either git cloned this repository into, or that you downloaded this repository into. Then run the following commands in either Anaconda terminal or Windows Powershell/Terminal:

```
conda env create -f environment.yml
```

As long as you are in the folder that you put Cell Counter into, this should create the environment you will need to use the necessary jupyter notebook (CellCounter.ipynb).

**Cell Counter is flexible!** If you would like to count cells in your entire image, you can do that. If you'd like to draw a region of interest and crop your image to just that area to count cells within, you can do that. If you'd like to draw multiple fields of view to count cells within, you can also do that. You can load previously-saved regions of interest by providing a path to them, and you can load previously-saved preprocessing parameters by providing a path to them. Please feel free to open any issues or function requests and I will address them as quickly as possible! Happy counting!