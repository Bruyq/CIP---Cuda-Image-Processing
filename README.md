# CIP - CUDA for image processing

## Purpose of this repository
This project is a way for me to learn GPU programming using CUDA in C++.  
It is also a way for me to display image processing knowledge I have by integrating it using CUDA.  
The goal is to add some basic image processing tools, as well as some more complex tools inspired by research papers.  


## How to use this repository/demo
### Code
You can either reuse my code for your own goals freely or run it on your device to see what I made and how it works.  
> [!IMPORTANT]
> This project uses **win32 API** windows for the demo. Therefore, you might not be able to run the demo on other operating systems.
> This project was coded using a **Nvidia GTX970 4GB graphic card** of **compute capability 5.2** and **CUDA Version 12.4**.  

> [!NOTE]
> This is important as you need an Nvidia card to make this program work and if your CUDA version or compute capability is older than what is mentionned, it may not work.
> Also, keep in mind that some functionalities now exist in CUDA that didn't under compute capability 5.2, which explains some choices I made in my CUDA kernels.  

### Demo
By executing this project, you should go through the following steps :  
 1. A command window appears and welcomes you, then you will be asked with a folder selection window to select a folder where your results will be saved.
 2. Then, you have to select an image to process in a file selection window. (Please select RGB images. There is an issue with 8 bit images).
 3. The command window will display available options to process your image. You have to enter the number corresponding to an option to select it.
 4. You will then have additional options selection if the selected image processing method requires it.
 5. You will have to enter a name corresponding to the result image filename. This file will be saved in the folder you previously selected.
 6. Finally, the command window will ask you if you want to keep using the demo (yes[y] / no[n]). By entering "n" the program finishes but if you select "y" the program will return to the second step.

| Step 1 | Step 2 | Step 3 | Step 4 | Step 5 | Step 6 |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  ![Step 1](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/f1a7fff8-3dfa-4d4c-9292-14d3ffeb687d) | ![Step 2](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/1d76c6a1-ed01-487c-8335-dc6fa138cb17) | ![Step 3](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/4003bd15-240e-4bfa-ae10-f1685a54a2da) | ![Step 4](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/90866a27-90c2-49f2-8d27-d277943eb1f6) | ![Step 5](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/081a7a85-0c61-46c4-83c4-5d675997e462) | ![Step 6](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/65f6681c-bbba-47ec-9f63-fc7e6654a0ea) |

### Results example
| Initial image | Guided filter smoothing |
| :---: | :---: |
|![flower](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/30df5b4c-9c4c-45c2-a5f1-43ff51b9b398)|![guidedsmoothing](https://github.com/Bruyq/CIP---Cuda-Image-Processing/assets/162017415/a589942c-4b3a-44b2-9ed1-fafdb0b960cd)|

## TODO
- [ ] Implement histogram related CUDA kernels and functions
- [ ] Add better error handling to have a more complete demo (Unexpected behavior with command windows, kernels too long to execute returning error 701 when the GPU isn't powerful enough)
- [ ] Improve the quality of life and the form of the demo (Render the current image on screen, display options in win32 API windows, etc.)
- [ ] Search for cool traditional computer vision methods that can be implemented using CUDA and implement them


## Known issues
- Kernels too long to execute (choosen parameters leading to too high complexity, working image with a too high resolution) will provoke CUDA error 701 as the GPU is too long to respond and the CPU will think that it is stuck in a loop.
- stb library doesn't work properly when images have a very very low resolution and doesn't work either with 1 channel images (8 bit). I don't know if it comes from me using the library in an unexpected way or if it is a known issue.


## Sources
- Guided Filter : https://kaiminghe.github.io/publications/pami12guidedfilter.pdf  
He, K.; Sun, J.; Tang, X. Guided Image Filtering. IEEE Transactions on Pattern Analysis and MachineIntelligence 2013,35, 1397–1409




