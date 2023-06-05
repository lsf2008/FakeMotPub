This is a PyTorch-lightning/GPU implementation of the paper 
Video anomaly detection based on a motion pseudo-sample synthesizer.

# Requirements
To run the program, please install the following packages: 
- pytorch==1.11.0
- pytorch-lightning==1.9.0
- pytorchvideo==0.1.5
- opencv-python==4.7.0.68
- PyYAML==6.0
- prettytable==3.6.0
- matplotlib==3.7.0
- torchvision==0.12.0


# How to run it
1. To start, prepare the data using a CSV file. Since the video is stored with images in a single folder, provide the folder name in the CSV file followed by the ground truth for each frame.
2. Next, configure the YAML file using the data CSV file, and adjust other parameters such as batch size, learning rate, weight decay, and block size, etc.
3. Finally, specify the "flg" in the [tst.py](http://tst.py/) file and run it.

# Acknowledgement
In our model implementation, we refer to the model implementation of the article "Latent Space Autoregression 
for Novelty Detection", and we thank for the implementation of AE model.

