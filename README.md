# Image-Similarity

## Steps to RUN
From the root folder, build the docker file 
`docker build -t image-similarity-service .`

To run the docker image, use the following command
`docker run -p 8000:8000 image-similarity-service`

## Description:

The code uses a VGG16 model to extract features from the last dense layer "fc2". 

Cosine Similarity is used to calculate similarity between the features of the new image and the features of images in the dataset. 
To reduce computation, <b>Approximate Nearest Neighbours</b> can also be implemented instead of comparing with the entire dataset. However at this size of the data set, latency was not observed. 



