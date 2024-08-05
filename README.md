# Image-Similarity

## Steps to RUN
From the root folder, build the docker file 
```docker build -t image-similarity-service .```

To run the docker image, use the following command
```docker run -p 8000:8000 image-similarity-service```

## Description:

The code uses a VGG16 model to extract features from the last dense layer "fc2". 

Cosine Similarity is used to calculate similarity between the features of the new image and the features of images in the dataset. 
To reduce computation, <b>Approximate Nearest Neighbours</b> can also be implemented instead of comparing with the entire dataset. However at this size of the data set, latency was not observed. 

Testing was done using the following command 
```curl -X POST "http://localhost:8000/similar-products" -F "file=@E:\work\Image-Similarity\app\Images\8.png"```

Currently it returns json object with 2 keys. 1) list of images in np array format, and 3) the indices of up to top 10 similar products (min similarity score of 0.6 to avoid unwanted matches)

Alternative models such as EfficientNet2M or EfficientNetB1 could be used to improve performance, however, the cost of obtaining features was observed to be approximately 15x the cost for VGG16. 

The Dataset with all the feature vector is split into 4 pickle files to reduce RAM utlization. 

If you have any other questions, please feel free to contact me at preet.modi00@gmail.com


