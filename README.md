# Image-Similarity

## Steps to RUN
From the root folder, build the docker file

```docker build -t image-similarity-service .```

To run the docker image, use the following command

```docker run -p 8000:8000 image-similarity-service```

## Description

This project uses a VGG16 model to extract features from the last dense layer, "fc2." The features are then used to calculate the similarity between a new image and the images in the dataset using cosine similarity. 

To optimize computation, **Approximate Nearest Neighbours** (ANN) could be implemented instead of comparing against the entire dataset. However, given the current dataset size, no significant latency was observed, so ANN has not been implemented.

### Testing

You can test the service using the following command:

```curl -X POST "http://localhost:8000/similar-products" -F "file=@E:\work\Image-Similarity\app\Images\8.png"```



The response returns a JSON object with two keys:

1) images: A list of images in numpy array format.

2) indices: The indices of up to the top 10 similar products from the `app/Images` folder, filtered by a minimum similarity score of 0.6 to avoid irrelevant matches.
Future Enhancements

To further enhance performance, alternative models such as EfficientNetV2M or EfficientNetB1 could be considered. However, these models have been observed to have a feature extraction cost of approximately 15 times higher than VGG16.

The Dataset with all the feature vectors is split into 4 pickle files to reduce RAM utilization. 

If you have any other questions, please feel free to contact me at preet.modi00@gmail.com


