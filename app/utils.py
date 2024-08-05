import numpy as np
from PIL import Image
import os
import requests
import pandas as pd
from io import BytesIO
import base64

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50,preprocess_input

# from keras.applications import EfficientNetB1


# Defining Folder Paths
cwd = os.getcwd()
root_dir = os.path.split(cwd)[0]
model_dir = os.path.join(root_dir, "Models")
inter_dir = os.path.join(root_dir, "Intermediaries")
image_dir = os.path.join(root_dir, "Images")

# Constants
img_size_model = (224, 224)

def create_folder(folder_name, root=False):
    """ Create folder if there is not
    Args:
        folder_name: String, folder name
    Returns:
        None
    """
    if root:
        folder_path = os.path.join(root_dir, folder_name)
    else:
        folder_path = os.path.join(model_dir, folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created")

def download_images(df):

    for i, row in df.iterrows():
        response = requests.get(row['0'])

        if response.status_code != 200:
            print("Failed to download image!")
            exit()

        filename = str(i)+".png" # You can name the file as you want
        file_path = os.path.join(image_dir, filename)
        with open(file_path, 'wb') as file:
            file.write(response.content)

def load_model(model_name, include_top=True):
    """ Load pre-trained Keras model
    Args:
        model_name: String, name of model to load
        include_top: String, the model is buildt with 'feature learning block' + 'classification block'
    Returns:
        model: Keras model instance
    """
    
    if model_name == 'vgg16':
        model = VGG16(weights='imagenet', include_top=include_top)
    else:
        print("Wrong Model Name")
    
    print(f">> '{model.name}' model successfully loaded!")
    
    return model

def image_processing(img_array):
    """ Preprocess image to be used in a keras model instance
    Args:
        img_array: Numpy array of an image which will be predicte
    Returns:
        processed_img = Numpy array which represents the processed image
    """
    # Expand the shape
    img = np.expand_dims(img_array, axis=0)

    # Convert image from RGB to BGR (each color channel is zero-centered with respect to the ImageNet dataset, without scaling)
    processed_img = preprocess_input(img)

    return processed_img


def get_feature_vector(model, img):
    """ Get a feature vector extraction from an image by using a keras model instance
    Args:
        model: Keras model instance used to do the classification.
        img_path: String to the image path which will be predicted
    Returns:
        feature_vect: List of visual feature from the input image
    """

    # Creation of a new keras model instance without the last layer
    layername_feature_extraction = 'predictions'
    model_feature_vect = Model(inputs=model.input, outputs=model.get_layer(layername_feature_extraction).output)

    # Image processing
    
    if type(img)==str:
        img = image.load_img(img, target_size=img_size_model)
    img_arr = np.array(img)
    img_ = image_processing(img_arr)

    # Visual feature extraction
    feature_vect = model_feature_vect.predict(img_)

    return feature_vect





from scipy import spatial

def calculate_similarity(vector1, vector2):
    """Compute similarities between two images using 'cosine similarities'
    Args:
        vector1: Numpy vector to represent feature extracted vector from image 1
        vector2: Numpy vector to represent feature extracted vector from image 1
    Returns:
        sim_cos: Float to describe the similarity between both images
    """
    sim_cos = 1-spatial.distance.cosine(vector1, vector2)

    return sim_cos


def load_image_features(folder_path, model):
    df = pd.DataFrame(columns=['file_name', 'vector'])
    if os.path.exists(folder_path):
        for i, file in enumerate(os.listdir(folder_path)):
            # if i<1801:
            #     continue
            try:
                file_path = os.path.join(folder_path, file)
                feature_vec = get_feature_vector(model, file_path)
                df.loc[len(df), :] = [file, feature_vec]
                if i%300 == 0:
                    feature_file_name = os.path.join(inter_dir, model.name + "_features_" +str(i) +".pkl")

                    df.to_pickle(feature_file_name)
                    df = pd.DataFrame(columns=['file_name', 'vector'])

            except:
                print("Error at file: ", file)
    else:
        print(f"Folder {folder_path} does not exists, NO IMAGES FOUND")

# load_image_features(image_dir, model)

def compare_image(model, x, img_feature_vec):
    x = np.array(x)
    cosine_similarity = calculate_similarity(x[0], img_feature_vec[0])
    return cosine_similarity   

def encode_image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_images(top_k_file_names):
    images = []
    for file in top_k_file_names:
        file_path = os.path.join(image_dir, file)
        img = image.load_img(img, target_size=img_size_model)
        img_arr = np.array(img)
        images.append(encode_image_to_base64(img_arr))
    return images


def find_top_k_similar(model, input_image, k:int = 10):
    img_feature_vec = get_feature_vector(model, input_image)
    df_similarity_list = []
    for file in os.listdir(inter_dir):
        df_similarity = pd.DataFrame()
        df = pd.read_pickle(os.path.join(inter_dir, file))
        df_similarity['file_name'] = df['file_name']
        df_similarity['score'] = df['vector'].apply(lambda x: compare_image(model, x, img_feature_vec))
        df_similarity_list.append(df_similarity)
    df_similarity = pd.concat(df_similarity_list)
    df_similarity = df_similarity.sort_values(by=['score'], ascending=False)
    return df_similarity.iloc[:k, 0]
