from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from keras.preprocessing import image
import io
import numpy as np
from PIL import Image
import os
import requests
import pandas as pd
import json
from scipy import spatial


from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.preprocessing import image

# from keras.applications.resnet50 import ResNet50,preprocess_input

# from keras.applications import EfficientNetB1


# Defining Folder Paths
root_dir = os.getcwd()
print("Content of CWD is ", os.listdir(root_dir))
app_dir = os.path.join(root_dir, "app")
# root_dir = os.path.split(cwd)[0]
inter_dir = os.path.join(app_dir, "Intermediaries")
image_dir = os.path.join(app_dir, "Images")

# Constants
img_size_model = (224, 224)



# Helper Functions
def create_folder(folder_name, root=False):
    """ Create folder if there is not
    Args:
        folder_name: String, folder name
    Returns:
        None
    """
    folder_path = os.path.join(app_dir, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder {folder_path} created")

def download_images(file_path):
    """ One time function to Download the images from the df
    Args:
        file_path: String, File path
    Returns:
        None
    """
    df = pd.read_csv(file_path)
    for i, row in df.iterrows():
        response = requests.get(row['0'])

        if response.status_code != 200:
            print("Failed to download image!")
            exit()

        filename = str(i)+".png" # You can name the file as you want
        file_path = os.path.join(image_dir, filename)
        with open(file_path, 'wb') as file:
            file.write(response.content)

def load_model(include_top=True):
    """ Load pre-trained VGG16 model
    Args:
        include_top: String, the model is buildt with 'feature learning block' + 'classification block'
    Returns:
        model: Keras model instance
    """
    

    model_path = os.path.join(os.path.join(app_dir, "Models"), "vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    
    # Check if local file available
    if os.path.exists(model_path):
        model = VGG16(weights=None, include_top=include_top)
        model.load_weights(model_path)
    else:
        model = VGG16(weights='imagenet', include_top=include_top)
    
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
    layername_feature_extraction = 'fc2'
    model_feature_vect = Model(inputs=model.input, outputs=model.get_layer(layername_feature_extraction).output)

    # Image processing
    
    if type(img)==str:
        img = image.load_img(img, target_size=img_size_model)
    else:
        img = img.resize(img_size_model, Image.Resampling.LANCZOS)
    img_arr = np.array(img)
    img_ = image_processing(img_arr)

    # Visual feature extraction
    feature_vect = model_feature_vect.predict(img_)

    return feature_vect




def calculate_similarity(vector1, vector2):
    """Compute similarities between two images using 'cosine similarities'
    Args:
        vector1: Numpy vector to represent feature extracted vector from image 1
        vector2: Numpy vector to represent feature extracted vector from image 1
    Returns:
        sim_cos: Float to describe the similarity between both images
    """
    vector1 = np.array(vector1)
    sim_cos = 1-spatial.distance.cosine(vector1[0], vector2[0])

    return sim_cos


def load_image_features(folder_path:str, model):
    """One time loads the features of the images in the dataset
    Args:
        folder_path: Path of folder which contains the Data
        model: Keras model instance used to obtain the features
    Returns:
        None
    """
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


def get_images(top_k_file_names):
    """ Load the top K images from the dataset 
    Args:
        top_k_file_names: List, Top K file names
    Returns:
        Jsonified object of the images sorted in order of similarity.
    """
    images = []
    for file in top_k_file_names:
        file_path = os.path.join(image_dir, file)
        img = Image.open(file_path)
        img_arr = np.array(img)
        images.append(img_arr.tolist())
    return json.dumps(images)


def find_top_k_similar(model, input_image, k:int = 10):
    """ Compare the dataset to identify the top k similar
    Args:
        model: String, folder name
        input_image: New image that is to be checked
        k: Number of top matches to be find
    Returns:
        List, file names of the top K matches 
    """
    print(f"Finding upto Top {k} Similar Matches")
    img_feature_vec = get_feature_vector(model, input_image)
    df_similarity_list = []
    # RATHER THAN COMPARING ALL APPROXIMATE NEAREST NEIGHBOURS CAN BE IMPLEMENTED
    for file in os.listdir(inter_dir):
        df_similarity = pd.DataFrame()
        df = pd.read_pickle(os.path.join(inter_dir, file))
        df_similarity['file_name'] = df['file_name']
        df_similarity['score'] = df['vector'].apply(lambda x: calculate_similarity(x, img_feature_vec))
        df_similarity_list.append(df_similarity)
    df_similarity = pd.concat(df_similarity_list)

    # COULD BE OPTIMIZED TO TAKE MAX 10 times
    df_similarity = df_similarity.sort_values(by=['score'], ascending=False)
    # Considering only the matches above 0.6 similarity score
    df_similarity = df_similarity[df_similarity['score']>0.6]
    return df_similarity.iloc[:k, 0]



app = FastAPI()

create_folder("Images", True)
create_folder("Intermediaries", True)

model = load_model(include_top=True)
model.trainable=False


@app.post("/similar-products")
async def similar_products(file: UploadFile):
    try:
        image_bytes = await file.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        top_k_file_names = find_top_k_similar(model, input_image, 10).to_list()
        images_arr = get_images(top_k_file_names)
        
        response = {"similar_products": images_arr,
                    "indices": top_k_file_names}
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


