from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from utils import create_folder, load_model, find_top_k_similar, encode_image_to_base64, get_images
from PIL import Image
from keras.preprocessing import image
import io


app = FastAPI()

create_folder("Images", True)
create_folder("Intermediaries", True)
create_folder("model", True)

available_models = ['vgg16', 'resnet50']
selected_model = available_models[0]

model = load_model(selected_model, include_top=True)
model.trainable=False


# @app.post("/similar-products")
# def similar_products(file: UploadFile):
#     try:
#         image_bytes = file.read()
#         input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#         top_k_file_names = find_top_k_similar(model, input_image, 10).to_list()
#         base64_images = get_images(top_k_file_names)
        
#         response = {"similar_products": base64_images}
#         return JSONResponse(content=response)
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=400)
