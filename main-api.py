from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = load_model('model.h5')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = load_img(io.BytesIO(contents), color_mode='grayscale', target_size=(48, 48))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0

    prediction = model.predict(image)
    predicted_class = (prediction > 0.5).astype("int32")[0][0]  # Adjusted for binary classification

    class_names = {0: 'happy', 1: 'unhappy'}

    return JSONResponse(content={'class': class_names[predicted_class]})
