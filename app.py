from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the trained model
model = load_model("model/my_model.h5")

# Define class labels (update based on your model's classes)
CLASS_LABELS =  [
    "Apple Apple scab",
    "Apple Black rot",
    "Apple Cedar apple rust",
    "Apple healthy",
    "Background without leaves",
    "Blueberry healthy",
    "Cherry Powdery mildew",
    "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot",
    "Corn Common rust",
    "Corn Northern Leaf Blight",
    "Corn healthy",
    "Grape Black rot",
    "Grape Esca (Black Measles)",
    "Grape Leaf blight (Isariopsis Leaf Spot)",
    "Grape healthy",
    "Orange Haunglongbing (Citrus greening)",
    "Peach Bacterial spot",
    "Peach healthy",
    "Pepper bell Bacterial spot",
    "Pepper bell healthy",
    "Potato Early blight",
    "Potato Late blight",
    "Potato healthy",
    "Raspberry healthy",
    "Soybean healthy",
    "Squash Powdery mildew",
    "Strawberry Leaf scorch",
    "Strawberry healthy",
    "Tomato Bacterial spot",
    "Tomato Late blight",
    "Tomato Early blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot",
    "Tomato Tomato Yellow Leaf Curl Virus",
    "Tomato Tomato mosaic virus",
    "Tomato healthy"
]

# Function to preprocess the image
def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((256, 256))  # Resize gambar sesuai input model
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension
    print(f"Processed image shape: {img_array.shape}")  # Debugging
    return img_array

@app.get("/health")
async def health_check():
    """Endpoint untuk mengecek apakah API berjalan dengan benar."""
    return JSONResponse(content={"status": "API is running", "message": "Health check passed!"})

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Validasi jenis file
#         if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
#             return JSONResponse(content={"error": "File must be an image (PNG, JPG, JPEG)"}, status_code=400)

#         # Read and preprocess image
#         image_data = await file.read()
#         img = Image.open(BytesIO(image_data))
#         processed_image = preprocess_image(img)

#         # Perform prediction
#         predictions = model.predict(processed_image)
#         predicted_class = CLASS_LABELS[np.argmax(predictions)]
#         confidence = float(np.max(predictions))

#         # Return prediction as JSON
#         return JSONResponse(content={
#             "predicted_class": predicted_class,
#             "confidence": confidence
#         })
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Validasi jenis file
        if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
            return JSONResponse(content={"error": "File must be an image (PNG, JPG, JPEG)"}, status_code=400)

        # Read and preprocess image
        image_data = await file.read()
        img = Image.open(BytesIO(image_data))
        processed_image = preprocess_image(img)

        # Perform prediction
        predictions = model.predict(processed_image)
        predicted_index = np.argmax(predictions)  # Index kelas dengan probabilitas tertinggi
        predicted_class = CLASS_LABELS[predicted_index]  # Ambil nama kelas dari CLASS_LABELS
        confidence = float(np.max(predictions))  # Probabilitas tertinggi

        # Return prediction as JSON
        return JSONResponse(content={
            "predicted_class": predicted_class,  # Nama penyakit
            "confidence": confidence  # Probabilitasnya
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
