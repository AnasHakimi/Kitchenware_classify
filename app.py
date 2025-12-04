from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
from torchvision import transforms
from PIL import Image
import io
from model import KitchenwareCNN
import os

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Load Model
MODEL_PATH = "kitchenware_model.pth"
CLASS_NAMES_PATH = "class_names.txt"

class_names = []
if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
else:
    # Fallback if not trained yet
    class_names = ['Class1', 'Class2', 'Class3', 'Class4']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KitchenwareCNN(num_classes=len(class_names))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
else:
    print("Warning: Model file not found. Predictions will be random.")

# Transform for inference
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
])

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Get top predictions
        top_probs, top_idxs = torch.topk(probabilities, len(class_names))
        
        results = []
        for i in range(len(class_names)):
            results.append({
                "class": class_names[top_idxs[0][i].item()],
                "probability": top_probs[0][i].item()
            })
            
        return JSONResponse(content={"predictions": results})
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
