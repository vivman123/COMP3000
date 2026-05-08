import os, io, base64, torch, cv2, numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from flask import Flask, request, jsonify, render_template
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
device = torch.device('cpu') # force cpu only

def load_galaxy_model(path):
    # use resnet50
    model = models.resnet50()
    # add dropout and change the final layer to match the 3 classes
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 3))
    
    # load weights from pth
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"✔️ Loaded weights from {path}")
    else:
        print(f" {path} not found check the path name") 
        
    return model.eval()

# mapping classes based on the training script
model = load_galaxy_model('galaxy_model100.pth')
class_names = ['Spiral', 'Elliptical', 'Irregular']

# copy the training crop exactly as it was used in the training script
transform = transforms.Compose([
    transforms.Resize(424),      
    transforms.CenterCrop(200), 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# use the final convalutional layer for the gradcam
target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    results = []
    
    # handle batch uploads
    for file in request.files.getlist('files'):
        try:
            
            img_raw = Image.open(file).convert('RGB')
            input_tensor = transform(img_raw).unsqueeze(0)

            with torch.no_grad():
                output = model(input_tensor)
                print(f"DEBUG - Raw Scores: {output[0].tolist()}") # useful for sanity checks
                probs = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred_idx = torch.max(probs, 0)

            label = class_names[pred_idx.item()]
            
            # generate the heatmap
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_idx.item())])[0, :]
            
            # blur it
            grayscale_cam = cv2.GaussianBlur(grayscale_cam, (11, 11), 0)
            
            img_np = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
            img_np = np.clip(np.array([0.229, 0.224, 0.225]) * img_np + np.array([0.485, 0.456, 0.406]), 0, 1)

            # overlay the heatmap
            vis = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, image_weight=0.7)
            vis = np.ascontiguousarray((vis * 255).astype(np.uint8))

            # draw a box around the highest activation area if it's confident enough
            if np.max(grayscale_cam) > 0.2:
                mask = (grayscale_cam > 0.5).astype(np.uint8)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
                    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # encode back to base64 so frontend can render it directly
            buf = io.BytesIO()
            Image.fromarray(vis).save(buf, format="JPEG")
            
            results.append({
                'prediction': label,
                'confidence': f"{conf.item() * 100:.2f}%",
                'image': base64.b64encode(buf.getvalue()).decode('utf-8')
            })
        except Exception as e:
            results.append({'error': str(e)})
            
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True, port=5010)