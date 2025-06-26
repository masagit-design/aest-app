import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import clip
import os

# モデル定義
class AestheticPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

# デバイス設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデルと前処理の読み込み（変更箇所）
model_clip, preprocess = clip.load("ViT-B/32", device=device)

# Aestheticスコア予測モデルの読み込み（変更箇所）
@st.cache_resource
def load_aesthetic_model():
    model = AestheticPredictor()
    model_path = "aesthetic_model/sac+logos+ava1-l14-linearMSE.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

model = load_aesthetic_model()

# Streamlit UI
st.title("Aesthetic Score Predictor")
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model_clip.encode_image(image_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = model(image_features).item()

    st.success(f"✨ Aesthetic Score: {score:.2f}")
