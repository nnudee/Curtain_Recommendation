import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from PIL import Image  # สำหรับแสดงรูปภาพ

# โหลดโมเดล
model = SentenceTransformer("nnudee/Curtain_Recommendation")

# โหลด embeddings
embeddings = torch.load('tensor+.pt')
if isinstance(embeddings, list):
    embeddings = torch.stack(embeddings)

# โหลดข้อมูลม่านจาก CSV
name_df = pd.read_csv("name-2.csv")

# ฟังก์ชันพยากรณ์
def predict_curtain_type(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)
    top_k_indices = torch.topk(cosine_scores.flatten(), 5).indices 
    return name_df.iloc[top_k_indices.tolist()] 

# สร้าง UI
st.title("ผ้าม่านที่เหมาะสมกับห้องของคุณ")
query = st.text_input("กรุณากรอกคำอธิบายของห้องของคุณ")

if query:
    results = predict_curtain_type(query)
    for _, row in results.iterrows():
        st.write(f"### {row['Name']}")  # 
        st.write(f"**คำอธิบาย:** {row['Desc']}")  
        image_path = f"images/{row['images']}"
        try:
            st.image(Image.open(image_path), use_column_width=True)
        except FileNotFoundError:
            st.write(f"ไม่พบรูปภาพ: {image_path}")
