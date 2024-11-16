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
name_df = pd.read_csv("names.csv")

# ฟังก์ชันพยากรณ์
def predict_curtain_type(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)
    top_k_indices = torch.topk(cosine_scores.flatten(), 5).indices  # 5 ชนิด
    return name_df.iloc[top_k_indices.tolist()]  # คืนผลลัพธ์เป็น DataFrame

# สร้าง UI
st.title("ผ้าม่านที่เหมาะสมกับห้องของคุณ")
query = st.text_input("กรุณากรอกคำอธิบายของห้องของคุณ")

if query:
    results = predict_curtain_type(query)
    for _, row in results.iterrows():
        st.write(f"**ชื่อผ้าม่าน:** {row['Name']}")
        image_path = f"images/{row['Image']}"  # สร้างเส้นทางรูปภาพ
        st.image(Image.open(image_path), use_column_width=True)  # แสดงรูปภาพ
