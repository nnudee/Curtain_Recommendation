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
    cosine_scores = util.cos_sim(query_embedding, embeddings).flatten()  # คำนวณ cosine similarity
    top_k_indices = torch.topk(cosine_scores, 10).indices  # เลือก 10 อันดับแรก
    top_k_scores = cosine_scores[top_k_indices].tolist()  # เก็บคะแนนของ 10 อันดับ
    results = name_df.iloc[top_k_indices.tolist()].copy()  # ดึงข้อมูลม่าน
    results['Score'] = top_k_scores  # เพิ่มคอลัมน์คะแนน
    results = results.sort_values(by='Score', ascending=False)  # เรียงลำดับตามคะแนน
    return results

# สร้าง UI
st.title("ผ้าม่านที่เหมาะสมกับห้องของคุณ")
query = st.text_input("กรุณากรอกคำอธิบายลักษณะผ้าม่านที่คุณต้องการ")

if query:
    results = predict_curtain_type(query)
    for _, row in results.iterrows():
        st.write(f"### {row['Name']}")  # ชื่อผ้าม่าน
        st.write(f"**คำอธิบาย:** {row['Desc']}")  # คำอธิบายผ้าม่าน
        # ใช้เส้นทางที่ตรงกับโฟลเดอร์รูปภาพ
        image_path = f"PIC_Curtain/{row['images']}"
        try:
            st.image(Image.open(image_path), use_column_width=True)  # แสดงรูปภาพ
        except FileNotFoundError:
            st.write(f"ไม่พบรูปภาพ: {image_path}")
