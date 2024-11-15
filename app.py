import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd

# โหลดโมเดลที่ได้ fine-tune
model = SentenceTransformer("nnudee/Curtain_Recommendation")

# โหลด embeddings จากไฟล์ .pt
embeddings = torch.load('tensor+.pt')

# หาก embeddings เป็น list ของ Tensor และคุณต้องการรวมเป็น Tensor เดียว
if isinstance(embeddings, list):
    embeddings = torch.stack(embeddings)

# โหลดข้อมูล `name` จากไฟล์ CSV
name_df = pd.read_csv("Query+.csv")  # ระบุชื่อไฟล์ CSV ของคุณ
name = name_df["Name"].tolist()  # สร้าง list ของชื่อผ้าม่าน

# ฟังก์ชันสำหรับรับ input จากผู้ใช้
def predict_curtain_type(query):
    # สร้าง embedding สำหรับ query
    query_embedding = model.encode(query, convert_to_tensor=True)
    # คำนวณ cosine similarity
    cosine_scores = util.cos_sim(query_embedding, embeddings)
    # ดึง index ของผลลัพธ์ที่มี similarity สูงสุด
    top_k_indices = torch.topk(cosine_scores.flatten(), 1).indices
    # ส่งคืนชื่อผ้าม่านที่เหมาะสมที่สุด
    return name[top_k_indices[0]]

# สร้าง UI ด้วย Streamlit
st.title("ผ้าม่านที่เหมาะสมกับความต้องการของคุณ")
query = st.text_input("กรุณากรอกคำอธิบายของผ้าม่านที่คุณต้องการ")

if query:
    result = predict_curtain_type(query)
    st.write(f"ผ้าม่านที่เหมาะสมคือ: {result}")

