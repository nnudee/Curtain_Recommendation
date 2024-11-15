import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# โหลดโมเดลที่ได้ fine-tune
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nnudee/Curtain_Recommendation")

# โหลด embeddings จากไฟล์ .pt
embeddings = torch.load('tensor+.pt')

# หาก embeddings เป็น list ของ Tensor และคุณต้องการรวมเป็น Tensor เดียว
if isinstance(embeddings, list):
    embeddings = torch.stack(embeddings)

# ใช้ unsqueeze() กับ Tensor ที่โหลดมา
query_embedding = embeddings.unsqueeze(0)  # เพิ่มมิติให้กับ Tensor ถ้าจำเป็น

# ฟังก์ชันสำหรับรับ input จากผู้ใช้
def predict_curtain_type(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings)
    top_k_indices = torch.topk(cosine_scores.flatten(), 1).indices
    return name[top_k_indices[0]]

# สร้าง UI
st.title("ผ้าม่านที่เหมาะสมกับห้องของคุณ")
query = st.text_input("กรุณากรอกคำอธิบายของห้องของคุณ")

if query:
    result = predict_curtain_type(query)
    st.write(f"ผ้าม่านที่เหมาะสมคือ: {result}")
