import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from PIL import Image  

model = SentenceTransformer("nnudee/Curtain_Recommendation")

embeddings = torch.load('tensor+.pt')
if isinstance(embeddings, list):
    embeddings = torch.stack(embeddings)

name_df = pd.read_csv("name-2.csv")

def predict_curtain_type(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embedding, embeddings).flatten() 
    top_k_indices = torch.topk(cosine_scores, 10).indices  
    top_k_scores = cosine_scores[top_k_indices].tolist()  
    results = name_df.iloc[top_k_indices.tolist()].copy()  
    results['Score'] = top_k_scores 
    results = results.sort_values(by='Score', ascending=False)  
    return results

# สร้าง UI
st.title("ผ้าม่านที่เหมาะสมกับห้องของคุณ")
query = st.text_input("กรุณากรอกคำอธิบายลักษณะผ้าม่านที่คุณต้องการ")

if query:
    results = predict_curtain_type(query)
    for _, row in results.iterrows():
        st.write(f"### {row['Name']}")  
        st.write(f"**คำอธิบาย:** {row['Desc']}") 
            
        image_path = f"PIC_Curtain/{row['images']}"
        try:
            st.image(Image.open(image_path), use_column_width=True)  
        except FileNotFoundError:
            st.write(f"ไม่พบรูปภาพ: {image_path}")
