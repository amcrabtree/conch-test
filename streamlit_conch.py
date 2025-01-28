import streamlit as st
from io import StringIO
import os
from PIL import Image
import pandas as pd
import torch

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

st.set_page_config(page_title="CONCH Matcher")

# Load model
model_path = "/Users/amc/tutorials/CONCH/checkpoints/conch/pytorch_model.bin"
model, preprocess = create_model_from_pretrained(
    model_cfg='conch_ViT-B-16', checkpoint_path=model_path)
model = model.eval()
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)

# File input in sidebar
with st.sidebar:
    st.header("File Input")
    seq_file_extensions = ["jpg", "jpeg", "png"]
    patch_file = st.file_uploader(
        "##### Image patch file:", type=seq_file_extensions)
    search_text = st.text_area(
        "Enter search terms, one phrase per line:", 
        "tumor\nstroma\nimmune cells\nred blood cells\nductal carcinoma") 

# Match text to image
if patch_file and search_text:

    # Load image
    image = Image.open(patch_file)
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    st.image(image.resize((224, 224)))

    # Load search terms list
    my_search_list = search_text.split("\n")
    tokenizer = get_tokenizer()
    tokenized_prompts = tokenize(texts=my_search_list, tokenizer=tokenizer).to(device)

    # Run model
    with torch.inference_mode():
        image_embeddings = model.encode_image(image_tensor)
        text_embeddings = model.encode_text(tokenized_prompts)
        sim_scores = (image_embeddings @ text_embeddings.T).squeeze(0)

    # Sort and display results
    search_results = []
    score_results = []

    ranked_scores, ranked_idx = torch.sort(sim_scores, descending=True)
    for idx, score in zip(ranked_idx, ranked_scores):
        search_results.append(my_search_list[idx])
        score_results.append(f"{score:.3f}")

    # Final dataframe output
    df = pd.DataFrame({'search_term': search_results, 'similarity_score': score_results})
    st.dataframe(df, use_container_width=True, hide_index=True)