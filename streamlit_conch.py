import streamlit as st
from io import StringIO
import os
from PIL import Image
import pandas as pd
import subprocess

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch

from huggingface_hub import hf_hub_download

st.set_page_config(page_title="CONCH Matcher")

# Sidebar for model download and token entry
with st.sidebar:
    st.header("Model Setup")
    model_path = "./pytorch_model.bin"
    if not os.path.exists(model_path):
        st.write("The required model file (`pytorch_model.bin`) is missing. Please download it before proceeding.")
        hf_token = st.text_input("Hugging Face token", placeholder="Enter your token here...")
        if st.button("Download Model"):
            if hf_token:
                try:
                    subprocess.run(["huggingface-cli", "login", "--token", hf_token], check=True)
                    hf_hub_download(repo_id="MahmoodLab/CONCH", filename="pytorch_model.bin", local_dir="./")
                    st.success("Model downloaded successfully! Refresh the page to proceed.")
                except subprocess.CalledProcessError as e:
                    st.error(f"An error occurred during login or download: {e}")
            else:
                st.error("Please provide a valid Hugging Face token.")
        st.stop()  # Prevent the rest of the app from loading until the model is available.
    else:
        st.success("Model file found! You can proceed.")

    # Load model
    try:
        model, preprocess = create_model_from_pretrained(model_cfg='conch_ViT-B-16', checkpoint_path=model_path)
        model = model.eval()
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    # File input
    st.header("File Input")
    seq_file_extensions = ["jpg", "jpeg", "png"]
    patch_file = st.file_uploader("##### Image patch file [[example](https://github.com/amcrabtree/conch-test/blob/main/test/tcga_test6.png)]:", type=seq_file_extensions)
    search_file = st.file_uploader("##### Search terms file [[example](https://github.com/amcrabtree/conch-test/blob/master/test/search_terms.txt)]:", type=["csv", "tsv", "txt"])

# Match text to image
if patch_file and search_file:
    try:
        # Load image
        image = Image.open(patch_file)
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        st.image(image.resize((224, 224)))

        # Load search terms list
        search_text = search_file.read().decode("utf-8")
        my_search_list = search_text.split("\n")

        tokenizer = get_tokenizer()
        tokenized_prompts = tokenize(texts=my_search_list, tokenizer=tokenizer).to(device)

        with torch.inference_mode():
            image_embeddings = model.encode_image(image_tensor)
            text_embeddings = model.encode_text(tokenized_prompts)
            sim_scores = (image_embeddings @ text_embeddings.T).squeeze(0)

        search_results = []
        score_results = []

        ranked_scores, ranked_idx = torch.sort(sim_scores, descending=True)
        for idx, score in zip(ranked_idx, ranked_scores):
            search_results.append(my_search_list[idx])
            score_results.append(f"{score:.3f}")

        # Final dataframe output
        df = pd.DataFrame({'search_term': search_results, 'similarity_score': score_results})
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Error during processing: {e}")