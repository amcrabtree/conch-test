import streamlit as st
from PIL import Image
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

# Set Streamlit page configuration
st.set_page_config(page_title="CONCH Matcher")

# Define paths and constants
MODEL_REPO = "MahmoodLab/CONCH"
MODEL_FILENAME = "pytorch_model.bin"

@st.cache_resource
def download_and_load_model(token):
    """
    Downloads the model file from Hugging Face Hub and loads it.
    """
    # Download the model file using the provided token
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, 
        filename=MODEL_FILENAME, 
        use_auth_token=token
    )
    
    # Load the model and preprocessing function
    model, preprocess = create_model_from_pretrained(
        model_cfg="conch_ViT-B-16", checkpoint_path=model_path
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, preprocess, device

# Sidebar: Hugging Face Token Input and Model Setup
with st.sidebar:
    st.header("Model Setup")
    st.write("The application uses CONCH, a pre-trained model from Hugging Face to process inputs.")
    st.write("Please provide your Hugging Face personal access token to download and load the model.")

    # Token input
    hf_token = st.text_input("Hugging Face Token", type="password")
    load_model_button = st.button("Load Model")

    # Model loading status
    model_status = st.empty()

# Check if token is provided and attempt to load the model
if "model" not in st.session_state and load_model_button:
    if hf_token:
        model_status.info("Downloading and loading the model...")
        try:
            model, preprocess, device = download_and_load_model(hf_token)
            st.session_state["model"] = model
            st.session_state["preprocess"] = preprocess
            st.session_state["device"] = device
            model_status.success("Model loaded successfully!")
        except Exception as e:
            model_status.error(f"Error loading model: {e}")
    else:
        model_status.warning("Please enter a valid Hugging Face token.")

# Main Application
if "model" in st.session_state:
    model = st.session_state["model"]
    preprocess = st.session_state["preprocess"]
    device = st.session_state["device"]

    # File input section
    st.header("File Input")

    # Image patch file uploader
    seq_file_extensions = ["jpg", "jpeg", "png"]
    patch_file = st.file_uploader("##### Image patch file:", type=seq_file_extensions)
    st.link_button("Example file", "https://github.com/amcrabtree/conch-test/blob/master/test/tcga_test6.png")

    # Search terms file uploader
    search_file = st.file_uploader("##### Search terms file:", type=["csv", "tsv", "txt"])
    st.link_button("Example file", "https://github.com/amcrabtree/conch-test/blob/master/test/search_terms.txt")

    # Matching section
    st.header("Match Text to Image")
    if patch_file and search_file:
        try:
            # Load and preprocess the image
            image = Image.open(patch_file)
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            st.image(image.resize((224, 224)), caption="Uploaded Image", use_column_width=True)

            # Load and tokenize search terms
            search_text = search_file.read().decode("utf-8")
            search_terms = search_text.split("\n")
            tokenizer = get_tokenizer()
            tokenized_prompts = tokenize(texts=search_terms, tokenizer=tokenizer).to(device)

            # Perform inference
            with torch.inference_mode():
                image_embeddings = model.encode_image(image_tensor)
                text_embeddings = model.encode_text(tokenized_prompts)
                sim_scores = (image_embeddings @ text_embeddings.T).squeeze(0)

            # Sort and display results
            ranked_scores, ranked_indices = torch.sort(sim_scores, descending=True)
            results = [{"Search Term": search_terms[i], "Similarity Score": f"{score:.3f}"} 
                       for i, score in zip(ranked_indices, ranked_scores)]
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error during matching: {e}")
