import torch
import json
import torchvision.transforms as transforms
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline

# Load multimodal models
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
text_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def process_image(image_path):
    """Process and encode medical images using CLIP."""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs)
    return image_features.detach().numpy()


def summarize_text(text):
    """Summarize clinical trial text using an LLM."""
    summary = text_summarizer(text, max_length=50, min_length=20, do_sample=False)
    return summary[0]['summary_text']


def analyze_multimodal_data(trial_json, output_json):
    """Perform multimodal analysis of clinical trials (text + image)."""
    with open(trial_json, "r") as f:
        trial_data = json.load(f)

    for trial in trial_data:
        trial["summary"] = summarize_text(trial["text"])
        trial["image_features"] = process_image(trial["image_path"]).tolist()

    with open(output_json, "w") as f:
        json.dump(trial_data, f, indent=4)

    print(f"Multimodal clinical trial analysis saved to {output_json}")


def main():
    trial_json = "clinical_trial_text.json"
    output_json = "multimodal_trial_analysis.json"
    analyze_multimodal_data(trial_json, output_json)
    print("Multimodal analysis completed.")


if __name__ == "__main__":
    main()
