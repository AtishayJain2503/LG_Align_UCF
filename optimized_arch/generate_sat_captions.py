import os
import io
import time
import base64
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
import argparse

# Usage:
# export OPENAI_API_KEY="your-api-key"
# python generate_sat_captions.py --split train
# python generate_sat_captions.py --split val

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_caption(client, base64_image):
    prompt = (
        "You are an expert satellite imagery analyst and cartographer. "
        "Describe the explicit geometric layout and topology of this top-down aerial image. "
        "Focus rigidly on the intersection patterns, road shapes (straight, curved), building layouts, "
        "and presence of vegetation or distinct environmental boundaries. Avoid assumptions about macro-geography."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=80,  # Keep descriptions structurally dense and concise
            temperature=0.3
        )
        return response.choices[0].message.content.strip().replace('\n', ' ')
    except Exception as e:
        print(f"API Error: {e}")
        return "Error"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='val', help='Dataset split to process')
    parser.add_argument('--data_path', type=str, default='/lustre/fs1/home/at387336/LGAlign_project', help='Root dataset path')
    args = parser.parse_args()

    client = OpenAI() # Assumes OPENAI_API_KEY is in environment

    split_file = f'{args.data_path}/splits/{args.split}-19zl.csv'
    out_file = f'{args.data_path}/lang/gpt-4o/T1_{args.split}_sat-19zl.csv'
    
    print(f"Loading split: {split_file}")
    df = pd.read_csv(split_file, header=None)
    sat_image_paths = df.iloc[:, 0].values # First column is satellite image path
    
    # Check if we already have partial progress
    if os.path.exists(out_file):
        out_df = pd.read_csv(out_file)
        start_idx = len(out_df)
        print(f"Found existing file with {start_idx} rows. Resuming...")
    else:
        out_df = pd.DataFrame(columns=['Text'])
        start_idx = 0

    print(f"Processing {len(sat_image_paths) - start_idx} images for {args.split} split...")
    
    new_texts = []
    
    try:
        for i in tqdm(range(start_idx, len(sat_image_paths))):
            img_path = f"{args.data_path}/{sat_image_paths[i]}"
            
            if not os.path.exists(img_path):
                print(f"Skipping missing image: {img_path}")
                new_texts.append("Missing image")
                continue
                
            base64_img = encode_image(img_path)
            caption = generate_caption(client, base64_img)
            new_texts.append(caption)
            
            # Rate limiting for OpenAI API (if not tier 4/5)
            time.sleep(0.05) 
            
            # Save progress every 100 images
            if (i + 1) % 100 == 0:
                temp_df = pd.DataFrame({'Text': new_texts})
                out_df = pd.concat([out_df, temp_df], ignore_index=True)
                out_df.to_csv(out_file, index=False)
                new_texts = [] # clear buffer

    except KeyboardInterrupt:
        print("\nProcess interrupted. Saving progress...")
    finally:
        if new_texts:
            temp_df = pd.DataFrame({'Text': new_texts})
            out_df = pd.concat([out_df, temp_df], ignore_index=True)
            out_df.to_csv(out_file, index=False)
        print(f"Finished processing up to index {len(out_df)}. Saved to {out_file}.")

if __name__ == "__main__":
    main()
