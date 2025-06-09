import tkinter as tk
import custom tkinter as ctk
import os
from PIL import Image, ImageTk
from datetime import datetime
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from authtoken import auth_token

# Initial settings
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained( 
"CompVis/stable-diffusion-v1-4", 
revision="fp16" if device == "cuda" else "fp32", 
torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
use_auth_token=auth_token
).to(device)

# GUI
app = ctk.CTk()
app.geometry("1024x800")
app.title("Stable Diffusion Advanced GUI")
ctk.set_appearance_mode("dark")

# Image generation function
def generate(): 
try: 
prompt = prompt_entry.get() 
g_scale = float(scale_entry.get()) 
steps = int(steps_entry.get()) 
seed = int(seed_entry.get()) 
filename = filename_entry.get() or datetime.now().strftime("%Y%m%d_%H%M%S") 
resolution = resolution_option.get() 

width, height = map(int, resolution.split("x")) 

status_label.configure(text="Generating image...", text_color="yellow") 
app.update() 

generator = torch.manual_seed(seed) 
with autocast(device): 
image = pipeline(prompt, guidance_scale=g_scale, num_inference_steps=steps, generator=generator).images[0] 

os.makedirs("output_images", exist_ok=True) 
full_path = f"output_images/{filename}.png" 
image.save(full_path) 

# Display 
img_resized = image.resize((512, 512)) 
img_tk = ImageTk.PhotoImage(img_resized) 
image_label.configure(image=img_tk) 
image_label.image = img_tk 

# Save history 
with open("output_images/history.txt", "a", encoding="utf-8") as f: 
f.write(f"[{datetime.now()}] Prompt: {prompt} | File: {full_path}\n") 

status_label.configure(text=" ✅ Image generated successfully!", text_color="lightgreen") 

except Exception as e: 
status_label.configure(text=f"❌ Error: {str(e)}", text_color="red")

# Load the last generated image
def load_last_image(): 
try: 
files = [f for f in os.listdir("output_images") if f.endswith(".png")] 
files.sort(reverse=True) 
if files: 
img = Image.open(os.path.join("output_images", files[0])).resize((512, 512)) 
img_tk = ImageTk.PhotoImage(img) 
image_label.configure(image=img_tk) 
image_label.image = img_tk 
except: 
pass

# Widgets
prompt_entry = ctk.CTkEntry(app, width=600, height=40, placeholder_text="Enter your prompt...")
prompt_entry.pack(paddy=20)

filename_entry = ctk.CTkEntry(app, width=300, placeholder_text="Save file as (optional)")
filename_entry.pack(paddy=5)

scale_entry = ctk.CTkEntry(app, width=200, placeholder_text="Guidance Scale (e.g. 7.5)")
scale_entry.pack(paddy=5)

steps_entry = ctk.CTkEntry(app, width=200, placeholder_text="Inference Steps (e.g. 50)")
steps_entry.pack(paddy=5)

seed_entry = ctk.CTkEntry(app, width=200, placeholder_text="Seed (e.g. 42)")
seed_entry.pack(paddy=5)

resolution_option = ctk.CTkOptionMenu(app, values=["512x512", "768x512", "768x768"])
resolution_option.set("512x512")
resolution_option.pack(paddy=10)

generate_button = ctk.CTkButton(app, text="🎨 Generate Image", command=generate)
generate_button.pack(paddy=15)

status_label = ctk.CTkLabel(app, text="", text_color="white")
status_label.pack(paddy=5)

image_label = ctk.CTkLabel(app, text="")
image_label.pack(paddy=10)

load_last_image()
app.mainloop()
