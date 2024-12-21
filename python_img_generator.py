import torch
from diffusers import StableDiffusionPipeline

def generate_image(prompt, model_path, output_file="output_image.png"):

    
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"generating image for prompt:{prompt}")
        image = pipe(prompt.image)[0]
        
        image.save(output_file)
        print(f"Image saved as {output_file}")
        
    except Exception as e:
        print(f"Error generting image{e}")
        
if __name__ == "__main__":
    MODEL_PATH = "python_img_generator.py"
    
    user_prompt = input("Enter the description of the image you want to generate:")
    
    generate_image(user_prompt, MODEL_PATH)