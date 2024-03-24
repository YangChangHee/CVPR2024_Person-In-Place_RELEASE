import gradio as gr
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import os

from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from src.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
     "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
 )

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if torch.cuda.is_available():
    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to('cuda')


with gr.Blocks().queue() as demo:
    with gr.Group():
        with gr.Column():
            with gr.Row() as row_1:
                with gr.Column() as step_1:
                    gr.Markdown('### <center>Input Background Image</center>')  
                    input_image = gr.Image(type="filepath",label="Input Image", elem_classes="image_upload")

                with gr.Column() as step_2:  
                    gr.Markdown('### <center>Guidance Image</center>')   
                    guidance_image = gr.Image(type="filepath", label="Guidance Image", elem_id="guidance_image")

                with gr.Column() as step_3:  
                    gr.Markdown('### <center>Merge Image</center>')   
                    Merge_images = gr.Image(type="numpy", label="Merge Images", elem_id="Merge_image")
            with gr.Row() as row_2:
                with gr.Column() as step_3:
                    merge_button = gr.Button("Merge Images !!") 
            with gr.Row() as row_3:
                with gr.Column() as step_4:
                    edit_image = gr.ImageEditor(type="filepath", label="edit Image", elem_id="edit_image")
                with gr.Column() as step_5:
                    res_image = gr.Image(type="filepath", label="result Image", elem_id="result_image")
            with gr.Row() as row_4:
                with gr.Column() as step_6:
                    masked_button = gr.Button("HOI image editing Button !!") 
            with gr.Row() as row_5:
                with gr.Column() as step_7:
                    prompt = gr.Textbox(label='Prompt')
                with gr.Accordion('Advanced options', open=False):
                    num_steps = gr.Slider(label='Steps',
                                        minimum=1,
                                        maximum=100,
                                        value=20,
                                        step=1)
                    text_scale = gr.Slider(label='Text Guidance Scale',
                                                minimum=0.1,
                                                maximum=30.0,
                                                value=7.5,
                                                step=0.1)
                    seed = gr.Slider(label='Seed',
                                    minimum=1,
                                    maximum=2147483647,
                                    step=1)  
                    
                    sketch_scale = gr.Slider(label='Sketch Guidance Scale',
                                                minimum=0.0,
                                                maximum=1.0,
                                                value=1.0,
                                                step=0.05)

    def merge_images(input_image, guidance_image):
        inp_image = cv2.imread(input_image)
        gid_image = cv2.imread(guidance_image)
        alpha = 0.5
        img_list = []

        for i in [0.5]:
            dst = cv2.addWeighted(inp_image, i, gid_image, (1 - i), 0)
            img_list.append(dst)

        return img_list[0]
    
    def masked_images(input_image, guidance_image,edit_image,prompt,
             num_steps,
             text_scale,
             sketch_scale,
             seed):
        inp_image = Image.fromarray(cv2.imread(input_image))
        gid_image = Image.fromarray(cv2.imread(guidance_image))
        layer_1=cv2.imread(edit_image['layers'][0])
        index_position=np.where(layer_1!=0)
        memory=np.zeros(layer_1.shape)
        memory[index_position]=255
        memory=Image.fromarray(memory.astype(np.uint8))
        new_image = pipe(
            prompt,
            num_inference_steps=num_steps,
            guidance_scale=text_scale,
            generator=torch.manual_seed(seed),
            image=inp_image,
            control_image=gid_image,
            controlnet_conditioning_scale=sketch_scale,
            mask_image=memory
        ).images[0]
        new_image = np.array(new_image)
        return new_image


    merge_button.click(fn=merge_images, inputs=[input_image, guidance_image], outputs=Merge_images)
    masked_button.click(fn=masked_images, inputs=[input_image, guidance_image,edit_image,prompt,num_steps,text_scale,sketch_scale,seed],outputs=res_image)
    # #merged_images.select(select_image, None, sketch)

if __name__ == '__main__':
    demo.launch(inbrowser=True)