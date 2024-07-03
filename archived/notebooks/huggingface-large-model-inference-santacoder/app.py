"""
This code is adopted from the hugging face SantaCode source code 
https://huggingface.co/spaces/bigcode/santacoder-demo/blob/main/app.py

"""
import gradio as gr
import os
import sagemaker
from sagemaker.base_predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import argparse





description = """# <p style="text-align: center; color: white;"> 🎅 <span style='color: #ff75b3;'>SantaCoder:</span> Code Generation </p>
<span style='color: white;'>This is a demo to generate code with <a href="https://huggingface.co/bigcode/santacoder" style="color: #ff75b3;">SantaCoder</a>,
a 1.1B parameter model for code generation in Python, Java & JavaScript. Provide the endpoint name under the first text field of the Advanced settings"""



FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"

GENERATION_TITLE= "<p style='font-size: 16px; color: white;'>Generated code:</p>"



def post_processing(prompt, completion):
    completion = "<span style='color: #ff75b3;'>" + completion + "</span>"
    prompt = "<span style='color: #727cd6;'>" + prompt + "</span>"
    code_html = f"<br><hr><br><pre style='font-size: 12px'><code>{prompt}{completion}</code></pre><br><hr>"
    return GENERATION_TITLE + code_html



def code_generation(prompt, endpoint_name, max_new_tokens, temperature=0.2, seed=42):
    #set_seed(seed)
    predictor = Predictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker.Session(),
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    input_data = {
      "inputs": prompt,
      "parameters": {
        "do_sample": True,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p":0.95
      }
    }
    completion = predictor.predict(input_data)[0]['generated_text']
    completion = completion[len(prompt):]
    return post_processing(prompt, completion)

def create_demo(endpoint_name="santacoder-sagemaker-endpoint"):
    
    demo = gr.Blocks(
        css=".gradio-container {background-color: #20233fff; color:white}"
    )
    with demo:
        with gr.Row():
            _, colum_2, _ = gr.Column(scale=1), gr.Column(scale=6), gr.Column(scale=1)
            with colum_2:
                gr.Markdown(value=description)
                code = gr.Code(lines=5, language="python", label="Input code", value="def all_odd_elements(sequence):\n    \"\"\"Returns every odd element of the sequence.\"\"\"")

                with gr.Accordion("Advanced settings", open=True):
                    inp = gr.Textbox(label="Endpoint Name", value=endpoint_name)
                    max_new_tokens= gr.Slider(
                        minimum=8,
                        maximum=1024,
                        step=1,
                        value=48,
                        label="Number of tokens to generate",
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.5,
                        step=0.1,
                        value=0.2,
                        label="Temperature",
                    )
                    seed = gr.Slider(
                        minimum=0,
                        maximum=1000,
                        step=1,
                        label="Random seed to use for the generation"
                    )
                run = gr.Button()
                output = gr.HTML(label="Generated code")

        event = run.click(code_generation, [code, inp, max_new_tokens, temperature, seed], output, api_name="predict")
        # gr.HTML(label="Contact", value="<img src='https://huggingface.co/datasets/bigcode/admin/resolve/main/bigcode_contact.png' alt='contact' style='display: block; margin: auto; max-width: 800px;'>")

    # demo.launch(inline=True, debug=True)
    return demo