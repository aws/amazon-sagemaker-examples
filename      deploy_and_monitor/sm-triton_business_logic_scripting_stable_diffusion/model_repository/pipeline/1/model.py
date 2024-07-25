# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import json
import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from torch import autocast
from torch.utils.dlpack import to_dlpack, from_dlpack
from transformers import CLIPTokenizer
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from tqdm.auto import tqdm


class TritonPythonModel:

    def initialize(self, args):
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(json.loads(args["model_config"]),
                                               "generated_image")["data_type"])
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            subfolder="unet",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

    def execute(self, requests):
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            input_text = inp.as_numpy()[0][0].decode()

            # tokenizing
            tokenized_text = self.tokenizer(
                [input_text],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids
            tokenized_text_uncond = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids

            # Querying the text_encoding model
            input_ids_1 = pb_utils.Tensor(
                "input_ids",
                np.concatenate([
                    tokenized_text_uncond.numpy().astype(np.int32),
                    tokenized_text.numpy().astype(np.int32),
                ]),
            )
            encoding_request = pb_utils.InferenceRequest(
                model_name="text_encoder",
                requested_output_names=["last_hidden_state"],
                inputs=[input_ids_1],
            )

            response = encoding_request.exec()
            if response.has_error():
                raise pb_utils.TritonModelException(response.error().message())
            else:
                text_embeddings = pb_utils.get_output_tensor_by_name(
                    response, "last_hidden_state")
            text_embeddings = from_dlpack(text_embeddings.to_dlpack()).clone()
            text_embeddings = text_embeddings.to("cuda")

            # Running Scheduler
            guidance_scale = 7.5
            latents = torch.randn((text_embeddings.shape[0] // 2,
                                   self.unet.in_channels, 64, 64)).to("cuda")

            self.scheduler.set_timesteps(50)
            latents = latents * self.scheduler.sigmas[0]

            for i, t in tqdm(enumerate(self.scheduler.timesteps)):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                print('================')
                print(latent_model_input.shape)
                print(text_embeddings.shape)
                

                with torch.no_grad(), torch.autocast("cuda"):
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[i],
                                              latents).prev_sample

            # VAE decoding
            latents = 1 / 0.18215 * latents

            input_latent_1 = pb_utils.Tensor.from_dlpack(
                "latent_sample", to_dlpack(latents))

            decoding_request = pb_utils.InferenceRequest(
                model_name="vae",
                requested_output_names=["sample"],
                inputs=[input_latent_1],
            )

            decoding_response = decoding_request.exec()
            if response.has_error():
                raise pb_utils.TritonModelException(
                    decoding_response.error().message())
            else:
                decoded_image = pb_utils.get_output_tensor_by_name(
                    decoding_response, "sample")
            decoded_image = from_dlpack(decoded_image.to_dlpack()).clone()

            decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
            decoded_image = decoded_image.detach().cpu().permute(0, 2, 3,
                                                                 1).numpy()
            decoded_image = (decoded_image * 255).round().astype("uint8")

            # Sending results
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "generated_image",
#                     np.array(decoded_image, dtype=self.output_dtype),
                    decoded_image,
                )
            ])
            responses.append(inference_response)
        return responses
    
