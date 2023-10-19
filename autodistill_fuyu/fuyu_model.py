from dataclasses import dataclass

import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from PIL import Image
from transformers import (AutoTokenizer, FuyuForCausalLM, FuyuImageProcessor,
                          FuyuProcessor)

pretrained_path = "adept/fuyu-8b"


@dataclass
class Fuyu(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.ontology = ontology

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

        self.image_processor = FuyuImageProcessor()
        self.processor = FuyuProcessor(
            image_processor=self.image_processor, tokenizer=self.tokenizer
        )

        self.model = FuyuForCausalLM.from_pretrained(
            pretrained_path, device_map="cuda:0" if torch.cuda.is_available() else "cpu"
        )

    def predict(self, input: str) -> sv.Classifications:
        prompts = self.ontology.prompts()

        image = Image.open(input)

        classifications = []

        for prompt in prompts:
            text_prompt = (
                "Does the image contain a " + prompt + "? Answer as yes or no."
            )

            model_inputs = self.processor(
                text=text_prompt, images=[image], device="cuda:0"
            )
            for k, v in model_inputs.items():
                model_inputs[k] = v.to("cuda:0" if torch.cuda.is_available() else "cpu")

            generation_output = self.model.generate(**model_inputs, max_new_tokens=7)
            generation_text = self.processor.batch_decode(
                generation_output[:, -7:], skip_special_tokens=True
            ).lower()

            if generation_text == "yes":
                classifications.append(True)
            else:
                classifications.append(False)

        return sv.Classifications(
            class_ids=prompts,
            confidence=[1.0] * len(prompts),
        )
