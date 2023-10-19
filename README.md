<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill Fuyu Module

This repository contains the code supporting the Fuyu base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[Fuyu-8B](https://www.adept.ai/blog/fuyu-8b), developed by [Adept](https://www.adept.ai/), is a multimodal language model. You can ask Fuyu a question about an image and retrieve a response. The Autodistill Fuyu integration enables you to use Fuyu for image classification.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

Read the [Fuyu Autodistill documentation](https://autodistill.github.io/autodistill/base_models/fuyu/).

## Installation

To use Fuyu with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-fuyu
```

## Quickstart

```python
from autodistill_fuyu import Fuyu

# define an ontology to map class names to our Fuyu prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = CLIP(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)
predictions = model.predict("image.png")

base_model.label("./context_images", extension=".jpeg")
```

## 

Fuyu is licensed under a [CC-BY-NC license](https://www.adept.ai/blog/fuyu-8b).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!