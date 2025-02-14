import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

# specify the path to the model
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

#define the question and image
question = "What color do you see?"
#define the image as a black slate image create an image and make it a black slate image
image = Image.new("RGB", (224, 224), color="black")

conversation = [
    {
        "role": "<|User|>",
        "content": f"<image_placeholder>\n{question}",
        "images": [image],
    },
    {"role": "<|Assistant|>", "content": ""},
]

# load images and prepare for inputs
pil_images = load_pil_images(conversation)
prepare_inputs = vl_chat_processor(
    conversations=conversation, images=pil_images, force_batchify=True
)

# move tensors to device and convert floating point tensors to bfloat16 if applicable
for key, value in vars(prepare_inputs).items():
    try:
        tensor_value = value.to(vl_gpt.device)
        if isinstance(tensor_value, torch.Tensor):
            if key == "input_ids":
                tensor_value = tensor_value.to(torch.long)
            elif key in ["attention_mask", "images_seq_mask", "images_emb_mask"]:
                tensor_value = tensor_value.to(torch.bool)
            else:
                tensor_value = tensor_value.to(torch.bfloat16)
        setattr(prepare_inputs, key, tensor_value)
    except AttributeError:
        pass

# convert image tensor to bfloat16
if hasattr(prepare_inputs, 'images'):
    prepare_inputs.images = [img.to(torch.bfloat16) for img in prepare_inputs.images]

# convert input tensor to bfloat16
if hasattr(prepare_inputs, 'input'):
    prepare_inputs.input = prepare_inputs.input.to(torch.bfloat16)

# run image encoder to get the image embeddings
inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

# run the model to get the response
outputs = vl_gpt.language_model.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=getattr(prepare_inputs, 'attention_mask'),
    pad_token_id=tokenizer.eos_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=512,
    do_sample=False,
    use_cache=True,
)

answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
print(answer)
