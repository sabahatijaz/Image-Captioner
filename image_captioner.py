from PIL import Image
import re
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from chatcaptioner.chat import AskQuestions, caption_image
from chatcaptioner.blip2 import Blip2
import openai
from decouple import config

openai.api_key = config("OPENAI_APIKEY")
blip2 = Blip2('FlanT5 XXL', device_id=0, bit8=True)
chat = AskQuestions(None, blip2, 'gpt-3.5-turbo', n_blip2_context=1)
# Initialize BLIP-2 processor and model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def object_identification(image_path):
    """
    Identify objects in the given image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Identified object in the image.
    """
    image = Image.open(image_path)
    prompt = "Question: Is the person wearing sneakers, boots, or other? Just give one word answer Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def make_promt(blip_generated_caption):
    prompt='Given the sentence: "Person wearing blue sneakers with a floral pattern, he is also wearing jeans and is sitting at the park" Remove any descriptors, and replace the descriptor of shoe with "ohwx" only remove descriptors. For example: "Person wearing ohwx sneakers, he is also wearing jeans and is sitting at the park" Now execute this for this sentence: '
    prompt=prompt+ blip_generated_caption + " Please only respond with the result without any extra text or explanation "
    return prompt

def get_gpt3_response(prompt, max_tokens=150, engine="text-davinci-003"):
    """
    Get a response from OpenAI's GPT-3 API.

    Args:
        prompt (str): The input prompt or question.
        max_tokens (int): Limit the length of the response.
        engine (str): Specify the engine to use (e.g., "text-davinci-003").

    Returns:
        str: The generated response from GPT-3.
    """
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1
    )
    answer = response.choices[0].text.strip()
    return answer

def generate_caption_chatcaptioner(image_path):
    """
    Generate captions for the given image using ChatCaptioner.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Generated caption for the image.
    """
    image = Image.open(image_path)
    n_rounds = int(config("N_CHAT_ROUNDS"))
    results = caption_image(blip2, image=image,
                            n_rounds=n_rounds,
                            n_blip2_context=-1,
                            model='gpt-3.5-turbo',
                            print_mode='no')
    return results['ChatCaptioner']['caption']

def merge_sentences(caption):
    """
    Merge individual sentences into a coherent caption.

    Args:
        caption (str): Caption to be merged.

    Returns:
        str: Merged caption.
    """
    sentences = re.split(r'\. ', caption)
    sentences = list(filter(None, sentences))
    merged_caption = sentences[0].capitalize()

    for sentence in sentences[1:]:
        sentence = sentence.capitalize()
        if sentence[0].islower():
            merged_caption += ','
        merged_caption += ' and ' + sentence

    if not re.search(r'[.!?]$', merged_caption):
        merged_caption += '.'

    return merged_caption



def filter_captions(caption, img_object):
    """
    Filter and modify the generated caption based on identified object.

    Args:
        caption (str): Generated caption from the model.
        img_object (str): Identified object in the image.

    Returns:
        str: Filtered and formatted caption.
    """
    caption = merge_sentences(caption)
    # Remove phrases like 'In the image, there is', 'The image depicts', 'The image features'
    caption = re.sub(r'^(In the image, there is|The image depicts|The image features)\s*,?\s*', '', caption,
                     flags=re.IGNORECASE)

    img_object_pattern = f'({config("SHOES")})'# r'(boots|sneakers|shoes|slippers)'
    match = re.search(img_object_pattern, caption, flags=re.IGNORECASE)
    if match:
        img_object = match.group(1)
    else:
        # Default value if no object is found
        img_object = img_object

    # Construct the regular expression pattern dynamically with img_object
    pattern = r'\b(red|blue|green|yellow|black|white|orange|purple)\b\s+and\b\s+\b(red|blue|green|yellow|black|white|orange|purple)\b\s+(?:{})\b'.format(
        img_object)

    # Replace the pattern with the specified object
    caption = re.sub(pattern, img_object, caption, flags=re.IGNORECASE)

    # Construct the regular expression pattern dynamically with img_object
    pattern = r'\b(red|blue|green|yellow|black|white|orange|purple)\b\s+(?:{})\b'.format(img_object)

    # Replace the pattern with the specified object
    caption = re.sub(pattern, img_object, caption, flags=re.IGNORECASE)
    # Find img_object, boots, sneakers, or shoes in the caption and place "ohwx" before them
    words_to_replace = r'\b(?:{}|{})\b'.format(img_object,config("SHOES"))
    caption = re.sub(words_to_replace, r'ohwx \g<0>', caption, flags=re.IGNORECASE)

    # Extract the first line of the caption
    caption = caption.split('.')[0].strip()

    # Capitalize the first letter of the filtered caption
    caption = caption.capitalize()
    return caption


def get_captioned(image_path):
    """
    Generate and filter captions for the given image.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Filtered and formatted caption for the image.
    """
    img_object = object_identification(image_path)
    chatcaptioner_generated_caption = generate_caption_chatcaptioner(image_path)
    # chatcaptioner_generated_caption = filter_captions(chatcaptioner_generated_caption, img_object)
    prompt = make_promt(chatcaptioner_generated_caption)
    # print(prompt)
    chatcaptioner_generated_caption = get_gpt3_response(prompt)
    print(chatcaptioner_generated_caption)
    return chatcaptioner_generated_caption



if __name__ == "__main__":
    image_path = 'input.jpg'
    caption = get_captioned(image_path=image_path)
    print(caption)
