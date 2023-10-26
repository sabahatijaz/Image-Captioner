# Image-Captioner

ChatCaptioner: Image Captioning and Chatbot Experience

ChatCaptioner Logo

Welcome to ChatCaptioner, your go-to solution for generating creative and meaningful captions for images! ChatCaptioner combines the power of generative AI models and chatbot technology to provide an interactive and enriching experience for users. Whether you need captivating captions for your images or want to engage in a conversation about visual content, ChatCaptioner has got you covered.
Features

    Image Captioning: Generate descriptive and engaging captions for your images.
    Interactive Chatbot: Engage in conversations with ChatCaptioner about images and their contents.
    Object Identification: Automatically identify objects in the given images.
    Dynamic Caption Filtering: Filter and modify generated captions based on identified objects.
    Seamless Integration: Easily integrate ChatCaptioner into your projects and applications.

Getting Started

Clone the ChatCaptioner repository to your local machine using the following command:

bash

git clone https://github.com/Vision-CAIR/ChatCaptioner.git

Navigate to the cloned repository and install the required dependencies:

bash

cd ChatCaptioner
pip install -r requirements.txt

Usage
1. Generate Captions for Images

To generate a caption for an image, simply run the image_captioner.py script with the path to the input image as an argument:

bash

python image_captioner.py --image_path /path/to/your/image.jpg

This will output a creative and descriptive caption for the given image.
2. Engage in a Conversation

ChatCaptioner allows you to have interactive conversations about images. Run the chatbot.py script to initiate a chat session:

bash

python chatbot.py

You can now ask questions and discuss images with ChatCaptioner, receiving informative and engaging responses.
Advanced Usage
Object Identification

ChatCaptioner can automatically identify objects in images. To identify objects in an image, run the following command:

bash

python image_captioner.py --image_path /path/to/your/image.jpg --identify_objects

This will provide you with a list of identified objects in the given image.
Dynamic Caption Filtering

You can filter and modify generated captions based on identified objects. Use the --filter_captions flag while generating captions:

bash

python image_captioner.py --image_path /path/to/your/image.jpg --filter_captions

ChatCaptioner will filter the generated caption to focus on the identified objects, providing you with a more tailored description.
Example

bash

python image_captioner.py --image_path /path/to/your/image.jpg

Output:

less

Generated Caption: "A person wearing ohwx sneakers, sitting on a bench in the park, enjoying a sunny day."

