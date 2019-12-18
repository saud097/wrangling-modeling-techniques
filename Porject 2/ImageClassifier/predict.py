import argparse
import matplotlib.pyplot as plt
import os
import random
import re
import torch
from PIL import Image
from torchvision import transforms

import load

# Create parser
parser = argparse.ArgumentParser(description="Predict type of flower from an image")
parser.add_argument("image_path", 
                    help="Path to image (or folder if predicting multiple)")
parser.add_argument("checkpoint", help="Path to checkpoint")
parser.add_argument("-k", "--top_k", type=int, default=5, 
                    help="Number of potential flowers to predict")
parser.add_argument("-c", "--category_names", action="store_true", 
                    help="Use real category names")
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU for prediction")
parser.add_argument("-d", "--display", action="store_true", 
                    help="Display predictions as image")
parser.add_argument("-m", "--multiple", action="store_true", 
                    help="Predict multiple random images")
parser.add_argument("-n", "--n_images", type=int, default=5,
                    help="Number of random images to predict")
args = parser.parse_args()

def predict(image_path, model, topk=5, category_names=True, 
            gpu=True, flower_names=None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Extract image
    image = Image.open(image_path)
    
    # Convert to tensor for PyTorch
    processed_image = load.process_image(image)
    tensor_image = torch.from_numpy(processed_image).float()
    
    # Format tensor for input into model
    tensor_image = tensor_image.unsqueeze(0)
    
    # Use GPU?
    if gpu:
        # Check GPU availability
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    
    tensor_image = tensor_image.to(device)
    
    # Run image through model
    model.eval() # Put model in evaluation mode (doesn't update weights)
    model.to(device)
    output = model.forward(tensor_image)
    ps = torch.exp(output) # reverse the log conversion
    probs, classes = ps.topk(topk)
    
    # Covert to cpu
    probs = probs.cpu()
    classes = classes.cpu()
    # Convert to numpy
    probs = probs.detach().numpy()[0]
    classes = [str(i) for i in classes.numpy()[0]] # Also convert classes to str
    
    # Convert to real names
    if category_names:
        folder_convert = model.class_to_idx.items()
        flower_idx = []
        for pic_class in classes:
            for item in folder_convert:
                if str(item[1]) == pic_class:
                    flower_idx.append(item[0])
        classes = [flower_names[idx] for idx in flower_idx]
    
    return probs, classes

def select_random_image(path):
    '''Selects a random file from location. Assumes file structure compatible with PyTorch
       data loading. Path is of the form main_directory/folder.'''
    # Select random file
    random_folder = random.choice(os.listdir(path))
    random_file = random.choice(os.listdir(os.path.join(path, random_folder)))

    # Create image path
    image_path = os.path.join(path, random_folder, random_file)

    return image_path

def display_preds(path, model, random=True, topk=5, category_names=True, gpu=True, 
                  flower_names=None):
    '''Displays a specified or random image and it's predicted categories as a chart.
       If random=True, assumes file structure compatible with PyTorch data loading, and 
       path is of the form main_directory/folder.'''
    if random:
        # Create image path
        image_path = select_random_image(path)
        first_split = image_path.split('\\')[1]
        folder_number = first_split.split('/')[0]
        title = flower_names[folder_number]
    else:
        image_path = path
        title = path.split('/')[-2]
        folder_convert = model.class_to_idx.items()
        #print(folder_convert)
        if category_names:
            for item in folder_convert:
                if str(item[1]) == title:
                    title = flower_names[str(item[1])]
        else:
            for item in folder_convert:
                if item[0] == title:
                    title = str(item[1])
    
    # Complete preds
    probs, classes = predict(image_path, model, topk=topk, 
                             category_names=category_names, 
                             gpu=gpu, flower_names=flower_names)
    
    fig, ax = plt.subplots(2, figsize=(4, 8))
    image = Image.open(image_path)
    image_transform = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224)])
    ax[0].set_title(title.title())
    ax[0].imshow(image_transform(image))
    ax[0].tick_params(
        axis='both',       # Changes apply to the both axes
        which='both',      # Both major and minor ticks are affected
        bottom=False,      # Ticks along the bottom edge are off
        labelbottom=False, # Labels on the bottom edge are off
        left=False,
        labelleft=False)
    
    # Show Prediction
    ax[1].set_title('Predictions')
    ax[1].barh(classes, probs)
    ax[1].invert_yaxis()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create prediction model
    model = load.pretrained_model(args.checkpoint)
    
    # Predict and display results in requested format
    if args.multiple:
        for i in range(args.n_images):
            display_preds(args.image_path, model, topk=args.top_k, 
                          flower_names=load.flower_names)
    else:
        if args.display:
            display_preds(args.image_path, model, topk=args.top_k, 
                          category_names=args.category_names, random=False, 
                          flower_names=load.flower_names)
        else:
            probs, classes = predict(args.image_path, model, topk=args.top_k, 
                                     category_names=args.category_names, 
                                     gpu=args.gpu, flower_names=load.flower_names)
            print(probs)
            print(classes)
