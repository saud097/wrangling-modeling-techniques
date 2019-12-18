import argparse
import os
import random

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torchvision import models

import load

# Create argparser
parser = argparse.ArgumentParser(description="Train a model to recognize 102 flower types")
parser.add_argument("data_dir", help="Main directory for image set")
parser.add_argument("-s", "--save_dir", help="Location for saving model checkpoint")
parser.add_argument("-a", "--arch",
                    choices=["vgg11", "vgg11_bn", "vgg13", "vgg13_bn",
                             "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"],
                    default=None, help="Pre-trained model type")
hyper = parser.add_argument_group('hyperparameters')
hyper.add_argument("-l", "--learning_rate", type=float, default=0.001, help="Learning rate")
hyper.add_argument("-u", "--hidden_units", nargs="+", type=int, default=[512, 256],
                    help="Number of nodes per hidden layer")
hyper.add_argument("-e", "--epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("-g", "--gpu", action="store_true", help="Use GPU for training")
randomize = parser.add_mutually_exclusive_group()
randomize.add_argument("-r", "--random_search", action="store_true",
                    help="Conduct a random search of optimizer, epochs, and learning rate")
randomize.add_argument("--full_search", action="store_true",
                       help="Conduct a random search of hidden layer architecture, optimizer," \
                            "epochs, and learning rate")
parser.add_argument("-n", "--n_iter", type=int, default=5,
                    help="Number of model iterations for random search")
args = parser.parse_args()

def create_model(arch_name):
    '''Creates model from architecture name and stores model name with model'''
    model = getattr(models, arch_name)(pretrained=True)
    model.name = arch_name
    return model

def update_classifier(model, hidden_layers):
    '''Updates the classifier of an existing model with a relu/dropout model with
       the specified hidden layers.'''
    # Freeze parameters to avoid training
    for param in model.parameters():
        param.requires_grad = False

    # Get input_size and output sizes
    input_size = model.classifier[0].in_features
    output_size = 102

    # Create classifier and replace pre-trained classifier
    classifier = load.Network(input_size=input_size, output_size=output_size,
                              hidden_layers=hidden_layers)
    model.classifier = classifier

def set_hyperparams(model, randomize=True, optimizer_name='Adam',
                    learnrate=0.0001, epochs=15, gpu=True):
    '''Creates optimizer, learnrate and epoch hyperparameters, with the
       option to randomize instead of manually providing inputs.'''
    # Move model to GPU if selected and available
    device = torch.device("cpu")
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # List of optimizer names
    optimizer_names = ['Adam', 'Adagrad', 'RMSprop', 'Adamax']

    # Create randomization of parameters
    if randomize:
        learnrate = random.uniform(5e-5, 1e-2)
        optimizer = getattr(optim, random.choice(optimizer_names))(
            model.classifier.parameters(), lr=learnrate)
        epochs=random.randint(10, 36) # Limited to 35 max to limit runtime
    else:
        optimizer = getattr(optim, optimizer_name)
        optimizer = optimizer(model.classifier.parameters(), lr=learnrate)

    return optimizer, epochs

def validation(model, testloader, criterion, gpu):
    '''Evaluates model accuracy on a dataset, returns test_loss and accuracy.'''
    test_loss = 0
    accuracy = 0

    device = torch.device("cpu")
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        # Convert back to softmax distribution
        ps = torch.exp(output)
        # Compare highest prob predicted class ps.max(dim=1)[1] with labels
        equality = (labels.data == ps.max(dim=1)[1])
        # Convert to cpu and type FloatTensor for mean
        equality.cpu()
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def model_train(model, trainloader, validloader,
                optimizer, epochs, gpu=True):
    '''Trains a model on a dataset, printing incremental gains for training and
       validation data, returns last captured train_loss, valid_loss, valid_accuracy.'''
    # Create variables for printing
    steps = 0
    running_loss = 0
    print_every = 20 # Set to 20 to minimize difference in loss collected for metrics

    # Conditional GPU activation
    device = torch.device("cpu")
    if gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using GPU?", torch.cuda.is_available())
    model.to(device)

    # Set criterion and scheduler
    criterion = nn.NLLLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=.1)

    # Display model hyperparameters
    learnrate = optimizer.state_dict()['param_groups'][0]['lr']
    optimizer_name = optimizer.__class__.__name__
    print('---Model Hyperparameters---\n'
          'Optimizer: {}  Epochs: {}  Learning Rate: {}\n'.format(
              optimizer_name, epochs, learnrate))

    # Run training
    for e in range(epochs):
        scheduler.step()
        model.train() # Set to training mode 'just in case'
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            steps += 1

            # Zero out gradients for each epoch
            optimizer.zero_grad()

            output = model.forward(images) # Feedforward through model
            loss = criterion(output, labels) # Calculate loss
            loss.backward() # Feed loss back through model
            optimizer.step() # Adjust weights

            running_loss += loss.item()

            # Print testing details
            if steps % print_every == 0:
                # Put in eval mode
                model.eval()
                model.to(device)

                # Turn off gradients
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader,
                                                     criterion, gpu)

                train_loss = running_loss/print_every
                valid_loss = test_loss/len(validloader)
                valid_accuracy = accuracy/len(validloader)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Valid. Loss: {:.3f}.. ".format(valid_loss),
                  "Valid. Accuracy: {:.3f}".format(valid_accuracy))

            # Reset running_loss
            running_loss = 0

            # Make sure training is back on
            model.train()

    # Return last run of metrics for later use
    return train_loss, valid_loss, valid_accuracy.item()

def save_model(model, arch, optimizer, epochs, trainimages,
               train_loss, valid_loss, valid_accuracy, save_location):
    '''Creates a dictionary of model parameters and saves as .pth file.'''
    # Check for no save location
    if save_location is None:
        save_location = os.getcwd()
    # Create directory
    if not os.path.exists(save_location):
        os.makedirs(save_location)

    # Count number of files in dir - accurate count based on dedicated save folder
    files_in_folder = len([name for name in os.listdir(save_location) \
                           if os.path.isfile(name)])

    # Map classes to idx
    model.class_to_idx = trainimages.class_to_idx

    # Save names
    model.name = arch
    optimizer_name = optimizer.__class__.__name__

    # Create dict
    params = {
        'arch': model.name,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'optimizer_name': optimizer_name,
        'optimizer': optimizer,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs,
        'last_train_loss': train_loss,
        'last_valid_loss': valid_loss,
        'last_valid_accuracy': valid_accuracy
    }

    # Save dict
    file_name = 'model_' + model.name + optimizer_name + str(epochs) + \
                str(files_in_folder) + '.pth'
    torch.save(params, os.path.join(save_location, file_name))

def randomize_hidden_layers():
    first_layer = random.randint(500, 4500)
    second_layer = random.randint(50, 300)
    return [first_layer, second_layer]

def randomize_arch():
    potential_arch = ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn",
                      "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]

    return random.choice(potential_arch)
  
def random_search(arch, hidden_layers, n_iter, trainloader, validloader, trainimages,
                  save_location, gpu, full_search=None):
    '''Conducts a random search of model arch and hyperparameters with the option
       to only search hyperparameters.'''

    # Count number of files in dir - accurate count based on dedicated save folder
    files_in_folder = len([name for name in os.listdir(save_location) \
                           if name.endswith('.pth')])
    
    if not arch:
        arch = randomize_arch()
    
    # Run n_iter number of tests
    for i in range(n_iter):
        # Manage 'GPU in use' RuntimeErrors
        try:
            # Create model
            print('\nTest No.', i + files_in_folder)
            model = create_model(arch)
            if full_search:
                hidden_layers = randomize_hidden_layers()
            else:
                hidden_layers = hidden_layers
            update_classifier(model, hidden_layers)

            # Randomize hyperparams
            optimizer, epochs = set_hyperparams(model, gpu=gpu)

            # Train model
            train_loss, valid_loss, valid_accuracy = model_train(
                model, trainloader, validloader, optimizer, epochs, gpu)
            # Save model
            save_model(model, arch, optimizer, epochs, trainimages,
                       train_loss, valid_loss, valid_accuracy, save_location)

        except RuntimeError as e:
            print(e, '\n Moving to next model.')
            continue

if __name__ == "__main__":
    # Set up loader objects
    data_transforms = load.data_transforms()
    image_datasets = load.image_datasets(args.data_dir, data_transforms)
    data_loaders = load.data_loaders(image_datasets)

    # Run random or specified model
    if (args.full_search | args.random_search):
        random_search(arch=args.arch, hidden_layers=args.hidden_units,
                      n_iter=args.n_iter, trainloader=data_loaders.train,
                      validloader=data_loaders.valid,
                      trainimages=image_datasets.train,
                      save_location=args.save_dir, gpu=args.gpu,
                      full_search=args.full_search)
    else:
        model = create_model(args.arch)
        update_classifier(model, args.hidden_units)

        optimizer, epochs = set_hyperparams(
            model, randomize=False, learnrate=args.learning_rate,
            epochs=args.epochs, gpu=args.gpu)

        train_loss, valid_loss, valid_accuracy = model_train(
            model, trainloader=data_loaders.train, validloader=data_loaders.valid,
            optimizer=optimizer, epochs=epochs, gpu=args.gpu)

        save_model(model, arch=args.arch, optimizer=optimizer, epochs=epochs,
                   train_loss=train_loss,valid_loss=valid_loss,
                   valid_accuracy=valid_accuracy, trainimages=image_datasets.train,
                   save_location=args.save_dir)