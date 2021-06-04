from __future__ import print_function
from __future__ import division

import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

#%load_ext tensorboard


class FlowerShazam():
    def __init__(self):
        self.path_flower = './flower_data/flower_data/'
        self.path_flower_train = self.path_flower+'/train/'
        self.path_flower_test = self.path_flower+'/valid/'

        self.BATCH_SIZE = 64
        self.CROP_DIMENSION = 224
        self.RESIZE_DIMENSION = 256

        #ImageNet Means and Stds
        self.MEANS = [0.485, 0.456, 0.406]
        self.STANDARD_DEVIATIONS = [0.229, 0.224, 0.225]

        self.ANGLE = 90

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(self.RESIZE_DIMENSION),
                transforms.CenterCrop(self.CROP_DIMENSION),
                transforms.RandomRotation(self.ANGLE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEANS, self.STANDARD_DEVIATIONS)
            ]),
            'valid': transforms.Compose([
                transforms.Resize(self.RESIZE_DIMENSION),
                transforms.CenterCrop(self.CROP_DIMENSION),
                transforms.RandomRotation(self.ANGLE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.MEANS, self.STANDARD_DEVIATIONS)
            ]),
        }

        json_flowers = './cat_to_name.json'
        self.list_flowers = json.load(open(json_flowers, 'r'))

        # Create training and validation datasets
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(
            self.path_flower, x), self.data_transforms[x]) for x in ['train', 'valid']}
        # Create training and validation dataloaders
        self.dataloaders_dict = {x: torch.utils.data.DataLoader(
            self.image_datasets[x], batch_size=self.BATCH_SIZE, shuffle=True, num_workers=2) for x in ['train', 'valid']}

        self.train_class_idx = self.image_datasets['train'].class_to_idx
        self.valid_class_idx = self.image_datasets['valid'].class_to_idx

        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.model_name = "resnet101"

        # Number of classes in the dataset
        self.num_classes = 102

        # Batch size for training (change depending on how much memory you have)
        self.batch_size = 64

        # Number of epochs to train for
        self.num_epochs = 3

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = True

        # Initialize the model for this run
        self.model_used, input_size = self.initialize_model(
            self.model_name, self.num_classes, self.feature_extract, use_pretrained=True)

    def check_dataset(self):
        # Train set
        print('Number of folders in the train set: ',
              len(os.listdir(self.path_flower_train)))
        print('Number of files in the train set: ', sum(
            [len(files) for r, d, files in os.walk(self.path_flower_train)]))

        # Test set
        print('Number of folders in the test set: ',
              len(os.listdir(self.path_flower_test)))
        print('Number of files in the test set: ', sum(
            [len(files) for r, d, files in os.walk(self.path_flower_test)]))

    def load_names_json(self):
        print(self.list_flowers)
        print('Number of flower categories: ', len(self.list_flowers))
        if len(os.listdir(self.path_flower_train)) != len(self.list_flowers):
          sys.exit('Not a good number of categories in train set')

    # def prepare_dataset(self):
    #     # Data augmentation and normalization for training
    #     # Just normalization for validation
    #     print("Initializing Datasets and Dataloaders...")

    #     # Get names of the classes within each dataset
    #     train_class_idx = self.image_datasets['train'].class_to_idx
    #     valid_class_idx = self.image_datasets['valid'].class_to_idx

    # def displayImage(self, image, image_name=None, normalize=True):
    #     forDisplay = image.numpy().transpose((1, 2, 0))

    #     if normalize:
    #         mean = np.array(self.MEANS)
    #         standard_deviation = np.array(self.STANDARD_DEVIATIONS)
    #         forDisplay = standard_deviation * forDisplay + mean
    #         forDisplay = np.clip(forDisplay, 0, 1)

    #     fig, ax = plt.subplots()
    #     ax.imshow(forDisplay)
    #     ax.set_axis_off()

    #     if image_name:
    #         ax.set_title(image_name)

    #     plt.savefig(image_name+".png")

    def get_dictionary_key(self, dictionary, value):
        return {val: key for key, val in dictionary.items()}[value]

    # def display_image_label(self, tensor, tensor_class_idx, count=1):
    #     images, labels = next(tensor)
    #     for i in range(count):
    #         self.displayImage(images[i], self.list_flowers.get(
    #             self.get_dictionary_key(tensor_class_idx, labels[i].item())))

    # def plot_flowers(self, tensor, tensor_class_idx, count=1, normalize=True):
    #     images, labels = next(tensor)
    #     a = np.floor(count**0.5).astype(int)
    #     b = np.ceil(1.*count/a).astype(int)
    #     fig = plt.figure(figsize=(3.*b, 3.*a))
    #     for i in range(1, count+1):
    #         ax = fig.add_subplot(a, b, i)
    #         ax.plot([1, 2, 3], [1, 2, 3])
    #         forDisplay = images[i].numpy().transpose((1, 2, 0))

    #         if normalize:
    #             mean = np.array(self.MEANS)
    #             standard_deviation = np.array(self.STANDARD_DEVIATIONS)
    #             forDisplay = standard_deviation * forDisplay + mean
    #             forDisplay = np.clip(forDisplay, 0, 1)
    #         ax.imshow(forDisplay)
    #         ax.set_title(self.list_flowers.get(
    #             self.get_dictionary_key(tensor_class_idx, labels[i].item())))
    #         ax.set_axis_off()
    #     #fig.suptitle("%d Flowers" % count, fontsize=16)

    #     plt.show()

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet101":
            """ Resnet101
            """
            model_ft = models.resnet101(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(
                512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if is_inception and phase == 'train':
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4*loss2
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double(
                ) / len(dataloaders[phase].dataset)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'valid':
                    val_acc_history.append(epoch_acc)

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history

    def size_image(self, image):
        width, height = image.size
        return width, height

    def create_thumbnail(self, image, width, height, short_side):
        if width < height:
            return short_side, height
        else:
            return width, short_side

    def test_process(self, image):
        preprocess = self.data_transforms['valid']

        img_pil = Image.open(image)
        print(img_pil)
        img_tensor = preprocess(img_pil)
        print(img_tensor.size())
        img_np = img_tensor.numpy()
        print(img_np.shape)
        #img_tensor.unsqueeze_(0)
        return img_np

    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array'''
        preprocess = self.data_transforms['valid']

        pil_image = Image.open(image)
        tensor_image = preprocess(pil_image)
        numpy_image = tensor_image.numpy()

        return numpy_image

    def imshow(self, image, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        if title is not None:
            ax.set_title(title)
        ax.axis('off')
        ax.imshow(image)

        return ax

    def process_image_tensor(self, image_path):
        process_transforms = self.data_transforms['valid']
        pil_image = Image.open(image_path)
        tensor_image = process_transforms(pil_image)
        return tensor_image

    def convert_labels(self, labels, class_idx):
        names_array = np.array([])
        for i in np.nditer(labels.cpu().numpy()):
            names_array = np.append(names_array, self.list_flowers.get(
                self.get_dictionary_key(class_idx, i.item())))
        return names_array

    def predict(self, image_path, model, class_idx, topk=5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        model.to(device)
        model.eval()
        image_tensor = self.process_image_tensor(image_path).unsqueeze_(0)
        #image_tensor = image_tensor.cuda()
        #model.to(check_gpu())
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model.forward(image_tensor)

        probabilities = torch.exp(output)
        probs, labels = probabilities.topk(topk)

        #print(probs)
        #print(labels)
        #Convert labels to numpy array of names

        #print("Labels:%s" %labels)
        #names = list(list_flowers.values())
        #tests = [names[x] for x in labels[0]]
        return probs.cpu().numpy()[0], self.convert_labels(labels, class_idx)

    #print(predict(valid_dir + '/46/image_01034.jpg', neural_network_model))

    def get_category_num(self, image_path):
        return image_path.split('/')[-2]

    def get_dataset_subtype(self, image_path):
        return image_path.split('/')[-3]

    # def get_class_idx(self, image_path):
    #     return dataset_info[self.get_dataset_subtype(image_path)]['class_idx']

    # Display an image along with the top 5 classes
    def view_class_probability(self, image_path, neural_network_model, class_idx):
        probabilities, classes = self.predict(
            image_path, neural_network_model, class_idx)

        image_elements = image_path.split('/')
        flower_category_num = image_elements[-2]
        # class_type = image_elements[-3]
        #print(dataset_info[class_type]['class_idx'])

        fig, (ax1, ax2) = plt.subplots(figsize=(6, 10), ncols=2)
        ax1 = plt.subplot(2, 1, 1)
        self.imshow(self.process_image(image_path), ax1)

        # Set up title
        title = self.list_flowers.get(str(flower_category_num))
        # Plot flower
        img = self.process_image(image_path)
        self.imshow(img, ax1, title)
        ax2 = fig.add_subplot(2, 1, 2)
        y_pos = np.arange(len(classes))
        #print(probabilities)
        '''performance = np.around(probabilities, decimals = 3)'''

        #print(performance)
        # error = np.random.rand(len(classes))

        ax2.barh(y_pos, probabilities, align='center',
                 color='blue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_xlabel('Probability')
        ax2.set_title('Class Probability')
        plt.tight_layout()

        print("Saving output to ", str(image_elements[-1]))
        plt.savefig(str(image_elements[-1]))

    def train_model_global(self):
        # Send the model to GPU
        self.model_used = self.model_used.to(device)

        # Gather the parameters to be optimized/updated in this run. If we are
        #  finetuning we will be updating all parameters. However, if we are
        #  doing feature extract method, we will only update the parameters
        #  that we have just initialized, i.e. the parameters with requires_grad
        #  is True.
        params_to_update = self.model_used.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name, param in self.model_used.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in self.model_used.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

        # Setup the loss fxn
        criterion = nn.CrossEntropyLoss()
        # Train and evaluate
        self.model_used, hist = self.train_model(
            self.model_used, self.dataloaders_dict, criterion, optimizer_ft, self.num_epochs, is_inception=False)

    def save_model(self, model_used):
        # Save the model checkpoint
        torch.save(model_used.state_dict(), self.model_name + '_save.ckpt')

    def load_model(self):
        # Save the model checkpoint
        torch.load(self.model_name + '_save.ckpt')


def main():
    FS = FlowerShazam()
    # FS.check_dataset()
    # FS.train_model_global()
    # FS.save_model(FS.model_used)

    FS.model_used.load_state_dict(torch.load(FS.model_name + '_save.ckpt'))

    image1 = FS.path_flower_test + '/25/image_06572.jpg'
    FS.view_class_probability(image1, FS.model_used, FS.valid_class_idx)


if __name__ == "__main__":
    main()

