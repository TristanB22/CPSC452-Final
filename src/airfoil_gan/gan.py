# standard library imports

# third party imports
import torch
import torch.nn as nn

# local imports


# set up constants
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

NUM_POINTS_POINT_CLOUD = 100
RANDOM_INPUT_NUM = 512


# define generator class
class Generator(nn.Module):

    def __init__(self, dropout_p=0.2):

        super(Generator, self).__init__()

        # define the sequential neural network
        self.fc = nn.Sequential(
            nn.Linear(RANDOM_INPUT_NUM + 3, 256),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_p),
            nn.Linear(256, NUM_POINTS_POINT_CLOUD * 2),
        )

    # define the forward pass of the sequential neural network
    def forward(self, noise, c_label):

        if c_label.dim() == 1:
            c_label = c_label.unsqueeze(1)

        # concatenate the tensor of variables we want the airfoil to have and the noise
        input_tensor = torch.cat((noise, c_label), dim=1)
        return self.fc(input_tensor.to(DEVICE))


# this is the discriminator that is used to assess whether the
# generator design has the correct lift value
class Discriminator(nn.Module):
    def __init__(self):

        super(Discriminator, self).__init__()

        # defining the sequential discriminator
        self.fc = nn.Sequential(
            nn.Linear(
                NUM_POINTS_POINT_CLOUD * 2, 64
            ),  # 100 coordinates with 3 computed or real labels and 3 input values
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    # defining the forward pass for the discriminator
    def forward(self, points):

        # putting the tensor of targets together
        # input_tensor = torch.cat((points, labels, target), dim=1)
        return self.fc(points.to(DEVICE))


# this is the validator that is going to decide what the likely value for a certain coefficient is
class Coeff_Validator(nn.Module):
    def __init__(self):

        super(Coeff_Validator, self).__init__()

        # defining the sequential discriminator
        self.fc = nn.Sequential(
            nn.Linear(NUM_POINTS_POINT_CLOUD * 2, 64),  # 100 coordinates
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
        )

    # defining the forward pass for the discriminator
    def forward(self, points):

        # putting the tensor of targets together
        return self.fc(points.to(DEVICE))


class EarlyStopping:

    # initialize the class
    def __init__(self, patience=7, min_delta=0, saving_path="best_model.ckpt"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.b_loss = None
        self.early_stop = False
        self.saving_path = saving_path

    def __call__(self, v_loss, e_model):

        # update the early stopping params
        if self.b_loss is None:
            self.b_loss = v_loss
            self.save_checkpoint(v_loss, e_model)

        # check if we should update
        elif v_loss > self.b_loss - self.min_delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.save_checkpoint(v_loss, e_model)
            self.b_loss = v_loss
            self.counter = 0

    # save the model checkpoint
    def save_checkpoint(self, val_loss, model):
        print(f"Lower Validation Loss ({self.b_loss:.6f} --> {val_loss:.6f})")
        torch.save(model.state_dict(), self.saving_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.saving_path))
