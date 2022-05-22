import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
import datetime
import matplotlib.pyplot as plt
import numpy as np

# DCGAN
# Ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# Ref: https://www.tensorflow.org/tutorials/generative/dcgan?hl=zh-cn

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
NOISE_DIM_DEFAULT = 100
BATCH_SIZE_DEFAULT = 64
LEARNING_RATE_DEFAULT = 0.0002
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 10


class Generator(nn.Module):
    def __init__(self, noise_dim, hidden_channels_base, output_channels):
        super(Generator, self).__init__()

        self.layers = nn.Sequential(
            # noise -> (8*base)*4*4
            nn.ConvTranspose2d(
                in_channels=noise_dim,
                out_channels=hidden_channels_base * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels_base * 8),
            nn.ReLU(inplace=True),
            # (4*base)*8*8
            nn.ConvTranspose2d(
                in_channels=hidden_channels_base * 8,
                out_channels=hidden_channels_base * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels_base * 4),
            nn.ReLU(inplace=True),
            # (2*base)*16*16
            nn.ConvTranspose2d(
                in_channels=hidden_channels_base * 4,
                out_channels=hidden_channels_base * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels_base * 2),
            nn.ReLU(inplace=True),
            # base*64*64
            nn.ConvTranspose2d(
                in_channels=hidden_channels_base * 2,
                out_channels=hidden_channels_base,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=hidden_channels_base),
            nn.ReLU(inplace=True),
            # output_channels*64*64
            nn.ConvTranspose2d(
                in_channels=hidden_channels_base,
                out_channels=output_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.weight_init()

    def weight_init(self):
        for m in self.layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, z):
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, input_channels, hidden_channels_base):
        super(Discriminator, self).__init__()

        self.layers = nn.Sequential(
            # input*64*64 -> base*32*32
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels_base,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            # (base*2)*16*16
            nn.Conv2d(
                in_channels=hidden_channels_base,
                out_channels=hidden_channels_base * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels_base * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (base*4)*8*8
            nn.Conv2d(
                in_channels=hidden_channels_base * 2,
                out_channels=hidden_channels_base * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels_base * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (base*8)*4*4
            nn.Conv2d(
                in_channels=hidden_channels_base * 4,
                out_channels=hidden_channels_base * 8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels_base * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # prediction: 0 or 1
            nn.Conv2d(
                in_channels=hidden_channels_base * 8,
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Sigmoid(),
        )

        self.weight_init()

    def weight_init(self):
        for m in self.layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.02)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img):
        return self.layers(img).flatten()


def train(
    dataloader,
    discriminator,
    generator,
    optimizer_G,
    optimizer_D,
    noise_dim=NOISE_DIM_DEFAULT,
    max_epochs=MAX_EPOCHS_DEFAULT,
    verbose=EVAL_FREQ_DEFAULT,
    save_interval=-1,
    save_image_path="./gen_images/",
    visual_model=False,
    quiet=False,
):
    if visual_model:
        G_loss_list = []
        D_loss_list = []

    loss = nn.BCELoss()
    for epoch in range(max_epochs):
        G_batch_loss_list = []
        D_batch_loss_list = []
        for img_batch, _ in dataloader:
            batch_size = len(img_batch)

            ones = torch.ones(batch_size).to(DEVICE)
            zeros = torch.zeros(batch_size).to(DEVICE)
            img_batch = img_batch.to(DEVICE)

            # Train Discriminator
            # -------------------

            # real images
            D_img_pred_batch = discriminator.forward(img_batch)
            D_img_loss = loss(D_img_pred_batch, ones)

            # fake images
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(DEVICE)
            fake_img_batch = generator.forward(noise)
            D_fake_pred_batch = discriminator.forward(fake_img_batch.detach())
            D_fake_loss = loss(D_fake_pred_batch, zeros)

            # loss real+fake
            D_loss = D_img_loss + D_fake_loss
            D_batch_loss_list.append(D_loss.item())

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # Train Generator
            # ---------------

            D_fake_pred_batch = discriminator.forward(fake_img_batch)
            G_loss = loss(D_fake_pred_batch, ones)
            G_batch_loss_list.append(G_loss.item())

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

        if (epoch + 1) % verbose == 0:
            if not quiet or visual_model:
                # train loss
                G_loss = sum(G_batch_loss_list) / len(G_batch_loss_list)
                D_loss = sum(D_batch_loss_list) / len(D_batch_loss_list)

                if visual_model:
                    G_loss_list.append(G_loss)
                    D_loss_list.append(D_loss)

            if not quiet:
                print(
                    datetime.datetime.now(),
                    "Epoch",
                    epoch + 1,
                    "\tGenerator Loss = %.5f" % G_loss,
                    "Discriminator Loss = %.5f " % D_loss,
                )

        if save_interval > 0:
            if (epoch + 1) % save_interval == 0:
                save_image(
                    fake_img_batch[:25],
                    os.path.join(save_image_path, f"{epoch + 1}.png"),
                    nrow=5,
                    normalize=True,
                )

    if visual_model:
        plt.plot(
            np.arange(verbose, epoch + 2, step=verbose),
            G_loss_list,
            "-",
            label="generator_loss",
        )
        plt.plot(
            np.arange(verbose, epoch + 2, step=verbose),
            D_loss_list,
            "-",
            label="discriminator_loss",
        )
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def main(args):
    # Create output image directory
    os.makedirs(args.save_image_path, exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # load data
    image_size = 64
    dataset = datasets.MNIST(
        args.data_dir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    )
    
    # indices = np.arange(51200) # specify a subset
    indices = np.arange(len(dataset)) # use full dataset
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=True,
    )

    # import torchvision.utils as vutils
    # real_batch = next(iter(dataloader))
    # plt.figure(figsize=(8,8))
    # plt.axis("off")
    # plt.title("Training Images")
    # plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.show()

    # Initialize models and optimizers
    generator = Generator(
        noise_dim=args.latent_dim, hidden_channels_base=128, output_channels=1
    ).to(DEVICE)
    discriminator = Discriminator(input_channels=1, hidden_channels_base=128).to(DEVICE)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    # Start training
    train(
        dataloader,
        discriminator,
        generator,
        optimizer_G,
        optimizer_D,
        noise_dim=args.latent_dim,
        max_epochs=args.max_epochs,
        verbose=args.eval_freq,
        save_interval=args.save_interval,
        save_image_path=args.save_image_path,
        visual_model=args.visual_model,
        quiet=args.quiet,
    )

    torch.save(generator.state_dict(), os.path.join("./model/", f"{args.model_save_name}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        "-d",
        type=str,
        default="./data/mnist/",
        help="Directory for storing data.",
    )
    parser.add_argument(
        "--save_image_path",
        type=str,
        default="./gen_images/",
        help="Directory for storing generated images.",
    )
    parser.add_argument(
        "--model_save_name",
        type=str,
        default="generator",
        help="Model name.",
    )
    parser.add_argument(
        "--learning_rate",
        "-l",
        type=float,
        default=LEARNING_RATE_DEFAULT,
        help="Learning rate.",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=BATCH_SIZE_DEFAULT, help="Batch size"
    )
    parser.add_argument(
        "--max_epochs",
        "-e",
        type=int,
        default=MAX_EPOCHS_DEFAULT,
        help="Number of epochs to run trainer.",
    )
    parser.add_argument(
        "--eval_freq",
        "-f",
        type=int,
        default=EVAL_FREQ_DEFAULT,
        help="Frequency of evaluation on the test set.",
    )
    parser.add_argument(
        "--visual_model",
        "-v",
        action="store_true",
        help="Visualize model after training.",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="No stdout when training."
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=100,
        help="dimensionality of the latent space",
    )
    parser.add_argument(
        "--save_interval",
        "-s",
        type=int,
        default=-1,
        help="save every SAVE_INTERVAL iterations",
    )
    args = parser.parse_args()

    main(args)
