from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torchvision.utils import save_image
import torch.nn.functional as F

from utils import device, initialize_run, save_checkpoint, load_checkpoint, lr_scheduler, save_train_history
from fontDataset import create_dataloader
from generator import Generator
from discriminator import Discriminator

torch.manual_seed(100)


def train(config: dict) -> None:
    run_name = config["run_name"]
    dataset_path = config["dataset_path"]
    chunk_size = config["chunk_size"]

    resume = config["resume"]

    # hyperparameters
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    noise_size = config["noise_size"]
    latent_vector_size = noise_size + 26
    lr = config["lr"]
    lr_decay_rate = config["lr_decay_rate"]
    lr_decay_intervall = config["lr_decay_intervall"]

    # path to the run logging folder
    run_folder = f"./runs/{run_name}/"

    # models
    generator = Generator(latent_vector_size=latent_vector_size).to(device)
    discriminator = Discriminator().to(device)

    # optimizers
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.99))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.99))

    # losses
    criterion_discriminator = nn.BCELoss()
    criterion_generator = nn.BCELoss()

    # resume training-run or initialize new run
    if resume:
        load_checkpoint(generator, optimizer_generator, lr, run_folder + "models/generator_model.pt", device)
        load_checkpoint(discriminator, optimizer_discriminator, lr, run_folder + "models/discrimnator_model.pt", device)
        lr = optimizer_generator.param_groups[0]["lr"]
    else:
        initialize_run(run_folder)

    # train loop
    iteration = 0
    for epoch in range(1, epochs):

        epoch_loss_disc, epoch_loss_gen = [], []
        for chunk_idx in range(56442 // chunk_size):

            for letter_idx in range(26):
                dataset = create_dataloader(dataset_path, chunk_size=chunk_size, chunk_idx=chunk_idx, batch_size=batch_size, letter_idx=letter_idx)
        
                for real_batch in tqdm(dataset, ncols=100, desc=f"epoch: {epoch}, chunk: {chunk_idx}/{56442 // chunk_size}, letter_idx: {letter_idx}"):
                    iteration += 1

                    lr_scheduler(optimizer_generator, iteration, 0, lr, lr_decay_rate, lr_decay_intervall)
                    lr_scheduler(optimizer_discriminator, iteration, 0, lr, lr_decay_rate, lr_decay_intervall)

                    """ train discriminator """
                    discriminator.zero_grad()

                    # train with real images
                    predictions_real = discriminator.train()(real_batch)
                    loss_real = criterion_discriminator(predictions_real, torch.full(tuple(predictions_real.size()), 1.0).to(device=device))
                    loss_real.backward(retain_graph=True)
                    
                    # create fake images
                    one_hot_letter = F.one_hot(torch.tensor([letter_idx for _ in range(batch_size)]), num_classes=26)
                    font_noise = torch.randn(batch_size, noise_size)
                    fake_batch = generator.train()(torch.cat((one_hot_letter, font_noise), dim=1).unsqueeze(2).unsqueeze(3).to(device=device))

                    # train with fake images
                    predictions_fake = discriminator.train()(fake_batch.detach())

                    loss_fake = criterion_discriminator(predictions_fake, torch.full(tuple(predictions_fake.size()), 0.0).to(device=device))
                    loss_fake.backward()
                    optimizer_discriminator.step()

                    # save losses
                    epoch_loss_disc.append(loss_real.item() + loss_fake.item())

                    """ train generator """
                    generator.zero_grad()

                    predictions_fake = discriminator.train()(fake_batch)
                    
                    loss_gen = criterion_generator(predictions_fake, torch.full(tuple(predictions_fake.size()), 1.0).to(device=device))
                    loss_gen.backward()
                    optimizer_generator.step()

                    epoch_loss_gen.append(loss_gen.item()) 

            # save generated images
            save_checkpoint(generator, optimizer_generator, run_folder + "models/generator_model.pt")
            save_checkpoint(discriminator, optimizer_discriminator, run_folder + "models/discrimnator_model.pt")
            save_image(fake_batch[0:5], run_folder + f"result_images/{epoch}_{chunk_idx}_{iteration}_fake_font.png")

            disc_loss = round(float(np.mean(epoch_loss_disc)), 3)
            gen_loss = round(float(np.mean(epoch_loss_gen)), 3)

            # save train history and model checkpoints
            save_train_history(run_name, disc_loss, gen_loss, optimizer_generator.param_groups[0]["lr"])

            print(f"epoch: {epoch} / {epochs}  -  iteration: {iteration}  -  disc_loss: {disc_loss}  -  gen_loss: {gen_loss}\n  -  lr: {optimizer_generator.param_groups[0]['lr']}")


def generate_font(run_name: str):
    generator = Generator(126).to(device=device)
    checkpoint = torch.load(f"./runs/{run_name}/models/generator_model.pt", map_location=device)
    generator.load_state_dict(checkpoint["state_dict"])

    one_hot_letter = F.one_hot(torch.tensor([idx for idx in range(26)]), num_classes=26)
    font_noise = torch.randn(100).repeat(26, 1)
    input_ = torch.cat((one_hot_letter, font_noise), dim=1).unsqueeze(2).unsqueeze(3).to(device=device)

    out = generator.eval()(input_).detach()

    save_image(out, f"./runs/{run_name}/generated_fonts/fake_font.png")



if __name__ == "__main__":
    config = {
        "run_name": "my-model",
        "dataset_path": "./datasets/fonts.hdf5",
        "chunk_size": 1000,
        "resume": False,
        "save_image_intervall": 200,
        "epochs": 20,
        "batch_size": 512,
        "noise_size": 100,
        "lr": 0.002,
        "lr_decay_rate": 0.975,
        "lr_decay_intervall": 250
    }

    train(config)
    # generate_font("my-model")

