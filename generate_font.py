import torch
from torchvision.utils import save_image
import torch.nn.functional as F
import argparse

from utils import device
from generator import Generator


def generate_font(run_name: str):
    generator = Generator(126).to(device=device)
    checkpoint = torch.load(f"./runs/{run_name}/models/generator_model.pt", map_location=device)
    generator.load_state_dict(checkpoint["state_dict"])

    one_hot_letter = F.one_hot(torch.tensor([idx for idx in range(26)]), num_classes=26)
    font_noise = torch.randn(100).repeat(26, 1)
    input_ = torch.cat((one_hot_letter, font_noise), dim=1).unsqueeze(2).unsqueeze(3).to(device=device)

    out = generator.eval()(input_).detach()

    save_image(out, f"./runs/{run_name}/generated_fonts/fake_font.png")



parser = argparse.ArgumentParser(description="Generate a font.")
parser.add_argument("--model", type=str, required=True, help="Name of the model (same as the folder name in the './run/' directory).")
run_name = vars(parser.parse_args())["model"]

generate_font(run_name)