import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from Generator import Generator
from Discriminator import Discriminator
from Dataset import ABDataset



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "/kaggle/input/vangogh2photo/vangogh2photo/train"
VAL_DIR = "/kaggle/input/vangogh2photo/vangogh2photo/val"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_B = "/kaggle/input/checkpoints/genB.pth.tar"
CHECKPOINT_GEN_A = "/kaggle/input/checkpoints/genA.pth.tar"
CHECKPOINT_DISC_A = "/kaggle/input/checkpoints/discA.pth.tar"
CHECKPOINT_DISC_B = "/kaggle/input/checkpoints/discB.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
    is_check_shapes=False
)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def train_fn(discA, discB, genB, genA, loader, optim_disc, optim_gen, l1, mse, d_scaler, g_scaler):
  H_reals = 0
  H_fakes = 0
  loop = tqdm(loader, leave=True)

  for idx, (imgA, imgB) in enumerate(loop):
    imgA = imgA.to(DEVICE)
    imgB = imgB.to(DEVICE)

    # Train Discriminator
    with torch.cuda.amp.autocast():
      fake_imgA = genA(imgB)
      dA_real = discA(imgA)
      dA_fake = discA(fake_imgA.detach())

      lossA_real = mse(dA_real, torch.ones_like(dA_real))
      lossA_fake = mse(dA_fake, torch.zeros_like(dA_fake))
      lossA = lossA_real + lossA_fake

      fake_imgB = genB(imgA)
      dB_real = discB(imgB)
      dB_fake = discB(fake_imgB.detach())

      lossB_real = mse(dB_real, torch.ones_like(dB_real))
      lossB_fake = mse(dB_fake, torch.zeros_like(dB_fake))
      lossB = lossB_real + lossB_fake

      D_loss = (lossA+lossB)/2

    optim_disc.zero_grad()
    d_scaler.scale(D_loss).backward()
    d_scaler.step(optim_disc)
    d_scaler.update()

    # Train Generator
    with torch.cuda.amp.autocast():
      # Adversarial Loss
      dA_fake = discA(fake_imgA)
      dB_fake = discB(fake_imgB)
      lossA_fake = mse(dA_fake, torch.ones_like(dA_fake))
      lossB_fake = mse(dB_fake, torch.ones_like(dB_fake))

      # Cycle Loss
      cycleB = genB(fake_imgA)
      cycleA = genA(fake_imgB)
      cycleA_loss = l1(imgA, cycleA)
      cycleB_loss = l1(imgB, cycleB)

      # Identity Loss
      identityA = genA(imgA)
      identityB = genB(imgB)
      identityA_loss = l1(imgA, identityA)
      identityB_loss = l1(imgB, identityB)

      # total loss
      G_loss = lossA_fake + lossB_fake + LAMBDA_CYCLE*(cycleA_loss + cycleB_loss) + LAMBDA_IDENTITY*(identityA_loss + identityB_loss)

    optim_gen.zero_grad()
    g_scaler.scale(G_loss).backward()
    g_scaler.step(optim_gen)
    g_scaler.update()

    if idx % 200 == 0:
      save_image(fake_imgA * 0.5 + 0.5, f"/kaggle/working/A_{idx}.png")
      save_image(fake_imgB * 0.5 + 0.5, f"/kaggle/working/B_{idx}.png")

    loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))


discA = Discriminator().to(DEVICE)
discB = Discriminator().to(DEVICE)
genB = Generator().to(DEVICE)
genA = Generator().to(DEVICE)
def main():
  optim_disc = optim.Adam(list(discA.parameters()) + list(discB.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))
  optim_gen = optim.Adam(list(genB.parameters()) + list(genA.parameters()), lr=LEARNING_RATE, betas=(0.5, 0.999))
  l1 = nn.L1Loss()
  mse = nn.MSELoss()

  if LOAD_MODEL:
    load_checkpoint(CHECKPOINT_GEN_A, genA, optim_gen, LEARNING_RATE)
    load_checkpoint(CHECKPOINT_GEN_B, genB, optim_gen, LEARNING_RATE)
    load_checkpoint(CHECKPOINT_DISC_A, discA, optim_disc, LEARNING_RATE)
    load_checkpoint(CHECKPOINT_DISC_B, discB, optim_disc, LEARNING_RATE)

  dataset = ABDataset(root_A=TRAIN_DIR + "/trainA", root_B=TRAIN_DIR + "/trainB", transform=transforms)
  val_dataset = ABDataset(root_A=VAL_DIR + "/testA", root_B=VAL_DIR + "/testB", transform=transforms)
  val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
  loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
  g_scaler = torch.cuda.amp.GradScaler()
  d_scaler = torch.cuda.amp.GradScaler()

  for epoch in range(NUM_EPOCHS):
    print("Epoch: ", epoch)
    train_fn(discA, discB, genB, genA, loader, optim_disc, optim_gen, l1, mse, d_scaler, g_scaler)

    if SAVE_MODEL:
      save_checkpoint(genA, optim_gen, filename="genA.pth.tar")
      save_checkpoint(genB, optim_gen, filename="genB.pth.tar")
      save_checkpoint(discA, optim_disc, filename="discA.pth.tar")
      save_checkpoint(discB, optim_disc, filename="discA.pth.tar")

if __name__ == "__main__":
    main()