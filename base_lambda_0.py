import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools
import time
import pickle
# own code
import Datasets
import Utility
import Models

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Utility.set_seed(719)

# Dataset
img_ds = Datasets.RandomPhotoAndMonetDataset()
img_dl = DataLoader(img_ds, batch_size=1, pin_memory=True)
sample_photo = torch.reshape(Datasets.PhotoDataset()[4], [1,3,256,256]) # shape must match the input expected by the network

# Training parameters

name = "base_lambda_0" # name of the attempt
epochs = 50 # number of epochs to train the model
start_lr = 2e-4
decay_epoch = int(epochs/2)
lmbda = 0
idt_coef = 0.5
# models
gen_mtp = Models.Generator(3, 3) # monet to photo
gen_ptm = Models.Generator(3, 3) # photo to monet, this is the one we care about later on
disc_m = Models.Discriminator(3) # discriminates whether it is a monet image
disc_p = Models.Discriminator(3) # discriminates whether it is an actual photo

mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
adam_gen = torch.optim.Adam(itertools.chain(gen_mtp.parameters(), gen_ptm.parameters()),
                                 lr = start_lr, betas=(0.5, 0.999))
adam_disc = torch.optim.Adam(itertools.chain(disc_m.parameters(), disc_p.parameters()),
                                  lr=start_lr, betas=(0.5, 0.999))
sample_fake_monet = Utility.Sample_Fake_Image()
sample_fake_photo = Utility.Sample_Fake_Image()
gen_lr = Utility.lr_sched(decay_epoch, epochs)
disc_lr = Utility.lr_sched(decay_epoch, epochs)
gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(adam_gen, gen_lr.step)
disc_lr_sched = torch.optim.lr_scheduler.LambdaLR(adam_disc, disc_lr.step)

# for quick saving of the model later on
model_state = {
    "gen_mtp": gen_mtp,
    "gen_ptm": gen_ptm,
    "disc_m": disc_m,
    "disc_p": disc_p
}

# init models, DOES NOT WORKT ATM
#Utility.init_weights(gen_mtp)
#Utility.init_weights(gen_ptm)
#Utility.init_weights(disc_m)
#Utility.init_weights(disc_p)
gen_mtp = gen_mtp.to(device)
gen_ptm = gen_ptm.to(device)
disc_m = disc_m.to(device)
disc_p = disc_p.to(device)

epoch_statistics_base_dict = {
    "idt_loss_monet": 0,
    "idt_loss_photo": 0,
    "cycle_loss_monet": 0,
    "cycle_loss_photo": 0,
    "adv_loss_monet": 0,
    "adv_loss_photo": 0,
    "real_monet_disc_loss": 0,
    "fake_monet_disc_loss": 0,
    "real_photo_disc_loss": 0,
    "fake_photo_disc_loss": 0
}

statistics = [] # list containining one dict per epoch
for epoch in range(epochs):
    epoch_statistics = epoch_statistics_base_dict.copy()
    start_time = time.time()
    avg_gen_loss = 0.0
    avg_disc_loss = 0.0
    for (real_photo, real_monet) in img_dl:
        (real_photo, real_monet) = real_photo.to(device), real_monet.to(device)
        # forward pass: at first the generators only
        ## make discriminators unaffected by the changes
        Utility.update_req_grad([disc_m, disc_p], False)
        adam_gen.zero_grad()
        # turn real photo to fake monet
        fake_photo = gen_mtp(real_monet)
        fake_monet = gen_ptm(real_photo)
        # regenerate the original by passing through the fake
        cycl_monet = gen_ptm(fake_photo)
        cycl_photo = gen_mtp(fake_monet)
        # transform monet to monet, meaning no changes should happen
        id_monet = gen_ptm(real_monet)
        id_photo = gen_mtp(real_photo)

        # calculate three kinds of generator losses - identity, cycle consistency and Adversarial
        ## calculate identity loss
        idt_loss_monet = l1_loss(id_monet, real_monet) * lmbda * idt_coef
        epoch_statistics["idt_loss_monet"] += idt_loss_monet.item()
        idt_loss_photo = l1_loss(id_photo, real_photo) * lmbda * idt_coef
        epoch_statistics["idt_loss_photo"] += idt_loss_photo.item()
        ## calculate cycle_consistency_loss
        cycle_loss_monet = l1_loss(cycl_monet, real_monet) * lmbda
        epoch_statistics["cycle_loss_monet"] += cycle_loss_monet.item()
        cycle_loss_photo = l1_loss(cycl_photo, real_photo) * lmbda
        epoch_statistics["cycle_loss_photo"] += cycle_loss_photo.item()
        ## calculate adversarial loss
        ### let discriminator rate the fake
        fake_monet_disc_result = disc_m(fake_monet)
        fake_photo_disc_result = disc_p(fake_photo)
        ### compare the discriminators opinion to the desired (wrong) answer (1)
        result_if_real = torch.ones(fake_monet_disc_result.size()).to(device)
        adv_loss_monet = mse_loss(fake_monet_disc_result, result_if_real)
        epoch_statistics["adv_loss_monet"] += adv_loss_monet.item()
        adv_loss_photo = mse_loss(fake_photo_disc_result, result_if_real)
        epoch_statistics["adv_loss_photo"] += adv_loss_photo.item()
        # total generator loss
        total_gen_loss = cycle_loss_monet + adv_loss_monet                      + cycle_loss_photo + adv_loss_photo                      + idt_loss_monet + idt_loss_photo

        # backward pass
        total_gen_loss.backward()
        adam_gen.step()

        # Forward pass through Descriminator
        ## this time, the discriminators should be affected
        Utility.update_req_grad([disc_m, disc_p], True)
        adam_disc.zero_grad()
        # cycle through 50 previous fakes to avoid oscillations
        fake_monet = sample_fake_monet([fake_monet.cpu().data.numpy()])[0]
        fake_photo = sample_fake_photo([fake_photo.cpu().data.numpy()])[0]
        fake_monet = torch.tensor(fake_monet).to(device)
        fake_photo = torch.tensor(fake_photo).to(device)

        real_monet_disc_result = disc_m(real_monet)
        fake_monet_disc_result = disc_m(fake_monet)
        real_photo_disc_result = disc_p(real_photo)
        fake_photo_disc_result = disc_p(fake_photo)
        # we are not feeding the network the images we know are to be in the wrong category

        result_if_real = torch.ones(real_monet_disc_result.size()).to(device)
        result_if_fake = torch.zeros(fake_monet_disc_result.size()).to(device)

        # Descriminator losses
        # --------------------
        real_monet_disc_loss = mse_loss(real_monet_disc_result, result_if_real)
        epoch_statistics["real_monet_disc_loss"] += real_monet_disc_loss.item()
        fake_monet_disc_loss = mse_loss(fake_monet_disc_result, result_if_fake)
        epoch_statistics["fake_monet_disc_loss"] += fake_monet_disc_loss.item()
        real_photo_disc_loss = mse_loss(real_photo_disc_result, result_if_real)
        epoch_statistics["real_photo_disc_loss"] += real_photo_disc_loss.item()
        fake_photo_disc_loss = mse_loss(fake_photo_disc_result, result_if_fake)
        epoch_statistics["fake_photo_disc_loss"] += fake_monet_disc_loss.item()

        monet_disc_loss = (real_monet_disc_loss + fake_monet_disc_loss) / 2
        photo_disc_loss = (real_photo_disc_loss + fake_photo_disc_loss) / 2
        total_disc_loss = monet_disc_loss + photo_disc_loss

        # Backward
        monet_disc_loss.backward()
        photo_disc_loss.backward()
        adam_disc.step()
    epoch_statistics["epoch"] = epoch + 1
    epoch_statistics["time"] = time.time() - start_time
    # save sample image
    Utility.update_req_grad([gen_ptm], False)
    epoch_statistics["monetified_sample_photo"] = torch.reshape(gen_ptm(sample_photo.to(device)).to("cpu"), [3,256,256])
    Utility.update_req_grad([gen_ptm], True)
    # save statistics
    statistics.append(epoch_statistics)
    Utility.save_checkpoint(model_state, f"{name}_{epoch+1}")
    print(f"Epoch {epoch+1} completed")

    gen_lr_sched.step()
    disc_lr_sched.step()
Utility.save_statistics(name, statistics)