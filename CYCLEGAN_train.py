
# TODO : CycleGAN test

import itertools
import os
import pickle
from time import localtime, strftime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from RED_CNN_util import build_dataset, train_dcm_data_loader
from CYCLEGAN_util import Generator, Discriminator, weights_init_normal, ReplayBuffer, LambdaLR_
from logger import Logger

os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nc = 1 # input channel
    output_nc = 1 # output channel
    START_EPOCH = 0
    DECAY_EPOCH = 50
    NUM_EPOCH = 100
    BATCH_SIZE = 3
    CROP_NUMBER = 10
    CROP_SIZE = 128
    LR = 0.0002
    N_CPU = 30

    save_path = '/home/shsy0404//result/cycleGAN_result/cyclegan_result_128patch/'

    Gene_AB = Generator(input_nc, output_nc, 9)
    Gene_BA = Generator(output_nc, input_nc, 9)
    Disc_A = Discriminator(input_nc)
    Disc_B = Discriminator(output_nc)

    if torch.cuda.device_count() > 1:
        print("Use {} GPUs".format(torch.cuda.device_count()), "=" * 9)
        Gene_AB = nn.DataParallel(Gene_AB)
        Gene_BA = nn.DataParallel(Gene_BA)
        Disc_A = nn.DataParallel(Disc_A)
        Disc_B = nn.DataParallel(Disc_B)

    Gene_AB.to(device)
    Gene_BA.to(device)
    Disc_A.to(device)
    Disc_B.to(device)

    Gene_AB.apply(weights_init_normal)
    Gene_BA.apply(weights_init_normal)
    Disc_A.apply(weights_init_normal)
    Disc_B.apply(weights_init_normal)

    # loss
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # optimizer
    optimizer_G = torch.optim.Adam(itertools.chain(Gene_AB.parameters(), Gene_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(Disc_A.parameters(), lr=LR, betas=(0.5,0.999))
    optimizer_D_B = torch.optim.Adam(Disc_B.parameters(), lr=LR, betas=(0.5,0.999))

    # learning rate schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR_(NUM_EPOCH, START_EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR_(NUM_EPOCH, START_EPOCH, DECAY_EPOCH).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR_(NUM_EPOCH, START_EPOCH, DECAY_EPOCH).step)

    # input & target
    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(BATCH_SIZE*CROP_NUMBER, input_nc, CROP_SIZE, CROP_SIZE)
    input_B = Tensor(BATCH_SIZE*CROP_NUMBER, output_nc, CROP_SIZE, CROP_SIZE)
    target_real = Variable(Tensor(BATCH_SIZE*CROP_NUMBER).fill_(1.0), requires_grad=False)
    target_real = target_real.reshape(-1,1)
    target_fake = Variable(Tensor(BATCH_SIZE*CROP_NUMBER).fill_(0.0), requires_grad=False)
    target_fake = target_fake.reshape(-1,1)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    input_dir, target_dir, test_input_dir, test_target_dir = build_dataset(['L067','L291'], "3mm", norm_range=(-1024.0, 3072.0))
    train_dcm = train_dcm_data_loader(input_dir, target_dir, crop_size=CROP_SIZE, crop_n=CROP_NUMBER)
    train_loader = DataLoader(train_dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU, drop_last=True)

    '''
    Gene_AB.load_state_dict(torch.load(os.path.join(save_path, '/L506_cyclegan_GAB_patch_70ep.ckpt')))
    Gene_BA.load_state_dict(torch.load(os.path.join(save_path, '/L506_cyclegan_GBA_patch_70ep.ckpt')))
    Disc_A.load_state_dict(torch.load(os.path.join(save_path, '/L506_cyclegan_DA_patch_70ep.ckpt')))
    Disc_B.load_state_dict(torch.load(os.path.join(save_path, '/L506_cyclegan_DB_patch_70ep.ckpt')))
    '''

    #logger = Logger(NUM_EPOCH, len(train_loader))
    logger = Logger(os.path.join(save_path, 'logs'))

    list_loss_id_A = []
    list_loss_id_B = []
    list_loss_gan_AB = []
    list_loss_gan_BA = []
    list_loss_cycle_ABA = []
    list_loss_cycle_BAB = []
    list_loss_D_B = []
    list_loss_D_A = []

    step = 0
    for epoch in range(START_EPOCH, NUM_EPOCH):
        print(strftime("%Y-%m-%d %I:%M", localtime()))

        for i, (inputs, targets) in enumerate(train_loader):
            step += 1

            inputs = inputs.reshape(-1, 1, CROP_SIZE, CROP_SIZE).to(device)
            targets = targets.reshape(-1, 1, CROP_SIZE, CROP_SIZE).to(device)

            # set model input
            real_A = Variable(input_A.copy_(inputs))
            real_B = Variable(input_B.copy_(targets))

            ##### Generator AB, BA #####
            optimizer_G.zero_grad()
            # identity loss
            same_B = Gene_AB(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            same_A = Gene_BA(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = Gene_AB(real_A)
            pred_fake = Disc_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake, target_real)
            fake_A = Gene_BA(real_B)
            pred_fake = Disc_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake, target_real)

            # cycle loss
            recovered_A = Gene_BA(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0
            recovered_B = Gene_AB(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_AB + loss_GAN_BA + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()


            ##### Discriminator A #####
            optimizer_D_A.zero_grad()
            # real loss
            pred_real = Disc_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = Disc_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()


            ##### Discriminator B #####
            optimizer_D_B.zero_grad()
            # real loss
            pred_real = Disc_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)
            # fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = Disc_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            # total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            list_loss_id_A.append(loss_identity_A.item())
            list_loss_id_B.append(loss_identity_B.item())
            list_loss_gan_AB.append(loss_GAN_AB.item())
            list_loss_gan_BA.append(loss_GAN_BA.item())
            list_loss_cycle_ABA.append(loss_cycle_ABA.item())
            list_loss_cycle_BAB.append(loss_cycle_BAB.item())
            list_loss_D_A.append(loss_D_A.item())
            list_loss_D_B.append(loss_D_B.item())

            # progress report (http://localhost:8097)
            #logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_AB + loss_GAN_BA), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)}, images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

            ##### Tensorboard Logging #####
            # Log scalar values (scalar summary)
            info = {'identity_A': loss_identity_A.item(), 'identity_B': loss_identity_B.item(), 'gan_AB': loss_GAN_AB.item(), 'gan_BA': loss_GAN_BA.item(), 'cycle_ABA': loss_cycle_ABA.item(), 'cycle_BAB': loss_cycle_BAB.item(), 'disc_A': loss_D_A.item(), 'disc_B': loss_D_B.item()}
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step+1)


            if (i + 1) % 10 == 0:
                print("EPOCH [{}/{}], STEP [{}/{}]".format(epoch+1, NUM_EPOCH, i+1, len(train_loader)))
                print("Loss G: {} \nLoss_G_identity: {} \nLoss_G_GAN: {} \nLoss_G_cycle: {} \nLoss D: {} \nLoss_DA: {} \nLoss DB: {} \n==== \n".format(loss_G, (loss_identity_A+loss_identity_B), (loss_GAN_AB+loss_GAN_BA), (loss_cycle_ABA+loss_cycle_BAB), (loss_D_A+loss_D_B), (loss_D_A), (loss_D_B)))

        if (epoch + 1) % 5 == 0:
            torch.save(Gene_AB.state_dict(), os.path.join(save_path, 'cyclegan_GAB_patch_{}ep.ckpt'.format(epoch+1)))
            torch.save(Gene_BA.state_dict(), os.path.join(save_path, 'cyclegan_GBA_patch_{}ep.ckpt'.format(epoch+1)))
            torch.save(Disc_A.state_dict(), os.path.join(save_path, 'cyclegan_DA_patch_{}ep.ckpt'.format(epoch+1)))
            torch.save(Disc_B.state_dict(), os.path.join(save_path, 'cyclegan_DB_patch_{}ep.ckpt'.format(epoch+1)))

            # loss save per each epoch
            result_loss = {'id_A': list_loss_id_A, 'id_B': list_loss_id_B, 'gan_AB': list_loss_gan_AB, 'gan_BA': list_loss_gan_BA, 'cycle_ABA': list_loss_cycle_ABA, 'cycle_BAB': list_loss_cycle_BAB, 'disc_A': list_loss_D_A, 'disc_B': list_loss_D_B}

            with open(os.path.join(save_path, '{}_losslist.pkl'.format(epoch + 1)), 'wb') as f:
                pickle.dump(result_loss, f)


        # update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()


if __name__ == "__main__":
    main()
