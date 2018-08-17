import itertools
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from Mayo_REDCNN_dataloader import train_dcm_data_loader
from RED_CNN_util import build_dataset

os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class Residual_block(nn.Module):
    def __init__(self, in_feature):
        super(Residual_block, self).__init__()
        conv_block = [nn.ReflectionPad2d(1), nn.Conv2d(in_feature, in_feature, 3), nn.InstanceNorm2d(in_feature), nn.ReLU(inplace=True), nn.ReflectionPad2d(1), nn.Conv2d(in_feature, in_feature, 3), nn.InstanceNorm2d(in_feature)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_res_block):
        super(Generator, self).__init__()
        # initial conv block
        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True)]

        # downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        # residual block
        for _ in range(n_res_block):
            model += [Residual_block(in_features)]
        # upsampling
        out_features = in_features // 2

        model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]
        in_features = out_features
        out_features = in_features // 2

        model += [nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1), nn.InstanceNorm2d(out_features), nn.ReLU(inplace=True)]

        # output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        model = [nn.Conv2d(input_nc, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]

        for ic, oc in [(64,128),(128,256),(256,512)]:
            model += [nn.Conv2d(ic, oc, 4, stride=2, padding=1), nn.InstanceNorm2d(oc), nn.LeakyReLU(0.2, inplace=True)]

        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_nc = 1 # input channel
    output_nc = 1 # output channel
    NUM_EPOCH = 500
    BATCH_SIZE = 3
    CROP_NUMBER = 50
    LR = 0.0002
    DECAY_EPOCH = 100
    N_CPU = 20
    CROP_SIZE = 55

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

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    optimizer_G = torch.optim.Adam(itertools.chain(Gene_AB.parameters(), Gene_BA.parameters()), lr=LR, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(Disc_A.parameters(), lr=LR, betas=(0.5,0.999))
    optimizer_D_B = torch.optim.Adam(Disc_B.parameters(), lr=LR, betas=(0.5,0.999))

    Tensor = torch.cuda.FloatTensor
    input_A = Tensor(BATCH_SIZE*CROP_NUMBER, input_nc, CROP_SIZE, CROP_SIZE)
    input_B = Tensor(BATCH_SIZE*CROP_NUMBER, output_nc, CROP_SIZE, CROP_SIZE)
    target_real = Variable(Tensor(BATCH_SIZE*CROP_NUMBER).fill_(1.0), requires_grad=False)
    target_real = target_real.reshape(-1,1)
    target_fake = Variable(Tensor(BATCH_SIZE*CROP_NUMBER).fill_(0.0), requires_grad=False)
    target_fake = target_fake.reshape(-1,1)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()



    #data_path = '/data1/AAPM-Mayo-CT-Challenge/'

    input_dir, target_dir, test_input_dir, test_target_dir = build_dataset('L506', "3mm", norm_range=(-1024.0, 3072.0))
    train_dcm = train_dcm_data_loader(input_dir, target_dir, crop_size=CROP_SIZE, crop_n=CROP_NUMBER)
    train_loader = DataLoader(train_dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CPU)

    #logger = Logger(NUM_EPOCH, len(train_loader))

    list_loss_id_A = []
    list_loss_id_B = []
    list_loss_gan_AB = []
    list_loss_gan_BA = []
    list_loss_cycle_ABA = []
    list_loss_cycle_BAB = []
    list_loss_D_B = []
    list_loss_D_A = []
    for epoch in range(NUM_EPOCH):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.reshape(-1, 1, 55, 55).to(device)
            targets = targets.reshape(-1, 1, 55, 55).to(device)

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

            list_loss_id_A.append(loss_identity_A)
            list_loss_id_B.append(loss_identity_B)
            list_loss_gan_AB.append(loss_GAN_AB)
            list_loss_gan_BA.append(loss_GAN_BA)
            list_loss_cycle_ABA.append(loss_cycle_ABA)
            list_loss_cycle_BAB.append(loss_cycle_BAB)
            list_loss_D_A.append(loss_D_A)
            list_loss_D_B.append(loss_D_B)

            if (i + 1) % 10 == 0:
                print("EPOCH [{}/{}], STEP [{}/{}]".format(epoch+1, NUM_EPOCH, i+1, len(train_loader)))
                print("Loss G: {}, \nLoss_G_identity: {}, \nLoss_G_GAN: {}, \nLoss_G_cycle: {}, \nLoss D: {}".format(loss_G, (loss_identity_A+loss_identity_B), (loss_GAN_AB+loss_GAN_BA), (loss_cycle_ABA+loss_cycle_BAB), (loss_D_A+loss_D_B)))


        if (epoch + 1) % 10 == 0:
            torch.save(Gene_AB.state_dict(), '/home/shsy0404/result/cycleGAN_result/{}_cyclegan_patch_{}ep.ckpt'.format("L506", epoch+1))
            torch.save(Gene_BA.state_dict(), '/home/shsy0404/result/cycleGAN_result/{}_cyclegan_patch_{}ep.ckpt'.format("L506", epoch + 1))
            torch.save(Disc_A.state_dict(), '/home/shsy0404/result/cycleGAN_result/{}_cyclegan_patch_{}ep.ckpt'.format("L506", epoch + 1))
            torch.save(Disc_B.state_dict(), '/home/shsy0404/result/cycleGAN_result/{}_cyclegan_patch_{}ep.ckpt'.format("L506", epoch + 1))

            result_loss = {'id_A':list_loss_id_A, 'id_B':list_loss_id_B, 'gan_AB':list_loss_gan_AB, 'gan_BA':list_loss_gan_BA, 'cycle_ABA':list_loss_cycle_ABA, 'cycle_BAB':list_loss_cycle_BAB, 'disc_A':list_loss_D_A, 'disc_B':list_loss_D_B}

            with open('/home/shsy0404/result/cycleGAN_result/{}_losslist'.format(epoch+1), 'wb') as f:
                pickle.dump(result_loss, f)


if __name__ == "__main__":
    main()
