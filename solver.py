from tqdm import tqdm
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import optimizers, metrics

from src import losses
from src.loader import DataLoader
from src.gene import Generator
from src.disc import Discriminator
from src.losses import (
    generator_loss,
    discriminator_loss,
    cycle_loss,
    identity_loss
)


class Solver:
    def __init__(self, config):
        self.mode = config["mode"]
        self.data_config = config["dataset"]
        self.aug_config = config["augmentation"]
        self.get_dataset()

        training_config = config["training"]
        self.epochs = training_config["epochs"]
        self.print_iter = training_config["print_iter"]
        self.save_epoch = training_config["save_epoch"]

        self.gene_AB = Generator()
        self.gene_BA = Generator()
        self.disc_A = Discriminator()
        self.disc_B = Discriminator()

        self.gene_AB_opt = optimizers.Adam(**training_config["g_optimizer"])
        self.gene_BA_opt = optimizers.Adam(**training_config["g_optimizer"])
        self.disc_A_opt = optimizers.Adam(**training_config["d_optimizer"])
        self.disc_B_opt = optimizers.Adam(**training_config["d_optimizer"])

        self.gene_AB_losses = metrics.Mean()
        self.gene_BA_losses = metrics.Mean()
        self.disc_A_losses = metrics.Mean()
        self.disc_B_losses = metrics.Mean()

        self.test_gene_AB_losses = metrics.Mean()
        self.test_gene_BA_losses = metrics.Mean()
        self.test_disc_A_losses = metrics.Mean()
        self.test_disc_B_losses = metrics.Mean()

        self.get_ckpt_manager(training_config["save_path"])

    def train(self):
        n_iters = len(self.dataset)
        valid_loss = 1e6
        for epoch in range(self.epochs):
            self.resets()
            for (i, (ldct, ndct)) in enumerate(self.dataset):
                self.train_batch(ldct, ndct)

                if (i+1) % self.print_iter == 0:
                    print(f"[{epoch+1}/{self.epochs}] Epoch, [{i+1}/{n_iters}] Iter")
                    print(f"Generator(LDCT->NDCT) Loss: {self.gene_AB_losses.result():.5f}", end="  ")
                    print(f"Generator(NDCT->LDCT) Loss: {self.gene_BA_losses.result():.5f}")
                    print(f"Discriminator(LDCT) Loss: {self.disc_A_losses.result():.5f}", end="  ")
                    print(f"Discriminator(NDCT) Loss: {self.disc_B_losses.result():.5f}")

            self.test_steps(True)
            print(f"===== Validation [{epoch+1}/{self.epochs}] Epoch")
            print(f"===== Generator(LDCT->NDCT) Loss: {self.test_gene_AB_losses.result():.5f}", end="  ")
            print(f"Generator(NDCT->LDCT) Loss: {self.test_gene_BA_losses.result():.5f}")
            print(f"===== Discriminator(LDCT) Loss: {self.test_disc_A_losses.result():.5f}", end="  ")
            print(f"Discriminator(NDCT) Loss: {self.test_disc_B_losses.result():.5f}")

            valid_total = self.test_gene_AB_losses.result().numpy()
            if valid_total < valid_loss:
                print(f"Validation loss reduced from {valid_loss:.5f} to {valid_total:.5f}")
                valid_loss = valid_total
                ckpt_save_path = self.ckpt_manager.save()

            if (epoch+1) % self.save_epoch == 0:
                f = self.config["training"]["save_path"] / Path(f"epoch_{epoch+1}.h5")
                self.model.save_weights(str(f))
                print("save ", str(f))

    def test(self, ep=None):
        self.load_weight()
        self.test_steps(False)
        raise NotImplementedError

    def get_dataset(self):
        if self.mode == "train":
            self.dataset = DataLoader(
                self.mode,
                **self.data_config,
                train_pair=False
            )
            self.dataset.set_params(**self.aug_config)
            self.valid_dataset = DataLoader(
                "valid",
                **self.data_config,
                train_pair=True
            )
        if self.mode == "test":
            self.dataset = DataLoader(
                self.mode,
                **self.data_config,
                train_pair=True
            )

    def get_ckpt_manager(self, ckpt_path, keep=5):
        self.ckpt = tf.train.Checkpoint(
            gene_AB=self.gene_AB, gene_BA=self.gene_BA,
            disc_A=self.disc_A, disc_B=self.disc_B,
            gene_AB_opt=self.gene_AB_opt, gene_BA_opt=self.gene_BA_opt,
            disc_A_opt=self.disc_A_opt, disc_B_opt=self.disc_B_opt
        )
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, ckpt_path, max_to_keep=keep)

    def resets(self):
        self.gene_AB_losses.reset_states()
        self.gene_BA_losses.reset_states()
        self.disc_A_losses.reset_states()
        self.disc_B_losses.reset_states()
        self.test_gene_AB_losses.reset_states()
        self.test_gene_BA_losses.reset_states()
        self.test_disc_A_losses.reset_states()
        self.test_disc_B_losses.reset_states()

    def load_weight(self):
        f = str(self.config["training"]["save_path"])
        self.ckpt.restore(tf.train.latest_checkpoint(f)).expect_partial()

    @tf.function
    def train_batch(self, A, B):
        with tf.GradientTape(persistent=True) as tape:
            fake_B = self.gene_AB(A)
            fake_A = self.gene_BA(B)

            cycle_A = self.gene_BA(fake_B)
            cycle_B = self.gene_AB(fake_A)

            iden_A = self.gene_BA(A)
            iden_B = self.gene_AB(B)

            real_A_logit = self.disc_A(A)
            fake_A_logit = self.disc_A(fake_A)
            real_B_logit = self.disc_B(B)
            fake_B_logit = self.disc_B(fake_B)

            AB_loss = generator_loss(fake_B_logit)
            BA_loss = generator_loss(fake_A_logit)
            iden_A_loss = identity_loss(A, iden_A)
            iden_B_loss = identity_loss(B, iden_B)
            c_loss = cycle_loss(A, cycle_A) + cycle_loss(B, cycle_B)

            gene_AB_loss = AB_loss + iden_A_loss + c_loss
            gene_BA_loss = BA_loss + iden_B_loss + c_loss
            disc_A_loss = discriminator_loss(real_A_logit, fake_A_logit)
            disc_B_loss = discriminator_loss(real_B_logit, fake_B_logit)

        gene_AB_grads = tape.gradient(gene_AB_loss, self.gene_AB.trainable_variables)
        gene_BA_grads = tape.gradient(gene_BA_loss, self.gene_BA.trainable_variables)
        disc_A_grads = tape.gradient(disc_A_loss, self.disc_A.trainable_variables)
        disc_B_grads = tape.gradient(disc_B_loss, self.disc_B.trainable_variables)

        self.gene_AB_opt.apply_gradients(zip(gene_AB_grads, self.gene_AB.trainable_variables))
        self.gene_BA_opt.apply_gradients(zip(gene_BA_grads, self.gene_BA.trainable_variables))
        self.disc_A_opt.apply_gradients(zip(disc_A_grads, self.disc_A.trainable_variables))
        self.disc_B_opt.apply_gradients(zip(disc_B_grads, self.disc_B.trainable_variables))

        self.gene_AB_losses(gene_AB_loss)
        self.gene_BA_losses(gene_BA_loss)
        self.disc_A_losses(disc_A_loss)
        self.disc_B_losses(disc_B_loss)

    @tf.function
    def test_batch(self, A, B):
        fake_B = self.gene_AB(A)
        fake_A = self.gene_BA(B)

        cycle_A = self.gene_BA(fake_B)
        cycle_B = self.gene_AB(fake_A)

        iden_A = self.gene_BA(A)
        iden_B = self.gene_AB(B)

        real_A_logit = self.disc_A(A)
        fake_A_logit = self.disc_A(fake_A)
        real_B_logit = self.disc_B(B)
        fake_B_logit = self.disc_B(fake_B)

        AB_loss = generator_loss(fake_B_logit)
        BA_loss = generator_loss(fake_A_logit)
        iden_A_loss = identity_loss(A, iden_A)
        iden_B_loss = identity_loss(B, iden_B)
        c_loss = cycle_loss(A, cycle_A) + cycle_loss(B, cycle_B)

        gene_AB_loss = AB_loss + iden_A_loss + c_loss
        gene_BA_loss = BA_loss + iden_B_loss + c_loss
        disc_A_loss = discriminator_loss(real_A_logit, fake_A_logit)
        disc_B_loss = discriminator_loss(real_B_logit, fake_B_logit)

        self.test_gene_AB_losses(gene_AB_loss)
        self.test_gene_BA_losses(gene_BA_loss)
        self.test_disc_A_losses(disc_A_loss)
        self.test_disc_B_losses(disc_B_loss)

    def test_steps(self, valid=True):
        if valid:
            datasets = self.valid_dataset
        else:
            datasets = tqdm(self.dataset)
        for ldct, ndct in datasets:
            self.test_batch(ldct, ndct)
