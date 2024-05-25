import os
import torch
import random
import argparse
from EncDec import *
from DLA3.EncDec.changerollno import *
from torch.utils.data import DataLoader


P = argparse.ArgumentParser()
P.add_argument("gpu", type=str)
P.add_argument("bonus", type=str)
A = P.parse_args()
    

if __name__ == "__main__":

    Data = DataLoader(dataset=AlteredMNIST(),
                      batch_size=BATCH_SIZE,
                      shuffle=True,
                      num_workers=2,
                      drop_last=True,
                      pin_memory=True)
    E = Encoder()
    D = Decoder()
    
    L = [AELossFn(),
         VAELossFn()]
    
    O = torch.optim.Adam(ParameterSelector(E, D), lr=LEARNING_RATE)
    
    print("Training Encoder: {}, Decoder: {} on Modified MNIST dataset in AE training paradigm".format(
        E.__class__.__name__,
        D.__class__.__name__,
    ))
    AETrainer(Data,
              E,
              D,
              L[0],
              O,
              A.gpu)
    
    print("Training Encoder: {}, Decoder: {} on Modified MNIST dataset in VAE training paradigm".format(
        E.__class__.__name__,
        D.__class__.__name__,
    ))
    VAETrainer(Data,
               E,
               D,
               L[1],
               O,
               A.gpu)
    
    print("AE, VAE Training Complete")
    
    if A.bonus == "T":
        CL = CVAELossFn()
        CVAE_Trainer(Data,
                     E,
                     D,
                     CL,
                     O)
    else:
        print("Bonus Question not done")
        
    AE_Pipeline = AE_TRAINED(gpu=False)
    VAE_Pipeline = VAE_TRAINED(gpu=False)
    
    
    
    """ For TAs Only """
    
    # TestData = TestMNIST()
    # AESSIM, VAESSIM = [], []
    # AEPSNR, VAEPSNR = [], []
    # for sample, original in TestData:
    #     AESSIM.append(AE_Pipeline.from_path(sample, original, type="SSIM"))
    #     VAESSIM.append(VAE_Pipeline.from_path(sample, original, type="SSIM"))
    #     AEPSNR.append(AE_Pipeline.from_path(sample, original, type="PSNR"))
    #     VAEPSNR.append(VAE_Pipeline.from_path(sample, original, type="PSNR"))
    
    # print("SSIM Score of AutoEncoder Training paradigm: {}".format(sum(AESSIM)/len(AESSIM)))
    # print("SSIM Score of Variational AutoEncoder Training paradigm: {}".format(sum(VAESSIM)/len(VAESSIM)))
    # print("PSNR Score of AutoEncoder Training paradigm: {}".format(sum(AEPSNR)/len(AEPSNR)))
    # print("PSNR Score of Variational AutoEncoder Training paradigm: {}".format(sum(VAEPSNR)/len(VAEPSNR)))
    
    # if A.bonus == "T":
    #     Generator = CVAE_Generator()
    #     for _ in range(24):
    #         Generator.save_image(digit=random.choice(range(10)), save_path=SAVEPATH)
    #
    #     print("Similarity Score for generated data is: {}".format(GeneratorSimilarity(SAVEPATH)))
