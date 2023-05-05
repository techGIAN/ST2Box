from modules.Trainer import STsim_Trainer
import torch
import os

if __name__ == '__main__':

    STsim = STsim_Trainer()

    load_model_name = None
    load_optimizer_name = None 

    STsim.ST_train(load_model=load_model_name, load_optimizer=load_optimizer_name)
