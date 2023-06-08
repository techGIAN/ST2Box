from modules.Trainer import STsim_Trainer
import torch
import os

if __name__ == '__main__':
    
    STsim = STsim_Trainer()

    load_model_name = '' # Fill this in with the model and optimizer name.
    load_optimizer_name = '' 

    STsim.ST_eval(load_model=load_model_name)

