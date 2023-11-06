This repo contains the code for the submitted paper: Prototype Learning for Explainable Brain Age Prediction.

The key elements are:
    - ExPeRT_arch.py: contains the model architecture of the proposed model
    - lightning_module_ExPert.py: Pytorch lightning module for training the ExPeRT architecture
    - loss_function.py: contains the loss functions used to train the ExPeRT model architecture (the metric and consistency loss)
    - optimal_transport_helpers.py: contains the OT matching implementation