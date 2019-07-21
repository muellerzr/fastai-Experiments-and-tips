# FastAI-Test-Set-Generation
A working notebook on how to have a labeled test set in Fast.AI

TL;DR:

The fastai library will shuffle the training dataloader when databunched, so we need to swap the two beforehand, then replace our validation dataloader in our Learner to our new test dataloader
