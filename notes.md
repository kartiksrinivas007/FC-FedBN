The important functions at the moment are:-
1. init_nets
2. get_dataloader

The loaders can be configured to give a particular output size, and we must then modify that one

Where in the code does the fedbn utils need to be added, clearly with the dataloaders.

1. Whaere Dataloader loaded with data?
    inside `get_dataloader()`

2. Where Dataloaders explicitly used?
    `used in main function only?`


What is the public data that is used in this paper and where is it in the code?

The public data is different entirely and you need not even measure the  performance on those ones

Where is the data sent inside the loader, as a whole?
Where is the loader sent into each client?

A further step would be to change the models that are being used for these ones

look at the parameter --structure.


copy fedBN behaviour onto these ones


1. Copy the data onto a particular location
2. Ensure that the `DigitsDataset` class uses that location appropriately
3. Ensure that `prepare_data` has been given the appropriate arguments for evaluation
4. Run and Test.


Batch_size
1. Reset `batch_size` in the original folder itself and copy that change to get_loaders()
2. I reset the arg.batch_size value inside the code
3. 