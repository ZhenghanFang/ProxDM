def infinite_loop(dataloader):
    while True:
        for x in dataloader:
            yield x
