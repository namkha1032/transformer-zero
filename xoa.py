from time import sleep

from tqdm import tqdm

# pbar = tqdm([i for i in range(100)])
# for idx, char in enumerate(pbar):
#     sleep(0.25)
#     if idx % 10 == 0:
#         pbar.set_description("Processing %s" % char)
        
with tqdm(total=100) as pbar:
    for i in range(10):
        sleep(0.1)
        pbar.update(10)