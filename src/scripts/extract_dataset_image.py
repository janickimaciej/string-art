import numpy as np
import matplotlib.pyplot as plt

path = "artifacts/generated_data/dataset/train/shard_00000.npz"

with np.load(path) as data:
    images = data["images"]
    tokens = data["tokens"]
    lengths = data["lengths"]

    print("Images shape:", images.shape)
    print("Image dtype:", images.dtype)
    print("Tokens shape:", tokens.shape)
    print("Lengths shape:", lengths.shape)

    plt.imshow(images[0], cmap="gray", vmin=0, vmax=255)
    plt.title("Training Sample 0")
    plt.axis("off")
    plt.show()
    plt.imsave("artifacts/training_sample_0.png", images[0],
           cmap="gray", vmin=0, vmax=255)
