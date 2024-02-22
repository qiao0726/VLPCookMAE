import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    def prepare_data(self):
        # Download only once
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Split dataset between train and val
        if stage == 'fit' or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
        # Load test dataset for test stage
        if stage == 'test' or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)  # Flatten the input
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x
    
    # Define the optimizer to be used here
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log("test_loss", loss)


if __name__ == '__main__':
    # Initialize the data module and model
    data_module = MNISTDataModule()
    model = LitModel()

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=5, callbacks=[ModelCheckpoint(dirpath='./checkpoints/', save_top_k=1, monitor='val_loss')], accelerator='gpu', devices=-1)

    # Train the model
    trainer.fit(model, data_module)

    # Eval model
    trainer.test(datamodule=data_module)



# # Inference
# model_path = "path/to/your_model.ckpt"  # Update this path to your actual model checkpoint path
# trained_model = LitModel.load_from_checkpoint(checkpoint_path=model_path)
# trained_model.eval()
# trained_model.freeze()  # Optional in PyTorch Lightning to prepare the model for exporting or making it ready for inference
# from PIL import Image
# # Load an image
# image_path = "path/to/your_image.png"
# image = Image.open(image_path).convert("L")  # Convert to grayscale

# # Transform the image to tensor
# transform = transforms.Compose([
#     transforms.Resize((28, 28)),  # Resize to the same size as MNIST images
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))  # Same normalization as during training
# ])

# image = transform(image).unsqueeze(0)  # Add batch dimension

# # Make a prediction
# with torch.no_grad():  # Disable gradient computation for inference
#     prediction = trained_model(image)

# predicted_label = prediction.argmax(dim=1)
# print(f"Predicted Label: {predicted_label.item()}")
