import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from model import VAE_CNN_Improved

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loss_function(x, x_recon, mu, logvar):
    x_recon = x_recon.detach().cpu()
    MSE = F.mse_loss(x_recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


def save_model(model, path):
    torch.save(model.state_dict(), path)

def trainer(train_loader, val_loader, config):
    device = torch.device(config["device"])
    model = VAE_CNN_Improved(latent_dim=config["latent_dim"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_history, latent_history = [], []
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        for grey_img, img in train_loader:
            grey_img = grey_img.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(grey_img)
            loss = loss_function(img, x_recon, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_loss = train_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{config["num_epochs"]}, Loss: {avg_loss:.4f}")

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                latent = []
                for grey_img, img in val_loader:
                    grey_img = grey_img.to(device)
                    x_recon, mu, logvar, z = model(grey_img)
                    latent.append(z)
                latent_history.append(torch.cat(latent))
    save_model(model, "./model.pth")
    return loss_history, latent_history
