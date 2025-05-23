import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Network takes both x and n as inputs
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x, n, maxN):
      # scaling
        n_scaled = torch.clamp(n / maxN, 0.0, 1.0)  # Clamped to prevent extreme values
        inputs = torch.cat([x, n_scaled * torch.ones_like(x)], dim=1)
        psi = self.net(inputs)
        return psi

class Inference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_default_dtype(torch.float64)

        self.MAX_SAMPLES_TRAINED = 10
        self.model = PINN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=2000, min_lr=1e-6)

    def train_pinn(self, epochs=50000, L=1.0, h_bar=1.0, m=1.0, sample=10):
        # Collocation points and range of n values
        x = torch.linspace(0, L, 750, device=self.device).view(-1, 1).requires_grad_(True)
        n_values = torch.linspace(1, sample, int(sample), device=self.device).float()  # Sample n from 1 to sample_size

        best_loss = float('inf')
        best_model = None
        loss_history = []
        pde_loss_history = []
        norm_loss_history = []
        bc_loss_history = []

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            total_loss = 0.0
            pde_loss_total = 0.0
            norm_loss_total = 0.0
            bc_loss_total = 0.0

            for n in n_values: # running for each and every different n values within [1, sample_size (say 10)]
                n_tensor = n.unsqueeze(0)

                psi = self.model(x, n_tensor, sample)

                # boundary conditions
                psi_0 = self.model(torch.tensor([[0.0]], device=self.device), n_tensor, sample)
                psi_L = self.model(torch.tensor([[L]], device=self.device), n_tensor, sample)

                # compute derivatives
                dpsi_dx = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
                d2psi_dx2 = torch.autograd.grad(dpsi_dx, x, torch.ones_like(dpsi_dx), create_graph=True)[0]

                # PDE loss
                pde_loss = torch.mean((-(h_bar**2 / (2 * m)) * d2psi_dx2 - (n**2 * torch.pi**2 * h_bar**2 / (2*m)) * psi)**2)

                # Boundary loss
                bc_loss = torch.mean(psi_0**2 + psi_L**2)

                # Normalization loss
                norm = torch.trapz(psi**2, x, dim=0) # integral of psi^2 from 0 to 1  = 1
                norm_loss = (norm - 1)**2

                # Custom total loss : using hyperbolic weighting to control individual losses proportionally
                norm_loss_weight = 1e2 * pde_loss.item() + 1e-4

                bc_loss_weight = 1.0
                if pde_loss.item() < 50.0 and norm_loss.item() < 0.01:
                    bc_loss_weight = 100.0

                loss = pde_loss + bc_loss_weight * bc_loss + norm_loss_weight * norm_loss

                total_loss += loss
                pde_loss_total += pde_loss
                norm_loss_total += norm_loss
                bc_loss_total += bc_loss

            # taking mean

            total_loss /= sample
            pde_loss_total /= sample
            norm_loss_total /= sample
            bc_loss_total /= sample

            if torch.isnan(total_loss):
                print(f"NaN detected at epoch {epoch}")
                break
            
            # usual back prop.

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step(total_loss)

            # logging and plotting

            loss_history.append(total_loss.item())
            pde_loss_history.append(pde_loss_total.item())
            norm_loss_history.append(norm_loss_total.item())
            bc_loss_history.append(bc_loss_total.item())

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_model = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Total Loss: {total_loss.item():.6f}, '
                    f'PDE Loss: {pde_loss_total.item():.6f}, '
                    f'Norm Loss: {norm_loss_total.item():.6f}, '
                    f'BC Loss: {bc_loss_total.item():.6f}')

        self.model.load_state_dict(best_model)

        return self.model, loss_history, pde_loss_history, norm_loss_history, bc_loss_history
    
    def analytic_sol(self, x, n):
        return np.sqrt(2/1) * np.sin(n * np.pi * x / 1)

    def infer(self, type: int, x, N: int, n: int):
        trained_output = self.train_pinn(sample=n+1)
        test_model = trained_output[0]

        with torch.no_grad():
            if type == 2:
                x_test = torch.linspace(0.0, 1.0, int(N)).view(-1, 1).to(self.device)
            else:
                x_test = torch.tensor([x], device=self.device).float().view(-1, 1)
            # x_test = torch.linspace(0.0, 1.0, int(N)).view(-1, 1).to(self.device)
            # n_test = np.random.randint(1, sample+1, (sample,))
            n_test = [int(n)]
            for n in n_test:
                n_tensor = torch.tensor([n], device=self.device).float()
                psi_pred = test_model(x_test, n_tensor, self.MAX_SAMPLES_TRAINED).cpu().numpy()

                psi_true = self.analytic_sol(x_test.cpu().numpy(), n)
                if np.dot(psi_pred.flatten(), psi_true.flatten()) < 0:
                    psi_pred = -psi_pred
                
                plt.figure(figsize=(10, 5))
                
                if type == 2:
                    plt.plot(x_test.cpu(), psi_pred, label='PINN Prediction')
                    plt.plot(x_test.cpu(), psi_true, '--', label='Analytic Solution')
                    plt.title(f'n = {n} (E ≈ {n**2 * np.pi**2 / 2:.4f})')
                    plt.legend()
                    plt.grid()

                else:
                    x_test1 = torch.linspace(0.0, 1.0, 250).view(-1, 1).to(self.device)
                    for n in n_test:
                        n_tensor = torch.tensor([n], device=self.device).float()
                        psi_pred1 = test_model(x_test1, n_tensor, self.MAX_SAMPLES_TRAINED).cpu().numpy()

                        psi_true1 = self.analytic_sol(x_test1.cpu().numpy(), n)
                        if np.dot(psi_pred1.flatten(), psi_true1.flatten()) < 0:
                            psi_pred1 = -psi_pred1

                    plt.plot([x, x], [0, psi_pred.flatten()[0]], 'r')
                    plt.plot([0, x], [psi_pred.flatten()[0], psi_pred.flatten()[0]], 'r')
                    plt.plot(x_test1.cpu(), psi_pred1, label='PINN Prediction')
                    plt.plot(x_test1.cpu(), psi_true1, '--', label='Analytic Solution')
                    plt.title(f'n = {n} (E ≈ {n**2 * np.pi**2 / 2:.4f}) @x = {x} : Soln = {psi_pred.flatten()[0]} and Soln_true = {psi_true.flatten()[0]}')
                    plt.legend()
                    plt.grid()

                plt.savefig(f"media/Images/Out_{n}_{str(type)}.png")

        if type == 1:
            return psi_pred.flatten()[0]
        return 1

    def infer(self, model_path: str, type: int, x, N: int, n: int):
        abs_path = os.path.join(model_path)
        # print(os.getcwd())
        test_model = PINN().to(self.device)
        test_model.load_state_dict(torch.load(abs_path, weights_only=False, map_location="cpu"))

        with torch.no_grad():
            if type == 2:
                x_test = torch.linspace(0.0, 1.0, int(N)).view(-1, 1).to(self.device)
            else:
                x = float(x)
                x_test = torch.tensor([x], device=self.device, dtype=torch.float64).view(-1, 1)
            # n_test = np.random.randint(1, sample+1, (sample,))
            n_test = [int(n)]
            for n in n_test:
                n_tensor = torch.tensor([n], device=self.device).float()
                psi_pred = test_model(x_test, n_tensor, self.MAX_SAMPLES_TRAINED).cpu().numpy()

                psi_true = self.analytic_sol(x_test.cpu().numpy(), n)
                if np.dot(psi_pred.flatten(), psi_true.flatten()) < 0:
                    psi_pred = -psi_pred
                
                plt.figure(figsize=(10, 5))

                if type == 2:
                    plt.plot(x_test.cpu(), psi_pred, label='PINN Prediction')
                    plt.plot(x_test.cpu(), psi_true, '--', label='Analytic Solution')
                    plt.title(f'n = {n} (E ≈ {n**2 * np.pi**2 / 2:.4f})')
                    plt.legend()
                    plt.grid()

                else:
                    x_test1 = torch.linspace(0.0, 1.0, 250).view(-1, 1).to(self.device)
                    for n in n_test:
                        n_tensor = torch.tensor([n], device=self.device).float()
                        psi_pred1 = test_model(x_test1, n_tensor, self.MAX_SAMPLES_TRAINED).cpu().numpy()

                        psi_true1 = self.analytic_sol(x_test1.cpu().numpy(), n)
                        if np.dot(psi_pred1.flatten(), psi_true1.flatten()) < 0:
                            psi_pred1 = -psi_pred1

                    plt.plot([x, x], [0, psi_pred.flatten()[0]], 'r')
                    plt.plot([0, x], [psi_pred.flatten()[0], psi_pred.flatten()[0]], 'r')
                    plt.plot(x_test1.cpu(), psi_pred1, label='PINN Prediction')
                    plt.plot(x_test1.cpu(), psi_true1, '--', label='Analytic Solution')
                    plt.title(f'n = {n} (E ≈ {n**2 * np.pi**2 / 2:.4f}) @x = {x} : Soln = {psi_pred.flatten()[0]} and Soln_true = {psi_true.flatten()[0]}')
                    plt.legend()
                    plt.grid()

                plt.savefig(f"media/Images/Out_{n}_{str(type)}.png")
        
        if type == 1:
            return psi_pred.flatten()[0]
        return 1

        
        
