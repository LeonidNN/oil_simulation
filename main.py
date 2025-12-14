import math
import numpy as np
import torch
import reservoir_core as core

HUMAN_RAW = np.array([5271, 5114, 5078, 5187, 3500], dtype=np.float64)

N_GLOBAL = 50_000
M_ASCENTS = 100
N_LOCAL = 3000
KEEP_TOP = 8000

EPOCHS = 35
BATCH = 256
LR = 3e-3

OPT_STEPS = 90
OPT_LR = 0.25

torch.set_default_dtype(torch.float64)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 64, bias=True)
        self.fc2 = torch.nn.Linear(64, 64, bias=True)
        self.fc3 = torch.nn.Linear(64, 1, bias=True)
        self.relu = torch.nn.ReLU()

        g = torch.Generator().manual_seed(123)
        for p in self.parameters():
            p.data.uniform_(-0.08, 0.08, generator=g)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)


def train_surrogate(Z, S):
    idx = np.argpartition(-S, min(KEEP_TOP, len(S) - 1))[: min(KEEP_TOP, len(S))]
    Zt = torch.from_numpy(Z[idx])
    St = torch.from_numpy(S[idx])

    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    rng = np.random.default_rng(42)
    order = np.arange(len(Zt))

    for ep in range(1, EPOCHS + 1):
        rng.shuffle(order)
        for s in range(0, len(order), BATCH):
            b = order[s : s + BATCH]
            xb = Zt[b]
            yb = St[b]

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = 0.5 * torch.mean((pred - yb) ** 2)
            loss.backward()
            opt.step()

        if ep % 5 == 0:
            m = min(600, len(Zt))
            with torch.no_grad():
                pred = model(Zt[:m])
                mse = torch.mean((pred - St[:m]) ** 2).item()
            print(f"epoch {ep}/{EPOCHS} mse~{mse}")

    return model


def optimize_with_mlp(model, start_raw):
    z0 = core.to_unit(np.asarray(start_raw, dtype=np.float64))
    z0 = np.clip(z0, 1e-6, 1.0 - 1e-6)

    u0 = np.log(z0 / (1.0 - z0))
    u = torch.tensor(u0, requires_grad=False)

    m = torch.zeros(5)
    v = torch.zeros(5)
    t = 0
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8

    for _ in range(OPT_STEPS):
        t += 1
        u_ = u.detach().clone().requires_grad_(True)
        x = torch.sigmoid(u_)
        y = model(x)

        dy_dx = torch.autograd.grad(y, x, retain_graph=False, create_graph=False)[0]
        dy_du = dy_dx * x * (1.0 - x)

        with torch.no_grad():
            m = b1 * m + (1.0 - b1) * dy_du
            v = b2 * v + (1.0 - b2) * (dy_du * dy_du)

            mhat = m / (1.0 - (b1 ** t))
            vhat = v / (1.0 - (b2 ** t))
            u = u + OPT_LR * mhat / (torch.sqrt(vhat) + eps)

    xfin = torch.sigmoid(u).detach().cpu().numpy().astype(np.float64)
    raw = core.to_raw(xfin)
    return np.asarray(raw, dtype=np.float64)


def pct(a, b):
    if abs(a) < 1e-12:
        return math.inf
    return (b - a) / a * 100.0


def main():
    print(f"=== Stage 1+2: sampling ===")
    Z, S, useM = core.generate_samples(N_global=N_GLOBAL, M_ascents=M_ASCENTS, N_local_per_ascent=N_LOCAL, B=32)
    Z = np.asarray(Z, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    print(f"Total samples: {len(S)} (ascents used: {useM})")

    best_i = int(np.argmax(S))
    best_z = Z[best_i]
    best_raw = np.asarray(core.to_raw(best_z), dtype=np.float64)
    best_res = core.simulate(best_raw)
    print(f"Best from samples: S={best_res['S']}")

    print(f"=== Stage 3: train surrogate on top-{min(KEEP_TOP, len(S))} ===")
    model = train_surrogate(Z, S)

    print(f"=== Stage 4: refine by gradient ascent ===")
    nn_raw = optimize_with_mlp(model, best_raw)
    nn_res = core.simulate(nn_raw)

    human_res = core.simulate(HUMAN_RAW)

    print("\n=== MAIN COMPARISON (by S only) ===")
    print(f"S_human = {human_res['S']}")
    print(f"S_AI    = {nn_res['S']}")
    print(f"Delta S = {nn_res['S'] - human_res['S']}")
    print(f"AI vs Human improvement = {pct(human_res['S'], nn_res['S'])} %")

    print("\nAI oper_con:", nn_raw)


if __name__ == "__main__":
    main()
