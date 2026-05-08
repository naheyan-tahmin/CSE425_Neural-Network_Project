import torch
from torch import nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True) #batch_first=True makes input (B*T*E)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        _, (h_n, _) = self.encoder(emb) #_ is the output, (h_n, _) is the hidden state and cell state
        return self.to_latent(h_n[-1]) #h_n[-1] is the last hidden state

    def decode(self, z: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(decoder_input)
        h0 = self.from_latent(z).unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        dec_out, _ = self.decoder(emb, (h0, c0))
        return self.output(dec_out)

    def forward(self, x: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        z = self.encode(x) #encode the input sequence into a latent vector
        return self.decode(z, decoder_input) #decode the latent vector into a sequence
