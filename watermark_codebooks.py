"""Explicit small codebooks and implicit hypercubes with little-endian bit order."""

import math

import torch


def make_key(dimension, bits, k, device, mode="orthogonal"):
    if not 0 < bits <= k <= dimension:
        raise ValueError("Require 0 < bits <= K <= flattened data dimension")
    if mode not in {"auto", "orthogonal", "hypercube"}:
        raise ValueError(mode)
    if mode != "hypercube" and bits > 20:
        raise ValueError("Explicit codebooks above 20 bits are unsupported; use hypercube")
    if mode == "orthogonal" and 2 ** bits > k:
        raise ValueError("Orthogonal encoding requires 2**bits <= K")
    P = torch.linalg.qr(torch.randn(dimension, k, device=device, dtype=torch.float32))[0][:, :k]
    if mode == "hypercube":
        return P, torch.eye(k, device=device)[:bits]
    codes = torch.randn(2 ** bits, k, device=device, dtype=torch.float32)
    if 2 ** bits <= k:
        codes = torch.linalg.qr(codes.T)[0].T
    return P, codes / codes.norm(dim=1, keepdim=True)


def message_code(message, codes, mode):
    if mode == "hypercube":
        signs = codes.new_tensor([2 * int(bit) - 1 for bit in message])
        return signs @ codes / math.sqrt(len(message))
    index = sum(int(bit) << i for i, bit in enumerate(message))
    return codes[index]


def decode_signature(signature, codes, bits, mode):
    scores = signature @ codes.T
    if mode == "hypercube":
        decoded = tuple(int(value > 0) for value in scores)
    else:
        index = int(scores.argmax())
        decoded = tuple((index >> i) & 1 for i in range(bits))
    return decoded, scores


def target_score(scores, message, mode):
    if mode == "hypercube":
        return float(scores @ scores.new_tensor([2 * int(b) - 1 for b in message]) / math.sqrt(len(message)))
    return float(scores[sum(int(bit) << i for i, bit in enumerate(message))])
