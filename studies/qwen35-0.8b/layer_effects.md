# Which layer decided the answer: direct push vs total effect (24 study chats, answer token vs runner-up)

Direct push = what Chat & trace highlights. Total effect = how much the answer's lead over the runner-up drops when the part is switched off (output -> its average over the text) and everything after it is re-run.

Same top part by both: **1/24** chats

Top part by direct push: attn L22 ×5, mlp L22 ×4, attn L19 ×3, mlp L23 ×3, mlp L21 ×2, attn L20 ×2
Top part by total effect: attn L0 ×7, attn L12 ×3, mlp L13 ×2, mlp L15 ×2, mlp L1 ×1, mlp L0 ×1, mlp L2 ×1, attn L2 ×1
Answer leads (logit lens) from layer: L2 ×2, L9 ×1, L12 ×2, L14 ×1, L15 ×2, L18 ×1, L19 ×1, L21 ×3, L22 ×6, L23 ×5

| Layer | Attention direct | Attention total | MLP direct | MLP total |
|---|---|---|---|---|
| 0 | -0.23 | +1.09 | -0.01 | +1.05 |
| 1 | +0.01 | +0.26 | -0.09 | +0.86 |
| 2 | +0.05 | +0.48 | +0.04 | +1.04 |
| 3 | -0.07 | +0.23 | +0.09 | +0.93 |
| 4 | -0.22 | +0.33 | +0.12 | +0.91 |
| 5 | -0.07 | +0.49 | +0.04 | +1.22 |
| 6 | +0.02 | +0.33 | +0.07 | +0.50 |
| 7 | -0.04 | +0.45 | +0.06 | +0.22 |
| 8 | -0.07 | +0.19 | +0.02 | +0.49 |
| 9 | +0.01 | +0.28 | +0.05 | +0.92 |
| 10 | +0.01 | +0.48 | +0.01 | +0.89 |
| 11 | -0.06 | +0.70 | +0.11 | +0.82 |
| 12 | -0.04 | +0.87 | -0.11 | +0.54 |
| 13 | +0.04 | +0.55 | +0.15 | +0.63 |
| 14 | -0.02 | +0.21 | +0.13 | +0.81 |
| 15 | +0.01 | -0.01 | -0.12 | +0.27 |
| 16 | -0.00 | +0.39 | +0.22 | +0.68 |
| 17 | -0.35 | -0.25 | +0.33 | +0.37 |
| 18 | -0.01 | +0.26 | +0.31 | +0.67 |
| 19 | +0.45 | -0.03 | +0.33 | -0.05 |
| 20 | +0.69 | +0.24 | -0.33 | -0.04 |
| 21 | +0.11 | +0.04 | +0.28 | -0.12 |
| 22 | +0.57 | +0.24 | +0.33 | +0.14 |
| 23 | -0.23 | -0.54 | -0.51 | -1.27 |

| Chat | Answer over runner-up | App highlights (direct) | Biggest total effects | Leads from |
|---|---|---|---|---|
| m01 | '4' over 'Final' (0.8) | attn L19 | attn L12 (+2.8), mlp L11 (+2.6), mlp L4 (+2.2) | L23 |
| m02 | '1' over 'Final' (2.6) | mlp L21 | attn L12 (+4.5), mlp L4 (+4.2), mlp L9 (+3.5) | L21 |
| m03 | '2' over '1' (10.0) | attn L19 | attn L0 (+11.4), attn L15 (+5.1), mlp L14 (+2.7) | L15 |
| m04 | '1' over 'Final' (4.8) | attn L19 | attn L12 (+6.0), mlp L9 (+4.3), attn L11 (+3.8) | L19 |
| f01 | ' Paris' over 'L' (3.6) | attn L20 | mlp L1 (+9.1), mlp L0 (+8.4), attn L6 (+7.6) | L2 |
| f02 | 'Tok' over 'Ky' (1.2) | attn L20 | mlp L13 (+1.7), mlp L18 (+1.6), mlp L9 (+1.4) | L9 |
| f03 | ' Canberra' over ' Sydney' (2.5) | mlp L20 | mlp L15 (+2.8), attn L18 (+2.7), mlp L16 (+2.4) | L15 |
| f04 | ' Ottawa' over 'Toronto' (3.7) | mlp L22 | mlp L0 (+7.0), mlp L1 (+6.4), mlp L4 (+6.1) | L12 |
| r01 | ' No' over ' Yes' (1.4) | mlp L18 | mlp L2 (+3.1), attn L2 (+2.7), mlp L4 (+2.5) | L18 |
| r02 | 'Tom' over 'Ann' (1.2) | attn L22 | attn L2 (+3.7), attn L1 (+1.8), mlp L9 (+1.5) | L22 |
| r03 | '1' over '2' (0.3) | attn L22 | attn L0 (+2.8), attn L10 (+1.0), attn L15 (+1.0) | L22 |
| r04 | '2' over '1' (0.6) | mlp L21 | mlp L6 (+2.1), mlp L5 (+1.9), mlp L15 (+1.9) | L21 |
| l01 | 'Merc' over 'D' (4.3) | mlp L17 | attn L0 (+7.7), attn L16 (+3.4), mlp L17 (+3.3) | L2 |
| l02 | 'Hola' over 'Hello' (5.1) | mlp L23 | attn L0 (+8.4), mlp L23 (+4.6), mlp L4 (+2.8) | L21 |
| l03 | 'Kat' over 'Cat' (0.7) | mlp L23 | attn L0 (+4.2), mlp L0 (+3.9), attn L16 (+3.6) | L23 |
| l04 | 'cold' over 'cool' (1.4) | mlp L23 | attn L0 (+2.2), mlp L23 (+2.0), attn L23 (+1.1) | L23 |
| c01 | '5' over '6' (0.4) | mlp L15 | mlp L15 (+1.0), mlp L14 (+0.9), attn L21 (+0.7) | L14 |
| c02 | '9' over '3' (0.4) | attn L22 | mlp L5 (+7.0), mlp L0 (+6.2), attn L0 (+5.5) | L22 |
| c03 | '3' over '1' (0.8) | attn L22 | attn L15 (+2.2), attn L4 (+2.0), mlp L8 (+1.7) | L22 |
| c04 | 'It' over 'ab' (0.4) | mlp L17 | attn L3 (+3.2), attn L11 (+2.1), mlp L3 (+1.6) | L23 |
| s01 | 'Be' over ' Honey' (0.2) | mlp L22 | attn L5 (+4.5), mlp L18 (+3.0), mlp L16 (+2.6) | L23 |
| s02 | 'A' over 'Sc' (2.0) | mlp L22 | mlp L13 (+2.0), mlp L5 (+1.4), mlp L9 (+1.3) | L22 |
| s03 | 'Blue' over 'Sky' (4.6) | attn L22 | mlp L4 (+2.9), attn L0 (+2.1), attn L10 (+2.1) | L22 |
| s04 | '7' over '5' (1.5) | mlp L22 | attn L0 (+3.0), mlp L20 (+1.2), attn L1 (+1.2) | L12 |
