# The Sandwich Rule: Mixed-Precision Quantization

## Problem: Information Collapse

In our initial experiments, **fully quantizing** the entire model (including the **embedding layer** and the **LM head / output projection**) led to **Information Collapse**: high perplexity and incoherent text. The input and output boundaries are especially sensitive because:

- **Embedding**: Maps discrete token IDs to a continuous space. Aggressive quantization here destroys the representational capacity at the very first step.
- **LM Head**: Maps the last hidden state to vocabulary logits. Quantizing this layer loses the fine-grained distinctions needed to choose the next token.

## Solution: Sandwich Rule

Keep **high precision** at the **edges** and use **ternary (or binary)** in the **middle**:

```
  [Embedding]     →  F16 / 8-bit   (high precision)
       ↓
  [Decoder Layer 0..N]  →  Ternary / Binary  (1.58-bit / 1-bit)
       ↓
  [LM Head]       →  F16 / 8-bit   (high precision)
```

- **Input (Embedding)** and **Output (LM Head)** stay in **Float16** or **EightBit**.
- **Hidden layers** (attention + FFN) use **Ternary** (or Binary) for ~20× compression vs FP16.

This gives a model that is **small enough for embedded devices** (e.g. ESP32) while **smart enough to generate coherent text**.

## Implementation

- **QuantMode** enum: `F16`, `EightBit`, `Ternary`, `Binary`.
- **LayerBitMap**: JSON that specifies per-layer (and embedding / lm_head) bit-width. The **search** crate produces this; **inference** and **training** consume it.
- **Default**: Embedding and LM Head = `F16`, all decoder layers = `Ternary`.

## References

- BitNet b1.58: Ternary weights {-1, 0, +1}.
- Annealing + Latent Clipping: Training latent F32 weights with soft→hard quantization and STE.
