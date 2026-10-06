# **CSC3042F: Grapheme-to-Phoneme (G2P) Translation**

## **Overview**
This project implements an end-to-end Sequence-to-Sequence (Seq2Seq) deep learning pipeline for Grapheme-to-Phoneme (G2P) conversion. The model learns to translate raw English text (characters/graphemes) into pronunciation tokens (ARPAbet phonemes). 

Because English spelling is highly irregular (e.g., *through, dough, cough*), the model cannot rely on hardcoded rules; it must learn statistical phonetic patterns. To fully understand the inner workings of recurrent neural networks, **the LSTM cells and sequence decoders were implemented entirely from scratch in PyTorch** (without relying on `nn.LSTM`). 

The primary goal of this repository is to benchmark how different decoder architectures handle the "information bottleneck" problem inherent to sequence modeling.

## **Architectures Evaluated**
The project evaluates three distinct encoder-decoder configurations:

1. **Bottleneck Baseline (No Context):** 
   A standard Seq2Seq architecture where the encoder's final hidden state initializes the decoder. No further encoder information is provided, forcing the entire input sequence to be compressed into a single vector.
2. **Fixed Context Vector:** 
   The final encoder hidden state is injected into the LSTM gates at *every* decoding time step, giving the decoder a continuous, static view of the global context.
3. **Dynamic Dot-Product Cross-Attention:** 
   The decoder computes a unique context vector at each time step by attending over all encoder hidden states. This allows the model to dynamically "focus" on different characters in the input word depending on the phoneme it is currently predicting.

## **Key Results & Insights**
Models were trained and evaluated on a subset of the **CMU Pronouncing Dictionary (CMUdict)**. Performance was measured using **Phoneme Error Rate (PER)** and exact word accuracy.

* **Attention Solves the Bottleneck:** The Cross-Attention architecture achieved the lowest test PER (0.1173) and the highest word accuracy (59.91%), outperforming both the baseline and fixed-context approaches.
* **Sequence Length Robustness:** Length-bucketed evaluation revealed that the baseline model degrades rapidly on longer inputs (dropping 38.8% in accuracy for words with 11+ characters). The attention mechanism successfully mitigated this degradation, yielding the largest performance gains on the longest, most complex words.

## **Tech Stack**
* **Deep Learning:** PyTorch (custom autograd-compatible LSTM cells, Embeddings, Linear layers)
* **Optimization:** Adam optimizer, Gradient Clipping, Custom Grid Search Hyperparameter Tuning
* **Evaluation & Visualization:** NumPy, Matplotlib (Loss curves, Attention Heatmaps), Edit Distance (Levenshtein)

## **Running the Project**

- Run the following command in your terminal:
``` jupyter notebook```

- Open `model.ipynb`.
- Click the Run All button.
- Make your you have a `dataset` directory with the datasets inside.
