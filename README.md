# Predicting Expert Confidence Levels in IPCC Climate Statements with LSTM Networks

**Fernando Nakamoto**

Can a neural language model learn how *certain* a climate statement is, purely from its
text? This project trains sequence models to predict the confidence level that
Intergovernmental Panel on Climate Change (IPCC) scientists assigned to their own findings,
and compares them against simple bag-of-words baselines.

## Motivation

Public discussion of climate change is flooded with claims that range from well-established
science to speculation, and to a non-expert the two can look linguistically identical. The
IPCC communicates the strength of its findings through a standardized *confidence* scale
(`low`, `medium`, `high`, `very high`) that summarizes the quality of the evidence and the
degree of agreement among studies. This project asks whether that expert judgment can be
recovered from the statement text alone — a first step toward tools that flag how
well-established a climate claim really is.

## Task

Single-label, four-class text classification. Given a statement with its confidence
qualifier removed, recover the label the authors assigned:

```
f: statement text  ->  y in {low, medium, high, very high}
```

## Data

- **ClimateX** (Lacombe, Wu & Dilworth, 2023): 8,094 expert-labeled statements extracted or
  paraphrased from the IPCC Sixth Assessment Report (AR6), covering Working Groups I, II and III.
- Predefined split: 7,794 train / 300 test (the test set was manually reviewed by the dataset authors).
- Classes are strongly imbalanced — `high` dominates; `low` and `very high` are rare.
- Statements are short (median ≈ 25 words).
- A local copy is included as `ipcc_statements_dataset.tsv`.

## Approach

1. **Preprocessing** — Keras `Tokenizer` fit on the training set only (vocabulary capped at
   20,000, `<OOV>` for unseen words), sequences padded/truncated to 60 tokens, a stratified
   validation split, and class weights to counteract the imbalance.
2. **Baselines** — a majority-class predictor and a TF-IDF + Logistic Regression classifier
   (unigrams + bigrams), which ignores word order.
3. **Sequence models** — an `Embedding → LSTM → Dense(softmax)` classifier with dropout, and
   a **Bidirectional LSTM** with an otherwise identical architecture, so the comparison
   isolates the effect of bidirectionality.

Because of the imbalance, evaluation reports accuracy against the baselines plus per-class
precision/recall/F1 and confusion matrices.

## Results (held-out 300-statement test set)

| Model                        | Accuracy | Macro-F1 |
|------------------------------|:--------:|:--------:|
| Majority-class floor         |  0.333   |    —     |
| **TF-IDF + Logistic Regression** | **0.493** | **0.451** |
| Bidirectional LSTM           |  0.473   |  0.444   |
| LSTM                         |  0.440   |  0.440   |

**Key finding:** the simplest model wins. TF-IDF + Logistic Regression — which ignores word
order entirely — is the strongest classifier overall, narrowly ahead of the Bi-LSTM. The
sequential structure the LSTMs are built to exploit bought little here: the signal about IPCC
confidence appears to live in word choice and topic vocabulary rather than in word order.
Every model sits only modestly above the majority-class floor, confirming that the cues
separating one confidence level from the next are subtle and largely non-lexical — a task on
which even large language models perform only slightly above chance.

## Repository contents

| File | Description |
|------|-------------|
| `Predicting-Expert-Confidence-IPCC_notebook.ipynb` | Full analysis: data exploration, preprocessing, baselines, LSTM and Bi-LSTM, evaluation. |
| `Predicting-Expert-Confidence-IPCC_notebook.pdf`   | Rendered read-only version of the notebook. |
| `ipcc_statements_dataset.tsv` | The ClimateX statements with confidence labels and the train/test split. |

Trained weights and other large artifacts are not committed (see `.gitignore`); the notebook
regenerates them by retraining, which runs in minutes on a laptop CPU.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install jupyter numpy pandas scikit-learn tensorflow matplotlib
jupyter notebook Predicting-Expert-Confidence-IPCC_notebook.ipynb
```

Run the cells top to bottom; the notebook reads `ipcc_statements_dataset.tsv` from the same folder.

## References

1. Lacombe, R., Wu, K., & Dilworth, E. (2023). *ClimateX: Do LLMs Accurately Assess Human
   Expert Confidence in Climate Statements?* Tackling Climate Change with Machine Learning
   Workshop, NeurIPS 2023. [arXiv:2311.17107](https://arxiv.org/abs/2311.17107) ·
   [dataset](https://huggingface.co/datasets/rlacombe/ClimateX) ·
   [code](https://github.com/rlacombe/ClimateX) (CC-BY-4.0).
2. IPCC (2021–2022). *Sixth Assessment Report (AR6)* — Working Groups I, II and III.
   [ipcc.ch/assessment-report/ar6](https://www.ipcc.ch/assessment-report/ar6/)
3. Mastrandrea, M. D., Field, C. B., Stocker, T. F., et al. (2010). *Guidance Note for Lead
   Authors of the IPCC Fifth Assessment Report on Consistent Treatment of Uncertainties.* IPCC.

## License

Released under the [MIT License](LICENSE) © Fernando Nakamoto. The ClimateX dataset is
distributed by its authors under CC-BY-4.0.
