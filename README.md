# FGAI Assignment 2: Text Generation Using Recurrent Neural Networks

## 1. Problem Statement and Objective

The objective of this assignment is to implement and compare recurrent neural network architectures for next-token prediction and text generation. The models learn patterns from Shakespearean text and generate new text based on a given seed sequence.

The architectures evaluated are:
- Simple Recurrent Neural Network (Simple RNN)
- Bidirectional RNN
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## 2. Dataset

**Dataset:** Tiny Shakespeare

**Source:** https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

The dataset contains excerpts from Shakespeare's works, including dialogue and character names. The text is preprocessed and converted into token sequences for next-token prediction.

## 3. Model Descriptions

- **Simple RNN:** Processes sequential input using a recurrent hidden state.
- **Bidirectional RNN:** Processes sequences in forward and reverse directions to capture information from both directions.
- **LSTM:** Uses a memory cell and gates to learn longer-term dependencies.
- **GRU:** Uses update and reset gates to control information flow, with a simpler structure than LSTM.

## 4. Methodology

The project follows these main steps:

1. Load and preprocess the Tiny Shakespeare dataset.
2. Build a vocabulary and convert text into token IDs.
3. Prepare training, validation, and test sequences.
4. Train each of the four recurrent architectures.
5. Evaluate the trained models using test loss, accuracy, perplexity, and training time.
6. Generate text from a common seed sequence and compare sampling temperatures.
7. Save model checkpoints, evaluation tables, training histories, plots, and generated samples.

The notebook contains the implementation and experimental workflow.

## 5. Model Comparison

| Model | Test Loss | Test Accuracy (%) | Perplexity | Training Time (s) |
|---|---:|---:|---:|---:|
| Simple RNN | 6.6830 | 13.4244 | 798.73 | 496.82 |
| Bidirectional RNN | 7.1653 | 11.5404 | 1293.72 | 760.23 |
| LSTM | 6.5996 | 14.0563 | 734.81 | 503.17 |
| GRU | 6.7546 | 13.1694 | 857.97 | 514.44 |

### Interpretation

The LSTM achieved the lowest test loss and perplexity and the highest test accuracy among the four models. The Simple RNN had the shortest recorded training time. The Bidirectional RNN had the highest test loss and perplexity and the longest training time.

## 6. Text Generation and Sampling Temperature

The models were evaluated using different sampling temperatures: **0.5, 0.8, and 1.2**, with the seed text `To be or not to be`.

- **Temperature 0.5:** Produces more predictable text, but may increase repetition.
- **Temperature 0.8:** Provides a balance between predictable and varied word selection.
- **Temperature 1.2:** Produces more diverse word choices, but may reduce coherence.

The generation statistics and generated text samples are saved in the processed results directory. The results show that diversity and repetition vary by model and temperature. Higher diversity does not necessarily imply better grammatical coherence.

## 7. Repository Structure

```text
FGAI-Assignment-2-Text-Generation/
├── Fgai_assignment.ipynb
├── README.md
├── .gitignore
├── models/
│   ├── simple_rnn.pt
│   ├── bidirectional_rnn.pt
│   ├── lstm_model.pt
│   └── gru_model.pt
└── data/
    └── processed/
        ├── model_comparison.csv
        ├── generation_statistics.csv
        ├── text_generation_results.csv
        ├── simple_rnn_history.csv
        ├── bidirectional_rnn_history.csv
        ├── lstm_history.csv
        ├── gru_history.csv
        ├── metadata.json
        ├── vocabulary.json
        ├── cleaned_text.txt
        ├── test_accuracy.png
        ├── test_loss.png
        ├── test_perplexity.png
        └── training_time.png
```

## 8. Requirements and Execution

Recommended environment: Python 3.11.

Main dependencies include:
- PyTorch
- NumPy
- Matplotlib
- Jupyter Notebook

Install the dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy matplotlib jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `Fgai_assignment.ipynb` and run the cells in order.

**Note:** The notebook may use additional libraries or expect particular data files. Install any additional dependencies required by its imports, and ensure the dataset and preprocessing steps are available before rerunning the entire notebook. The saved checkpoints and results are provided for reference.

## 9. Conclusion

The experiments indicate that LSTM performed best among the evaluated architectures on the reported test metrics. The Simple RNN trained fastest, while the Bidirectional RNN required the most training time and performed least favorably on the reported test metrics.

The temperature experiments demonstrate the trade-off between repetition and diversity in generated text. Overall, the project compares four recurrent architectures for next-token prediction and explores their effectiveness for text generation using the Tiny Shakespeare dataset.