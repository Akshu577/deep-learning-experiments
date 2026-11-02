# Deep Learning & Generative Models: Experiments

**Live page:** https://akshu577.github.io/deep-learning-experiments/

These are six experiments from my MS in Artificial Intelligence (University of the Cumberlands). Each one starts from the reference code in *Generative Deep Learning, 2nd ed.* (David Foster, O'Reilly; [code on GitHub](https://github.com/davidADSP/Generative_Deep_Learning_2nd_Edition), Apache 2.0). I then changed one thing (an activation function, the stride, the training length, the sampling temperature) and measured what happened. The notebooks say exactly what I changed from the book.

Built with TensorFlow / Keras 3.

| # | Experiment | Data | What I changed and what I found |
|---|---|---|---|
| 01 | [MLP activation functions](01-mlp-activations) | Fashion-MNIST | Same network with ReLU, LeakyReLU and Sigmoid. Test accuracy 88.0% / **88.7%** / 88.3%. On a network this shallow, the choice barely matters. |
| 02 | [CNN stride](02-cnn-stride) | CIFAR-10 | Stride 2 vs. stride 3. Accuracy dropped from **72.8%** to 65.5%, because the larger stride throws away spatial detail too early. |
| 03 | [Autoencoder](03-autoencoder) | Fashion-MNIST | Trained a convolutional autoencoder and studied where reconstructions lose detail. |
| 04 | [LSTM text generation](04-lstm-text-generation) | Epicurious recipes | Trained a word-level LSTM (loss 1.52 after 25 epochs) and compared generated recipes at temperature 0.5 vs. 0.8 ([sample output](04-lstm-text-generation/generated_text.txt)). |
| 05 | [RealNVP normalizing flow](05-realnvp-normalizing-flow) | Two Moons | Trained twice as long as the book (600 epochs). Loss went 2.56 → 1.70 and the flow maps the moons to a Gaussian and back. |
| 06 | [Energy-based model](06-energy-based-model) | MNIST | Trained for 120 epochs instead of 60. **Training collapsed after epoch 80**, and the notebook analyses why using the Langevin samples and the replay buffer. |

The most interesting one is 06. Training longer made the model *worse*, and figuring out why taught me more than the runs that worked.

## Run

```bash
pip install tensorflow matplotlib numpy jupyter
```

Most notebooks download their datasets through `keras.datasets`. Notebook 04 expects the [Epicurious recipes dataset](https://www.kaggle.com/datasets/hugodarwood/epirecipes) at `../data/epirecipes/full_format_recipes.json`. Trained model weights aren't included because of their size.
