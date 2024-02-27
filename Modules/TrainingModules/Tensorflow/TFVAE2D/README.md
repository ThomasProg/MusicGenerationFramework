# How to use

From the terminal, you can:
- Move into this directory
- Run ```python training.py``` to train the model
- Run ```python inference.py``` to run the model

# Training

The training can be interrupted at any moment ; it will be resumed at the next execution.

Once finished, the model is saved into the [a specific](MyModel.tf) folder.

# After Training

Convert the model into .onnx with:
```
python -m tf2onnx.convert --saved-model MyModel.tf --output MyModel.onnx
```

You can then infere with C++.





