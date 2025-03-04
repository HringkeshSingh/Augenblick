import onnx

# Load the ONNX model
model_path = "best.onnx"  # Update the path if necessary
model = onnx.load(model_path)

# Print basic model details
print(f"IR Version: {model.ir_version}")
print(f"Producer: {model.producer_name}, Version: {model.producer_version}")
print(f"Graph Name: {model.graph.name}")
print(f"Number of Nodes: {len(model.graph.node)}")

# Print input and output details
print("Inputs:")
for inp in model.graph.input:
    print(f"  - {inp.name}, Type: {inp.type}")

print("\nOutputs:")
for out in model.graph.output:
    print(f"  - {out.name}, Type: {out.type}")
