# Porting to Groq

This is a brief summary of how to port and run this model on Groq hardware.
The steps are 
1. Export the model to ONNX with static sized inputs.
2. Compile the ONNX to an IOP.
3. Run with groq runner.

## ONNX Export

To run on Groq hardware, we have to fix the model input sizes.  GNN datasets typically vary the number of vertices and edges and zero-padding would change the graph description.  Instead of zero padding we use "trivial graph padding".  The idea is to emulate batching, by having all inferences run in a batch of size 2, with the second graph in the batch being used for padding.  For GraphDRP with a max input of $N$ nodes, and $E$ edges, we use as inputs
* x, a tensor of size ($N+2$, node_features), initalized to 0
* edge_index, a tensor of size (2, $E$+1), initialized to $i\in(N,N+2)$
* batch, a tensor of size ($N+2$), initialized to 1
* target, a tensor of size (2, target_features), initalized to 0

When running an inference, set the initial addresses of the tensor to the actual graph data.  Take only the 0th element of the output.  

Then its advised to optimize the ONNX graph and reduce it to FP16.  This is done in `groq_utils/optimize_onnx.py`

## Compile IOP


## Groq Runtime