# Compiler TODOs

Goal: make imported best-submission ONNX graphs progressively editable and recompilable in NeuroGolf Lab.

## Phase 1: Visual Import

- Load best ONNX files as GUI nodes and edges.
- Add every op type found in the best submission to the node palette.
- Preserve ONNX attributes and initializer summaries for inspection.
- Make unsupported compiler status explicit when raw ONNX graphs are loaded.

## Phase 2: Low-Risk Compiler Ops

Add direct compiler support for common ONNX ops whose output shapes are straightforward:

- `Or`, `Xor`
- `ReduceMax`, `ReduceMin`
- `Squeeze`, `Unsqueeze`
- `Reshape`, `Flatten`
- `Min`, `Max`, `Sum`
- `Relu`, `Abs`, `Neg`, `Floor`, `Clip`, `Sign`, `Sqrt`, `Mod`
- `GreaterOrEqual`, `LessOrEqual`

## Phase 3: Indexing And Tensor Construction

Add compiler support for ops that need careful static-shape and initializer handling:

- `Gather`, `GatherElements`, `GatherND`
- `ScatterElements`
- `OneHot`
- `Expand`
- `Split`
- `CumSum`
- `TopK`

## Phase 4: Spatial And Neural Ops

Add support where ONNX Runtime can run the op but the compiler needs more shape inference:

- `MaxPool`, `AveragePool`
- `ConvTranspose`
- `MatMul`, `Gemm`, `QLinearMatMul`
- `GridSample`

## Phase 5: Representation Bridge

The current GUI compiler uses scalar color tensors shaped `[1,1,30,30]`.
Many imported best models use one-hot tensors shaped `[1,10,30,30]`.

Needed:

- explicit scalar-to-one-hot node;
- explicit one-hot-to-scalar node;
- graph mode metadata so loaded templates can declare their tensor representation;
- validation that accepts both representations at model boundaries.
