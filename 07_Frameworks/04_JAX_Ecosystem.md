# JAX Ecosystem

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[03_Semantic_Kernel|Semantic Kernel]]
**Next:** [[05_AWS_Strands_Agents|AWS Strands Agents]]

---

## Introduction

JAX is a library from Google used by DeepMind and Anthropic for their flagship models. If you are targeting positions at these companies or want to understand the cutting edge of ML research, familiarity with JAX is essential.

**Who uses JAX**: DeepMind builds AlphaFold and Gemini on JAX. Anthropic uses JAX for the Claude training infrastructure. Google Research develops PaLM and Gemma. Many research labs such as Stanford, Berkeley, and partially OpenAI use JAX for research.

JAX is not a replacement for PyTorch in all cases but rather a specialized tool for research-heavy workflows.

## JAX Philosophy

### NumPy + Autodiff + XLA

JAX can be described by a simple formula: JAX = NumPy API + Automatic Differentiation + XLA Compilation.

**NumPy API** provides a familiar interface for working with arrays. You import jax.numpy as jnp and work with it exactly as with regular NumPy: create arrays, perform multiplication and addition operations, compute dot products. This makes porting existing NumPy code extremely simple — just replace the import.

**Automatic Differentiation** allows automatic computation of gradients for any Python functions. You wrap your function in grad(), and JAX returns a new function that computes the derivative. For example, for a quadratic function f(x) = x² + 3x + 1, the gradient is 2x + 3. JAX can compute not only first derivatives but also higher-order derivatives — simply call grad(grad(f)) for the second derivative or use hessian() for the Hessian matrix.

**XLA Compilation** (Accelerated Linear Algebra) is a just-in-time compiler that optimizes your code. You decorate a function with @jit, and on the first call JAX compiles it into optimized machine code. Subsequent calls execute tens of times faster. The first call may take 100ms (compilation + execution), but all subsequent ones take only 1ms (execution only).

### Functional Paradigm

JAX encourages pure functions — functions without side effects. Unlike PyTorch, where the model contains parameters internally as class attributes (stateful approach), in JAX parameters are passed explicitly as function arguments.

In the PyTorch style, you create a class with nn.Parameter, and weights become part of the model object. When calling forward(x), the function implicitly uses self.weight. In the JAX style, you define a function forward(params, x) that explicitly accepts parameters as a dictionary and returns the result without modifying state.

**Advantages of the functional approach**: easier parallelization — without shared state, different devices can independently process data. Determinism — identical inputs always produce identical results, simplifying reproducibility. Easier debugging — no hidden state, all dependencies are explicit, data flow is easier to trace.

## Key Transformations

JAX provides four core function transformations that can be composed with each other.

### jit: Just-in-Time Compilation

The jit transformation compiles a Python function into optimized XLA code. You decorate a function with @jit or call jit(function). On the first call, JAX traces the function and compiles it — this takes time. On subsequent calls with the same shapes, the cached compiled version is used, running tens of times faster.

**jit limitations**: the function must not have side effects (e.g., modifying global variables or printing). Array dimensions must be known at compile time; otherwise, use static_argnums to mark arguments that affect shapes. Python conditionals (if/else) with data-dependent conditions require special handling via jnp.where or lax.cond.

### grad: Automatic Differentiation

The grad transformation takes a function and returns a new function that computes the gradient with respect to the first argument. For example, for a loss function loss_fn(params, x, y), calling grad(loss_fn) returns a function that computes gradients with respect to params.

A useful variation is value_and_grad, which returns both the function value and gradients simultaneously. This is more efficient than calling the function twice, as the forward pass is reused.

JAX supports derivatives of any order: call grad(grad(f)) for the second derivative or use hessian(f) to compute the full Hessian matrix (matrix of second derivatives).

### vmap: Vectorization

The vmap transformation automatically converts a function operating on a single example into a function operating on a batch of examples. You specify which argument axes to batch over via the in_axes parameter.

For example, the function single_example(x, w) computes a dot product for a single example. vmap(single_example, in_axes=(0, None)) creates a function that applies the operation to each element along the first dimension of x while using the same w for all examples.

Combining vmap with grad is particularly powerful — it enables computing per-example gradients (gradients for each example in a batch separately), which is needed for certain advanced training techniques.

### pmap: Parallel Mapping

The pmap transformation distributes computations across multiple devices (GPUs or TPUs). You decorate a function with @pmap, and JAX automatically distributes the first dimension of the input array across available devices. Each device performs computations on its portion of the data in parallel.

This is especially useful for data parallelism in distributed training. Input data must have shape [num_devices, ...], where the first dimension corresponds to the number of available devices.

## Ecosystem: Flax, Optax, Orbax

JAX by itself is a low-level library for numerical computation. An ecosystem of specialized libraries is used for building neural networks.

### Flax: Neural Networks

Flax is the primary library for building neural networks in JAX. You define models as classes inheriting from nn.Module and use the @nn.compact decorator to define the architecture inside the __call__ method.

**Key difference from PyTorch**: parameters exist separately from the model. First, you call model.init(random_key, example_input), which returns a dictionary of parameters. Then for inference you call model.apply(params, input_batch) instead of simply model(input_batch). This follows the JAX functional paradigm — the model does not store state; everything is passed explicitly.

Explicit management of random state via PRNG keys is also important: you create a key via random.PRNGKey(seed) and pass it during initialization for deterministic generation of initial weights.

### Optax: Optimizers

Optax provides gradient optimizers in a functional style. You create an optimizer (e.g., optax.adam(learning_rate)), initialize its state via optimizer.init(params), and in the training step update parameters via optimizer.update(grads, opt_state), which returns updates and a new optimizer state.

A powerful feature of Optax is composability via optax.chain(). You can combine gradient clipping, learning rate schedules (warmup, cosine decay), and the optimizer itself into a single pipeline. For example, adding gradient clipping before Adam with a cosine learning rate schedule is a single line of composition.

### Orbax: Checkpointing

Orbax solves the problem of saving and loading model state. You create a PyTreeCheckpointer that can handle arbitrary nested structures (pytrees). When saving, you pass a dictionary with parameters and optimizer state; when loading, you get the same dictionary back. This is critical for long training runs and experiments where state needs to be restored after interruptions.

## JAX vs PyTorch

### When to Choose JAX

**JAX is better for**: TPU workloads with native support, custom research with non-standard operations, when per-example gradients are needed, multi-device training with pmap, working in the DeepMind/Anthropic style.

**PyTorch is better for**: rapid prototyping, using ready-made models from HuggingFace, when the ecosystem matters, production deployment via TorchServe and ONNX, teams already familiar with PyTorch.

### Comparison Using an MLP Example

**PyTorch approach**: you create a model class with layers as attributes (self.fc1, self.fc2), instantiate the model and move it to GPU via .cuda(). You create an optimizer by passing model.parameters(). In the training loop, you perform a forward pass via model(x), call loss.backward() to compute gradients, optimizer.step() to update weights, and optimizer.zero_grad() to clear gradients.

**JAX/Flax approach**: you define a model with the @nn.compact decorator, but the model does not contain parameters. Parameters are initialized separately via model.init(). The optimizer is created via Optax, and its state is also initialized separately. The training step is typically wrapped in @jax.jit for compilation. Inside, you define a loss function, compute gradients via value_and_grad, obtain updates from the optimizer, and apply them to parameters — everything is explicit, with no hidden state. Device placement happens automatically.

## Practical Patterns

### PRNG Handling

JAX uses explicit random state management instead of a global random number generator. You create a key via random.PRNGKey(seed) and then "split" it via random.split(key) into several independent subkeys for different operations.

This ensures determinism: the same key always produces the same sequence. A typical pattern for training: at the beginning of an epoch, split the main key; at each step, split again for dropout, data augmentation, and other stochastic operations.

### Pytrees

JAX works with "pytrees" — arbitrary nested structures of dict, list, and tuple. Model parameters are typically stored as a nested dictionary.

The function jax.tree_map(fn, tree) applies the function fn to each leaf (terminal array) in the tree. This is convenient for operations like scaling all parameters, applying weight decay, or computing norms. The function jax.tree_util.tree_leaves(tree) returns a flat list of all leaf nodes.

### Debugging

Debugging compiled code is harder than eager execution. For debugging, you can temporarily disable JIT via the context manager with jax.disable_jit() — functions will then execute in eager mode, and you can use regular breakpoints and print.

To print values inside jit-compiled functions, use jax.debug.print() instead of regular print(). Regular print executes during tracing, not on each call, which can be confusing.

## Migration from PyTorch

### Checklist

When transitioning from PyTorch to JAX, follow these steps:

**Replace imports**: torch becomes jax.numpy as jnp, torch.nn becomes flax.linen as nn, torch.optim becomes optax.

**Explicit state**: all state is now explicit. Parameters are not stored in the model but passed as arguments. Random keys must be created and split manually. Optimizer state is also passed explicitly at each step.

**Pure functions**: avoid mutations and side effects. No assignments like self.parameter = ..., no modifications to global state. All function inputs and outputs must be explicit arguments and return values.

**Compilation**: wrap the training step in @jit for performance. Ensure that all array shapes are known statically or use static_argnums for arguments that affect shapes.

### Common Mistakes

**Array mutations**: JAX arrays are immutable. You cannot write x[0] = 1. Instead, use the functional API: x.at[0].set(1), which creates a new array with the modified value.

**Python control flow in jit**: inside @jit, you cannot use Python if with conditions depending on traced array values (the compiler does not know their values). Instead of if x > 0: return x else: return -x, use jnp.where(x > 0, x, -x), which works with arrays and is supported by the compiler.

## Key Takeaways

JAX represents a functional paradigm for deep learning with an emphasis on composability of transformations. The core transformations — jit for compilation, grad for automatic differentiation, vmap for vectorization, and pmap for parallelization — can be freely combined.

The Flax, Optax, and Orbax ecosystem provides the necessary tools for building and training neural networks in a functional style. Explicit state and random key management requires adaptation from PyTorch developers but offers advantages in determinism and parallelization.

JAX is particularly strong for research and TPU workloads, but PyTorch remains the leader for rapid prototyping and production deployment thanks to its mature ecosystem.

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[03_Semantic_Kernel|Semantic Kernel]]
**Next:** [[05_AWS_Strands_Agents|AWS Strands Agents]]
