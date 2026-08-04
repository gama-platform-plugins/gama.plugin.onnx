# gama.plugin.onnx

Run [ONNX](https://onnx.ai) models from GAML: neural networks, learned policies, surrogates of costly submodels, scikit-learn pipelines converted to ONNX.

Inputs are ordinary GAMA values, results come back as a `dataframe`.

## Quick start

```gaml
global {
    onnx_model classifier <- onnx_model("../includes/classifier.onnx");

    init {
        write classifier.info;                                    // the signature of the model

        dataframe out <- onnx_predict(classifier, [0.2, 0.9, 0.1, 0.4]);
        list<float> logits <- out["logits"][0];                   // column "logits", first row
        write "class " + (logits index_of (max(logits)));
    }
}
```

## Loading a model

```gaml
onnx_model m <- onnx_model("../includes/policy.onnx");
onnx_model tuned <- onnx_model("../includes/policy.onnx", ["intra_op_threads":: 1]);
```

| Option | Type | Maps to |
|---|---|---|
| `intra_op_threads` | `int` | `setIntraOpNumThreads` |
| `inter_op_threads` | `int` | `setInterOpNumThreads` |
| `memory_pattern` | `bool` | `setMemoryPatternOptimization` |
| `optimization_level` | `"none"`, `"basic"`, `"extended"`, `"layout"`, `"all"` | `setOptimizationLevel` |

Each key maps to the setter of the same name on the runtime's session options, described in the [OrtSession.SessionOptions javadoc](https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html), which is also where to look for the settings this plugin does not expose yet. Two models loaded from the same file with different options get their own session.

`onnx_file` is the same thing through the usual GAMA file idiom, with path resolution and the generic `file(...)` resolver:

```gaml
onnx_file f <- onnx_file("../includes/policy.onnx");
onnx_model m <- f.contents;
```

## Feeding a model

The value you pass is flattened in row-major order and reshaped into the shape the graph declares, so a list, a nested list, a matrix, a field or an image all work:

```gaml
onnx_predict(m, [1.0, 2.0, 3.0]);                    // a flat list
onnx_predict(m, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); // two samples
onnx_predict(m, my_matrix);                          // read row by row
onnx_predict(m, my_image);                           // see "Images" below
```

A dimension the model leaves free shows up as `-1` in `input_shapes` and is deduced from the amount of data you provide. An input declared `[-1, 18]` takes 18 values for one sample, 180 for ten, and gives one row per sample. When several dimensions are free, all but the first are taken as 1.

Ints are accepted wherever the graph expects a float. A count that does not fit raises an error naming the input, the expected shape and the number of values received.

Models with several inputs take a map from input name to data:

```gaml
onnx_predict(m, ["pixels":: my_image, "scale":: 1.0]);
```

## Reading the results

`onnx_predict(model, input)` returns a `dataframe`:

- **one column per output of the graph**, named after that output;
- **one row per element of the first dimension**, which usually means one row per sample, and in an agent-based model one row per agent;
- **each cell holds what is left of the output** once its first dimension has been spent on the rows: a scalar for a `[batch]` output, a list of ten floats for `[batch,10]`, nested lists for `[batch,3,8,8]`.

```gaml
dataframe out <- onnx_predict(policy, observations);
out.rows                        // how many samples came back
out.keys                        // the output names
out["action_mean"]              // the whole column, one entry per sample
out["action_mean"][0]           // the cell of the first row
```

A sequence output gives one row per element; a scalar or a map output gives a single row. Outputs whose first dimensions disagree cannot share a table, and the operator says which ones and by how much.

To get one output on its own, as nested lists matching its shape and without the surrounding table:

```gaml
list<list<float>> raw <- onnx_predict(policy, observations, "action_mean");
```

An `int64` value too large for a GAML `int` raises an error.

## Inspecting a model

The signature of a model is reached through its attributes:

| Attribute | Type | Contents |
|---|---|---|
| `path` | `string` | the path the model was loaded from |
| `inputs` / `outputs` | `list<string>` | node names, in graph order |
| `input_shapes` / `output_shapes` | `map<string, list<int>>` | shapes, where `-1` marks a free dimension |
| `input_types` / `output_types` | `map<string, string>` | `"float32"`, `"int64"`, `"bool"` and so on |
| `metadata` | `map<string, string>` | producer, graph name, domain, description, version, custom entries |
| `info` | `string` | a readable summary of all of the above, the first thing to write to the console when a model surprises you |

A model is also a container of its nodes, keyed by name:

```gaml
m["logits"]     // ["shape":: [-1,10], "type":: "float32", "role":: "output"]
length(m)       // how many nodes
```

## Images

An image can be fed straight to a vision model. The layout comes from the shape the graph declares: rank 4 is read as NCHW, unless only its last dimension is a plausible channel count (1, 3 or 4), in which case NHWC; rank 3 is the same without the batch; rank 2 is grayscale. One channel means luminance, three RGB, four RGBA, and values are normalised to [0,1].

The image carries its own height and width, so a model that leaves them free accepts any size. A model that fixes them requires that size exactly; resize with `with_size` beforehand.

## The `onnx` skill

For agents that carry a model of their own:

```gaml
species walker skills: [onnx] {
    init { do load_model(path: "../includes/policy.onnx"); }

    reflex act {
        list<float> action <- predict([speed, heading, distance_to_target])["action"][0];
    }
}
```

The model sits in the `inner_model` attribute and can also be assigned directly, which is how a whole population shares one model loaded once in `global`:

```gaml
init { inner_model <- world.brain; }
```

When many agents decide at the same moment, one batched call is much cheaper than one call per agent: concatenate the observations, and read back the row matching each agent.

## Sessions

A session lives as long as the experiment that loaded it, and is shared by everything inside it, so naming the same file in `global` and in a thousand agents loads it once. Within a batch experiment the simulations share their sessions too.

Sessions are released when the experiment is disposed. `onnx_free(model)` and `onnx_free_all(path_filter)` make that happen earlier, which is useful to reclaim memory during a long experiment or to pick up a model that changed on disk.

## Examples

In `models/ONNX/models/`, one concept per model:

| Model | Shows |
|---|---|
| `01 Introspection` | loading, the attributes, a model as a container of its nodes |
| `02 Inputs and Shapes` | flat lists, nested lists, matrices, batches, free dimensions, the dataframe layout |
| `03 Classification` | logits to class with `index_of` and `max`, and one batched call for a whole population |
| `04 Multiple Inputs and Outputs` | one column per output, and the raw three-operand form |
| `05 Images` | feeding an image, layout and normalisation |
| `06 Walkers` | the skill driving agents, with a map display and a chart |

The `.onnx` files in `models/ONNX/includes/` are a few hundred bytes each and are regenerated by `make_models.py`, pure Python with no dependency. Their weights are hand-set, so the examples show the wiring rather than meaningful behaviour.

## Requirements

The bundled ONNX Runtime is the CPU build, with natives for **win-x64, linux-x64, linux-aarch64 and macos-aarch64**.

`onnx_file` reads models; writing them is done by the training toolchain.
