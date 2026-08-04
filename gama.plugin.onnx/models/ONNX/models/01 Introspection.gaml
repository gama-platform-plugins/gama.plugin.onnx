/**
* Name: ONNX Introspection
* Author: Baptiste Lesquoy
* Description: How to load a model and read its signature. Everything that describes a model is an
*              attribute of onnx_model, not an operator, so a model can be inspected before being run.
* Tags: onnx, machine learning
*/
model onnx_introspection

global {

	// Two equivalent ways of naming a model. The session cache belongs to the experiment and is keyed
	// by (path, options), so the file is parsed once no matter how many times it is named -- here,
	// once for the two lines below.
	onnx_file policy_file <- onnx_file("../includes/tiny_policy.onnx");
	onnx_model policy <- onnx_model("../includes/tiny_policy.onnx");

	init {
		write "===== info: everything at a glance =====";
		// The attribute to reach for first when a model does not behave as expected
		write policy.info;

		write "\n===== the signature, attribute by attribute =====";
		write "path          : " + policy.path;
		write "inputs        : " + policy.inputs;         // ["x"]
		write "outputs       : " + policy.outputs;        // ["y"]
		write "input_shapes  : " + policy.input_shapes;   // ["x"::[-1,3]]
		write "output_shapes : " + policy.output_shapes;  // ["y"::[-1,2]]
		write "input_types   : " + policy.input_types;    // ["x"::"float32"]
		write "output_types  : " + policy.output_types;   // ["y"::"float32"]
		write "metadata      : " + policy.metadata;

		write "\n===== a -1 in a shape means a dynamic dimension =====";
		list<int> shape_of_x <- policy.input_shapes["x"];
		write "shape of x    : " + shape_of_x;
		write "batch is free : " + (first(shape_of_x) = -1);
		write "features      : " + last(shape_of_x);

		write "\n===== a model is also a container of its nodes =====";
		// Indexing a model by a node name gives the description of that node
		write "node 'x'      : " + policy["x"];           // ["shape"::[-1,3], "type"::"float32", "role"::"input"]
		write "node 'y'      : " + policy["y"];
		write "number of nodes: " + length(policy);

		write "\n===== a file casts to the model it holds =====";
		onnx_model from_file <- onnx_model(policy_file);
		write "same session  : " + (from_file = policy);  // true: the cache returned the same object

		// A session is a native resource, and it belongs to the EXPERIMENT: it is closed automatically
		// when the experiment is disposed, and nothing survives it. Within one experiment, the
		// simulations of a batch do share their sessions, which is what makes exploring a thousand
		// parameter sets against the same model affordable.
		//
		// So onnx_free / onnx_free_all are never required. They only make the release happen earlier,
		// which is worth doing to reclaim memory in a long experiment, or to pick up a model that
		// changed on disk. Uncommenting the line below would free the model this simulation is using,
		// and any later call on it would raise "has been freed and cannot be used any more".
		//
		write "released : " + onnx_free_all("");
		write "sessions are released when the experiment is disposed";
	}
}

experiment introspection type: gui { }
