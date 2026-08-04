/**
* Name: ONNX Multiple Inputs and Outputs
* Author: Baptiste Lesquoy
* Description: Models with more than one input are fed with a map from input name to data, and models
*              with more than one output return a map from output name to result. A single output can
*              also be asked for by name, which avoids converting the ones that are not needed.
* Tags: onnx, machine learning
*/
model onnx_multiple_inputs_and_outputs

global {

	// Two inputs a and b, two outputs: their sum and their difference
	onnx_model arithmetic <- onnx_model("../includes/multi_io.onnx");

	init {
		write arithmetic.info;
		write "inputs  : " + arithmetic.inputs;   // ["a","b"]
		write "outputs : " + arithmetic.outputs;  // ["sum","difference"]

		write "\n===== feeding several inputs: a map keyed by input name =====";
		map<string, unknown> feed <- ["a":: [3.0, 4.0], "b":: [1.0, 2.0]];

		// This is where the dataframe earns its keep: one column per output, named after it
		dataframe result <- onnx_predict(arithmetic, feed);
		write "columns       : " + result.keys;            // ["sum","difference"]
		write "rows          : " + result.rows;            // 1
		write "sum           : " + result["sum"][0];       // [4.0,6.0]
		write "difference    : " + result["difference"][0]; // [2.0,2.0]

		write "\n===== asking for a single output raw, without the table =====";
		// The three-operand form returns the tensor itself as nested lists. Only that output is
		// converted back into GAMA values.
		list<list<float>> only_sum <- onnx_predict(arithmetic, feed, "sum");
		write "sum only      : " + first(only_sum);

		write "\n===== batches work here too =====";
		// Two rows per input, so two rows in the dataframe, both columns aligned on them
		map<string, unknown> batch <- ["a":: [3.0, 4.0, 10.0, 20.0], "b":: [1.0, 2.0, 1.0, 1.0]];
		dataframe batched <- onnx_predict(arithmetic, batch);
		write "rows          : " + batched.rows;           // 2
		write "sums          : " + batched["sum"];
		write "differences   : " + batched["difference"];

		write "\n===== omitting an input is an error, not a silent default =====";
		// Uncomment to see it:
		//   onnx_predict(arithmetic, ["a":: [1.0, 2.0]]);
		//     -> The ONNX model expects 2 inputs [a, b] but only [a] were provided.
		//
		// Asking for an output that does not exist is caught too:
		//   onnx_predict(arithmetic, feed, "product");
		//     -> has no output named 'product'. Its outputs are [sum, difference].
		write "see the comments above for the error cases";
	}
}

experiment multiple_io type: gui { }
