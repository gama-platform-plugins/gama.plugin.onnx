/**
* Name: ONNX Inputs and Shapes
* Author: Baptiste Lesquoy
* Description: The conversion rules. Any GAMA value is flattened row-major and reshaped into the shape
*              the graph declares, so lists, nested lists and matrices are interchangeable. The model
*              used here is a 3 -> 2 affine map with a free batch dimension.
* Tags: onnx, machine learning
*/
model onnx_inputs_and_shapes

global {

	onnx_model policy <- onnx_model("../includes/tiny_policy.onnx");

	init {
		write "the model expects " + policy.input_shapes["x"] + " of " + policy.input_types["x"];

		write "\n===== one row =====";
		// onnx_predict always returns a dataframe: one column per output of the graph, named after
		// that output, and one row per element of the batch. Here that is one column "y", one row.
		dataframe one <- onnx_predict(policy, [1.0, 2.0, 3.0]);
		write "the dataframe : " + one;
		write "column y      : " + one["y"];        // [[4.5,4.5]] : one cell, holding the row
		write "the values    : " + one["y"][0];     // [4.5,4.5]

		write "\n===== the dynamic dimension is deduced from the data =====";
		// 6 values against a [-1,3] input can only mean a batch of 2, hence two rows
		dataframe two <- onnx_predict(policy, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
		write "rows          : " + two.rows;        // 2
		write "column y      : " + two["y"];        // [[4.5,4.5],[0.5,-0.5]]

		write "\n===== nesting is irrelevant: everything is flattened row-major =====";
		dataframe nested <- onnx_predict(policy, [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]);
		write "same result   : " + (nested["y"] = two["y"]);

		write "\n===== a matrix is read row by row =====";
		// matrix([[...],[...]]) is built column by column, so this matrix holds the same six values
		matrix<float> m <- matrix<float>([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);
		write "from a matrix : " + onnx_predict(policy, m)["y"];

		write "\n===== ints are accepted where the graph wants float32 =====";
		write "from ints     : " + onnx_predict(policy, [1, 2, 3])["y"];   // [[4.5,4.5]]

		write "\n===== asking for one output raw, without the table =====";
		// The three-operand form is the escape hatch: it returns the tensor as nested lists,
		// matching its shape exactly, with no dataframe around it.
		list<list<float>> raw <- onnx_predict(policy, [1.0, 2.0, 3.0], "y");
		write "raw tensor    : " + raw;             // [[4.5,4.5]]

		write "\n===== mismatches fail loudly rather than silently =====";
		// Uncomment either line to see the error message the plugin produces:
		//
		//   onnx_predict(policy, [1.0, 2.0, 3.0, 4.0]);
		//     -> The input 'x' has shape [-1, 3], so the number of values must be a multiple of 3,
		//        but 4 were provided.
		//
		//   onnx_predict(policy, ["a", "b", "c"]);
		//     -> The input 'x' of the ONNX model cannot accept a String. Expected a number, a list,
		//        a matrix, a field or an image.
		write "see the comments above for the error cases";
	}
}

experiment shapes type: gui { }
