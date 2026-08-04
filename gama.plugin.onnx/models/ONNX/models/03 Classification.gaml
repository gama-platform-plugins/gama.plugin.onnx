/**
* Name: ONNX Classification
* Author: Baptiste Lesquoy
* Description: Turning a feature vector into a class.
*              The model maps 4 features to 3 logits; the predicted class is the argmax of the logits.
* Tags: onnx, machine learning
*/
model onnx_classification

global {

	onnx_model brain <- onnx_model("../includes/classifier.onnx");

	list<string> class_names <- ["red", "green", "blue"];

	/**
	 * Runs the classifier on one feature vector and returns the index of the winning class.
	 */
	int classify (list<float> features) {
		// onnx_predict returns a dataframe: column "logits" (the name of the output), one row here.
		// A cell holds what is left of the [batch,3] output once the batch became the rows, so a
		// list of three floats.
		list<float> logits <- onnx_predict(brain, features)["logits"][0];
		return logits index_of (max(logits));
	}

	init {
		write brain.info;

		write "\n===== one vector at a time =====";
		// The weights push feature i towards class i, so a dominant feature picks its class
		loop i from: 0 to: 2 {
			list<float> features <- [0.0, 0.0, 0.0, 0.0];
			features[i] <- 1.0;
			int c <- classify(features);
			write "features " + features + " -> class " + c + " (" + class_names[c] + ")";
		}

		write "\n===== a whole batch in a single call =====";
		// One inference for many individuals is far cheaper than one call per individual:
		// the values of every row are simply concatenated.
		list<float> batch <- [];
		loop i from: 0 to: 2 {
			loop j from: 0 to: 3 { batch << (j = i ? 1.0 : 0.0); }
		}
		// One row per individual: the dataframe makes the correspondence explicit
		dataframe result <- onnx_predict(brain, batch);
		write "rows returned : " + result.rows;
		loop row over: result["logits"] {
			int c <- row index_of (max(row));
			write "  " + row + " -> " + class_names[c];
		}

		create sampled number: 12;
	}
}

/**
 * Agents that classify themselves once, at creation. Note that they use the global model through the
 * operator rather than the skill: when every agent shares one model, an attribute in global is enough.
 */
species sampled {

	list<float> features <- [rnd(1.0), rnd(1.0), rnd(1.0), rnd(1.0)];
	int predicted_class;
	rgb color;

	init {
		predicted_class <- world.classify(features);
		color <- [#red, #green, #blue] at predicted_class;
	}

	aspect default {
		draw circle(2) color: color border: #black;
	}
}

experiment classification type: gui {
	output {
		display "Classified agents" type: 2d {
			species sampled aspect: default;
		}
	}
}
