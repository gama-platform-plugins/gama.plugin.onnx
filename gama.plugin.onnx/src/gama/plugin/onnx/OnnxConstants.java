/*******************************************************************************************************
 *
 * OnnxConstants.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import gama.api.gaml.types.IType;

/**
 * Names, identifiers and documentation categories shared by the ONNX plugin.
 */
public interface OnnxConstants {

	/** The GAML name of the model type. */
	String MODEL = "onnx_model";

	/**
	 * The type id of {@code onnx_model}. Custom type ids must not clash with those of the other plugins loaded in the
	 * same platform: the values around {@code +29}, {@code +30}, {@code +35} and {@code +546654..+546661} are already
	 * taken by the image, kml, dataframe and bdi extensions of gama.core.
	 */
	int MODEL_ID = IType.BEGINNING_OF_CUSTOM_TYPES + 100;

	/** The documentation category of the operators. */
	String CATEGORY = "ONNX";

	/** The concept used to index the operators, types and skills of this plugin. */
	String CONCEPT = "onnx";

	/** The key of the option setting the number of threads used inside a single operator. */
	String OPT_INTRA_OP_THREADS = "intra_op_threads";

	/** The key of the option setting the number of threads used to run operators in parallel. */
	String OPT_INTER_OP_THREADS = "inter_op_threads";

	/** The key of the option setting the graph optimisation level ("none", "basic", "extended", "layout", "all"). */
	String OPT_OPTIMIZATION_LEVEL = "optimization_level";

	/** The key of the option enabling the memory pattern optimisation. */
	String OPT_MEMORY_PATTERN = "memory_pattern";

}
