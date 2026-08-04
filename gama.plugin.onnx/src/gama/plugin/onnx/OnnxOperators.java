/*******************************************************************************************************
 *
 * OnnxOperators.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import gama.annotations.doc;
import gama.annotations.example;
import gama.annotations.no_test;
import gama.annotations.operator;
import gama.annotations.support.IConcept;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.gaml.types.IType;
import gama.api.gaml.types.Types;
import gama.api.runtime.scope.IScope;
import gama.api.types.dataframe.GamaDataFrameFactory;
import gama.api.types.dataframe.IDataFrame;
import gama.api.types.list.GamaListFactory;
import gama.api.types.list.IList;
import gama.api.types.map.GamaMapFactory;
import gama.api.types.map.IMap;

/**
 * The GAML operators of the ONNX plugin: loading a model, running an inference and releasing the sessions.
 *
 * <p>
 * Everything that describes a model, meaning its inputs, outputs, shapes, element types and metadata, is reached
 * through the attributes of {@code onnx_model} rather than through operators: {@code m.inputs},
 * {@code m.input_shapes}, {@code m.info} and the others.
 * </p>
 */
public class OnnxOperators {

	private OnnxOperators() {}

	// ---------------------------------------------------------------------------------------------------------------
	// Loading
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Loads a model from a file.
	 *
	 * @param scope
	 *            the scope
	 * @param path
	 *            the path of the .onnx file
	 * @return the model
	 */
	@operator (
			value = OnnxConstants.MODEL,
			type = OnnxConstants.MODEL_ID,
			category = { OnnxConstants.CATEGORY },
			concept = { OnnxConstants.CONCEPT })
	@doc (
			value = "Loads an ONNX model from a file. The session belongs to the current experiment and is cached there, so naming the same file twice is free and nothing survives the experiment.",
			examples = { @example (
					value = "onnx_model m <- onnx_model(\"../includes/policy.onnx\");",
					isExecutable = false) })
	@no_test
	public static GamaOnnxModel model(final IScope scope, final String path) {
		return OnnxRuntimeManager.load(scope, path, null);
	}

	/**
	 * Loads a model from a file, with session options.
	 *
	 * @param scope
	 *            the scope
	 * @param path
	 *            the path of the .onnx file
	 * @param options
	 *            the session options
	 * @return the model
	 */
	@operator (
			value = OnnxConstants.MODEL,
			type = OnnxConstants.MODEL_ID,
			category = { OnnxConstants.CATEGORY },
			concept = { OnnxConstants.CONCEPT })
	@doc (
			value = "Loads an ONNX model with session options. Supported keys are 'intra_op_threads' (int), 'inter_op_threads' (int), 'memory_pattern' (bool) and 'optimization_level' ('none', 'basic', 'extended', 'layout' or 'all'), each mapping to the setter of the same name in OrtSession.SessionOptions, documented at https://onnxruntime.ai/docs/api/java/ai/onnxruntime/OrtSession.SessionOptions.html. Two models loaded from the same file with different options get their own session.",
			examples = { @example (
					value = "onnx_model m <- onnx_model(\"../includes/policy.onnx\", [\"intra_op_threads\"::1]);",
					isExecutable = false) })
	@no_test
	public static GamaOnnxModel model(final IScope scope, final String path, final IMap<String, Object> options) {
		return OnnxRuntimeManager.load(scope, path, options);
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Inference
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Runs an inference.
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model
	 * @param input
	 *            the input data, or a map from input name to input data
	 * @return a dataframe with one column per output of the model and one row per element of the batch
	 */
	@operator (
			value = "onnx_predict",
			type = IType.DATAFRAME,
			category = { OnnxConstants.CATEGORY },
			concept = { OnnxConstants.CONCEPT, IConcept.REGRESSION })
	@doc (
			value = "Runs the model on the given input and returns its results as a dataframe: one column per output of the graph, named after that output, and one row per element of the batch. "
					+ "The input is flattened in row-major order and reshaped into the shape declared by the graph, so a list, a matrix, a field or an image can all be passed directly; dimensions declared as dynamic are deduced from the amount of data provided. If the model has several inputs, pass a map from input name to data. "
					+ "A cell holds what is left of an output once its first dimension has been used for the rows: a scalar for a [batch] output, a list for a [batch,10] one, nested lists for a [batch,3,8,8] one. "
					+ "Use onnx_predict(model, input, output_name) to get one output as nested lists instead, without the surrounding table.",
			examples = { @example (
					value = "dataframe out <- onnx_predict(m, [0.2, 0.9, 0.1]);",
					isExecutable = false),
					@example (
							value = "list<float> logits <- out[\"logits\"][0];",
							isExecutable = false),
					@example (
							value = "dataframe batch <- onnx_predict(m, [\"pixels\":: my_image, \"scale\":: 1.0]);",
							isExecutable = false) })
	@no_test
	public static IDataFrame predict(final IScope scope, final GamaOnnxModel model, final Object input) {
		return asDataFrame(scope, model, run(scope, model, feed(scope, model, input)));
	}

	/**
	 * Lays the outputs of an inference out as a dataframe: one column per output of the graph, one row per element of
	 * the batch.
	 *
	 * <p>
	 * The rule is the same for every rank, because the conversion already nested the values: if the converted output
	 * is a list, its elements are the rows; otherwise the output is a single row. So a {@code [batch,3]} tensor gives
	 * one row per sample holding a list of three floats, a {@code [batch]} tensor gives one row per sample holding a
	 * scalar, and a {@code [batch,3,8,8]} tensor gives one row per sample holding the nested lists of that sample. A
	 * scalar or a map output is a single row.
	 * </p>
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model, for the error message
	 * @param outputs
	 *            the converted outputs, keyed by output name and in graph order
	 * @return the dataframe
	 */
	private static IDataFrame asDataFrame(final IScope scope, final GamaOnnxModel model,
			final IMap<String, Object> outputs) {
		final IList<String> columns = GamaListFactory.create(Types.STRING);
		final List<IList<Object>> byColumn = new ArrayList<>();
		int rowCount = -1;
		for (final Map.Entry<String, Object> entry : outputs.entrySet()) {
			final IList<Object> column = rowsOf(entry.getValue());
			if (rowCount < 0) {
				rowCount = column.size();
			} else if (rowCount != column.size()) throw GamaRuntimeException.error("The outputs of the ONNX model '"
					+ model.getPath() + "' do not agree on the size of their first dimension (" + entry.getKey()
					+ " has " + column.size() + " rows, the previous ones have " + rowCount
					+ "), so they cannot share a dataframe. Ask for one output at a time with onnx_predict(model, input, output_name).",
					scope);
			columns.add(entry.getKey());
			byColumn.add(column);
		}
		final IList<IList> rows = GamaListFactory.create(Types.LIST);
		for (int r = 0; r < rowCount; r++) {
			final IList<Object> row = GamaListFactory.create(Types.NO_TYPE);
			for (final IList<Object> column : byColumn) { row.add(column.get(r)); }
			rows.add(row);
		}
		return GamaDataFrameFactory.create(scope, columns, rows);
	}

	/**
	 * Splits a converted output into the cells of its column.
	 *
	 * @param converted
	 *            the value returned by the conversion
	 * @return one entry per row
	 */
	@SuppressWarnings ("unchecked")
	private static IList<Object> rowsOf(final Object converted) {
		if (converted instanceof IList<?> list) return (IList<Object>) list;
		final IList<Object> single = GamaListFactory.create(Types.NO_TYPE);
		single.add(converted);
		return single;
	}

	/**
	 * Runs an inference and returns a single named output.
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model
	 * @param input
	 *            the input data, or a map from input name to input data
	 * @param outputName
	 *            the name of the output to return
	 * @return the value of that output
	 */
	@operator (
			value = "onnx_predict",
			category = { OnnxConstants.CATEGORY },
			concept = { OnnxConstants.CONCEPT, IConcept.REGRESSION })
	@doc (
			value = "Runs the model and returns only the named output. Use the 'outputs' attribute of the model to know which names are available.",
			examples = { @example (
					value = "unknown probabilities <- onnx_predict(m, my_input, \"probabilities\");",
					isExecutable = false) })
	@no_test
	public static Object predict(final IScope scope, final GamaOnnxModel model, final Object input,
			final String outputName) {
		if (model == null) throw GamaRuntimeException.error("Cannot run a nil ONNX model", scope);
		if (!model.getOutputs().contains(outputName))
			throw GamaRuntimeException.error("The ONNX model '" + model.getPath() + "' has no output named '"
					+ outputName + "'. Its outputs are " + model.getOutputs() + ".", scope);
		return run(scope, model, feed(scope, model, input)).get(outputName);
	}

	/**
	 * Builds the map from input name to raw GAML value.
	 *
	 * <p>
	 * A map whose keys are exactly input names feeds several inputs at once; anything else feeds the single input of
	 * the model. This is why a single-input model that expects a map-like value is not ambiguous: only a map whose keys
	 * are all declared input names is treated as a multi-input feed.
	 * </p>
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model
	 * @param input
	 *            the value passed by the modeller
	 * @return the feed
	 */
	private static Map<String, Object> feed(final IScope scope, final GamaOnnxModel model, final Object input) {
		if (model == null) throw GamaRuntimeException.error("Cannot run a nil ONNX model", scope);
		final IList<String> names = model.getInputs();
		final Map<String, Object> feed = new LinkedHashMap<>();
		if (input instanceof IMap<?, ?> map && !map.isEmpty() && map.keySet().stream().allMatch(names::contains)) {
			map.forEach((key, value) -> feed.put(key.toString(), value));
			if (feed.size() != names.size())
				throw GamaRuntimeException.error("The ONNX model '" + model.getPath() + "' expects " + names.size()
						+ " inputs " + names + " but only " + feed.keySet() + " were provided.", scope);
			return feed;
		}
		if (names.size() != 1) throw GamaRuntimeException.error("The ONNX model '" + model.getPath() + "' has "
				+ names.size() + " inputs " + names + ", so onnx_predict needs a map from input name to data.", scope);
		feed.put(names.get(0), input);
		return feed;
	}

	/**
	 * Feeds the graph and converts every output back into GAMA values.
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model
	 * @param inputs
	 *            the raw GAML values, keyed by input name
	 * @return the outputs, keyed by output name
	 */
	private static IMap<String, Object> run(final IScope scope, final GamaOnnxModel model,
			final Map<String, Object> inputs) {
		final OrtSession session = model.getSession(scope);
		final OrtEnvironment env = OnnxRuntimeManager.getEnvironment(scope);
		final Map<String, OnnxTensor> tensors = new LinkedHashMap<>();
		try {
			for (final Map.Entry<String, Object> entry : inputs.entrySet()) {
				final String name = entry.getKey();
				final TensorInfo info = model.getInputTensorInfo(name);
				if (info == null) throw GamaRuntimeException.error("The input '" + name + "' of the ONNX model '"
						+ model.getPath() + "' is not a tensor, which this plugin cannot feed yet.", scope);
				tensors.put(name, OnnxConverters.toTensor(scope, env, name, info, entry.getValue()));
			}
			try (OrtSession.Result result = session.run(tensors)) {
				final IMap<String, Object> outputs = GamaMapFactory.create(Types.STRING, Types.NO_TYPE);
				for (final Map.Entry<String, OnnxValue> entry : result) {
					outputs.put(entry.getKey(), OnnxConverters.fromOnnx(scope, entry.getValue()));
				}
				return outputs;
			}
		} catch (final OrtException e) {
			throw GamaRuntimeException
					.error("The ONNX model '" + model.getPath() + "' failed to run: " + e.getMessage(), scope);
		} finally {
			for (final OnnxTensor tensor : tensors.values()) { tensor.close(); }
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Housekeeping
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Releases the session of a model.
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model
	 * @return true if a session was actually released
	 */
	@operator (
			value = "onnx_free",
			category = { OnnxConstants.CATEGORY },
			concept = { OnnxConstants.CONCEPT })
	@doc (
			value = "Releases the native session held by a model and removes it from the cache of the current experiment. The model cannot be used afterwards. Returns true if a session was actually released. Sessions are released automatically when the experiment is disposed, so this is only needed to reclaim memory early or to reload a model that changed on disk.",
			examples = { @example (
					value = "bool released <- onnx_free(m);",
					isExecutable = false) })
	@no_test
	public static boolean free(final IScope scope, final GamaOnnxModel model) {
		return OnnxRuntimeManager.free(scope, model);
	}

	/**
	 * Releases every session of the current experiment whose path contains the given filter.
	 *
	 * @param scope
	 *            the scope
	 * @param pathFilter
	 *            a substring of the paths to release, or an empty string for all of them
	 * @return the number of sessions released
	 */
	@operator (
			value = "onnx_free_all",
			type = IType.INT,
			category = { OnnxConstants.CATEGORY },
			concept = { OnnxConstants.CONCEPT })
	@doc (
			value = "Releases the sessions of the current experiment whose path contains the given string, an empty string meaning all of them, and returns how many were released. Sessions belong to the experiment and are released automatically when it is disposed; this operator only makes that happen earlier.",
			examples = { @example (
					value = "int released <- onnx_free_all(\"\");",
					isExecutable = false) })
	@no_test
	public static int freeAll(final IScope scope, final String pathFilter) {
		return OnnxRuntimeManager.freeAll(scope, pathFilter);
	}

}
