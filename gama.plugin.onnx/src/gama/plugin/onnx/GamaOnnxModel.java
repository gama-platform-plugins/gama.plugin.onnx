/*******************************************************************************************************
 *
 * GamaOnnxModel.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import java.util.Map;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxModelMetadata;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import gama.annotations.doc;
import gama.annotations.getter;
import gama.annotations.variable;
import gama.annotations.vars;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.gaml.types.IContainerType;
import gama.api.gaml.types.IType;
import gama.api.gaml.types.Types;
import gama.api.runtime.scope.IScope;
import gama.api.types.geometry.IPoint;
import gama.api.types.list.GamaListFactory;
import gama.api.types.list.IList;
import gama.api.types.map.GamaMapFactory;
import gama.api.types.map.IMap;
import gama.api.types.matrix.IMatrix;
import gama.api.types.misc.IContainer;
import gama.api.utils.json.IJson;
import gama.api.utils.json.IJsonValue;

/**
 * A loaded ONNX model, wrapping an {@link OrtSession} together with the description of its graph.
 *
 * <p>
 * The signature of the model (names, shapes and element types of its inputs and outputs) is read once at load time and
 * exposed to GAML as attributes: {@code m.inputs}, {@code m.input_shapes}, {@code m.metadata}, {@code m.info} and the
 * others listed below.
 * </p>
 *
 * <p>
 * The model is also a container of its nodes, keyed by node name, so that {@code m["logits"]} returns the description
 * of that node and {@code length(m)} the number of nodes. This is what allows {@code onnx_file} to expose an
 * {@code onnx_model} as its contents. Being a description of a file on disk, the container is read-only: every
 * modification operation raises an explicit error.
 * </p>
 */
@vars ({ @variable (
		name = GamaOnnxModel.PATH,
		type = IType.STRING,
		doc = @doc ("The path of the file this model was loaded from, as it was written in the model")),
		@variable (
				name = GamaOnnxModel.INPUTS,
				type = IType.LIST,
				of = IType.STRING,
				doc = @doc ("The names of the inputs of the model, in the order declared by its graph")),
		@variable (
				name = GamaOnnxModel.OUTPUTS,
				type = IType.LIST,
				of = IType.STRING,
				doc = @doc ("The names of the outputs of the model, in the order declared by its graph")),
		@variable (
				name = GamaOnnxModel.INPUT_SHAPES,
				type = IType.MAP,
				index = IType.STRING,
				of = IType.LIST,
				doc = @doc ("The shape of each input, as a list of int. A dimension of -1 is dynamic (typically the batch size) and is deduced from the data actually passed to onnx_predict")),
		@variable (
				name = GamaOnnxModel.OUTPUT_SHAPES,
				type = IType.MAP,
				index = IType.STRING,
				of = IType.LIST,
				doc = @doc ("The shape of each output, as a list of int. A dimension of -1 is only known once the inference has run")),
		@variable (
				name = GamaOnnxModel.INPUT_TYPES,
				type = IType.MAP,
				index = IType.STRING,
				of = IType.STRING,
				doc = @doc ("The element type of each input, such as 'float32', 'float64', 'int32', 'int64', 'bool' or 'string'")),
		@variable (
				name = GamaOnnxModel.OUTPUT_TYPES,
				type = IType.MAP,
				index = IType.STRING,
				of = IType.STRING,
				doc = @doc ("The element type of each output")),
		@variable (
				name = GamaOnnxModel.METADATA,
				type = IType.MAP,
				index = IType.STRING,
				of = IType.STRING,
				doc = @doc ("The metadata embedded in the file: producer, graph name, domain, description, version, plus any custom entry")),
		@variable (
				name = GamaOnnxModel.INFO,
				type = IType.STRING,
				doc = @doc ("A human-readable summary of the signature of the model, meant to be written in the console when a model does not behave as expected")) })
public class GamaOnnxModel
		implements IContainer.Addressable<String, Object, String, Object>, IContainer.Modifiable<String, Object, String, Object> {

	/** The name of the 'path' attribute. */
	public static final String PATH = "path";

	/** The name of the 'inputs' attribute. */
	public static final String INPUTS = "inputs";

	/** The name of the 'outputs' attribute. */
	public static final String OUTPUTS = "outputs";

	/** The name of the 'input_shapes' attribute. */
	public static final String INPUT_SHAPES = "input_shapes";

	/** The name of the 'output_shapes' attribute. */
	public static final String OUTPUT_SHAPES = "output_shapes";

	/** The name of the 'input_types' attribute. */
	public static final String INPUT_TYPES = "input_types";

	/** The name of the 'output_types' attribute. */
	public static final String OUTPUT_TYPES = "output_types";

	/** The name of the 'metadata' attribute. */
	public static final String METADATA = "metadata";

	/** The name of the 'info' attribute. */
	public static final String INFO = "info";

	/** The key, in a node description, of its shape. */
	public static final String NODE_SHAPE = "shape";

	/** The key, in a node description, of its element type. */
	public static final String NODE_TYPE = "type";

	/** The key, in a node description, of its role ('input' or 'output'). */
	public static final String NODE_ROLE = "role";

	/** The path as written by the modeller. */
	private final String path;

	/** The resolved absolute path. */
	private final String absolutePath;

	/** The canonical form of the options this session was built with. */
	private final String optionsSignature;

	/** The underlying session. */
	private final OrtSession session;

	/** The names of the inputs. */
	private final IList<String> inputNames;

	/** The names of the outputs. */
	private final IList<String> outputNames;

	/** The description of every node, keyed by name. */
	private final IMap<String, Object> nodes;

	/** The shape of each input. */
	private final IMap<String, Object> inputShapes;

	/** The shape of each output. */
	private final IMap<String, Object> outputShapes;

	/** The element type of each input. */
	private final IMap<String, Object> inputTypes;

	/** The element type of each output. */
	private final IMap<String, Object> outputTypes;

	/** The metadata of the model. */
	private final IMap<String, Object> metadata;

	/** Whether the session has been closed. */
	private volatile boolean closed;

	/**
	 * Wraps a freshly created session and reads its signature.
	 *
	 * @param scope
	 *            the scope
	 * @param path
	 *            the path as written by the modeller
	 * @param absolutePath
	 *            the resolved absolute path
	 * @param optionsSignature
	 *            the canonical form of the options
	 * @param session
	 *            the session to wrap
	 */
	GamaOnnxModel(final IScope scope, final String path, final String absolutePath, final String optionsSignature,
			final OrtSession session) {
		this.path = path;
		this.absolutePath = absolutePath;
		this.optionsSignature = optionsSignature;
		this.session = session;
		this.inputNames = GamaListFactory.create(Types.STRING);
		this.outputNames = GamaListFactory.create(Types.STRING);
		this.nodes = GamaMapFactory.create(Types.STRING, Types.MAP);
		this.inputShapes = GamaMapFactory.create(Types.STRING, Types.LIST);
		this.outputShapes = GamaMapFactory.create(Types.STRING, Types.LIST);
		this.inputTypes = GamaMapFactory.create(Types.STRING, Types.STRING);
		this.outputTypes = GamaMapFactory.create(Types.STRING, Types.STRING);
		this.metadata = GamaMapFactory.create(Types.STRING, Types.STRING);
		try {
			describe(session.getInputInfo(), inputNames, inputShapes, inputTypes, "input");
			describe(session.getOutputInfo(), outputNames, outputShapes, outputTypes, "output");
			readMetadata(session.getMetadata());
		} catch (final OrtException e) {
			throw GamaRuntimeException.error(
					"Cannot read the signature of the ONNX model '" + path + "': " + e.getMessage(), scope);
		}
	}

	/**
	 * Fills the description of one side (inputs or outputs) of the graph.
	 *
	 * @param infos
	 *            the node infos returned by the session
	 * @param names
	 *            the list of names to fill
	 * @param shapes
	 *            the map of shapes to fill
	 * @param types
	 *            the map of element types to fill
	 * @param role
	 *            "input" or "output"
	 */
	private void describe(final Map<String, NodeInfo> infos, final IList<String> names, final IMap<String, Object> shapes,
			final IMap<String, Object> types, final String role) {
		for (final Map.Entry<String, NodeInfo> entry : infos.entrySet()) {
			final String name = entry.getKey();
			names.add(name);
			final ValueInfo info = entry.getValue().getInfo();
			final IList<Integer> shape = GamaListFactory.create(Types.INT);
			String type = "unknown";
			if (info instanceof TensorInfo tensor) {
				for (final long dimension : tensor.getShape()) { shape.add((int) dimension); }
				type = OnnxConverters.nameOf(tensor.type);
			} else {
				// sequences and maps (typical of converted scikit-learn models) have no shape
				type = info.getClass().getSimpleName().replace("Info", "").toLowerCase();
			}
			shapes.put(name, shape);
			types.put(name, type);
			final IMap<String, Object> node = GamaMapFactory.create(Types.STRING, Types.NO_TYPE);
			node.put(NODE_SHAPE, shape);
			node.put(NODE_TYPE, type);
			node.put(NODE_ROLE, role);
			nodes.put(name, node);
		}
	}

	/**
	 * Reads the metadata embedded in the file.
	 *
	 * @param source
	 *            the metadata returned by the session
	 */
	private void readMetadata(final OnnxModelMetadata source) {
		metadata.put("producer", source.getProducerName());
		metadata.put("graph_name", source.getGraphName());
		metadata.put("graph_description", source.getGraphDescription());
		metadata.put("domain", source.getDomain());
		metadata.put("description", source.getDescription());
		metadata.put("version", String.valueOf(source.getVersion()));
		source.getCustomMetadata().forEach(metadata::put);
	}

	// ---------------------------------------------------------------------------------------------------------------
	// GAML attributes
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Returns the path of the file the model was loaded from.
	 *
	 * @return the path as written by the modeller
	 */
	@getter (PATH)
	public String getPath() { return path; }

	/**
	 * Returns the names of the inputs.
	 *
	 * @return the input names
	 */
	@getter (INPUTS)
	public IList<String> getInputs() { return inputNames; }

	/**
	 * Returns the names of the outputs.
	 *
	 * @return the output names
	 */
	@getter (OUTPUTS)
	public IList<String> getOutputs() { return outputNames; }

	/**
	 * Returns the shape of each input.
	 *
	 * @return a map from input name to shape
	 */
	@getter (INPUT_SHAPES)
	public IMap<String, Object> getInputShapes() { return inputShapes; }

	/**
	 * Returns the shape of each output.
	 *
	 * @return a map from output name to shape
	 */
	@getter (OUTPUT_SHAPES)
	public IMap<String, Object> getOutputShapes() { return outputShapes; }

	/**
	 * Returns the element type of each input.
	 *
	 * @return a map from input name to element type
	 */
	@getter (INPUT_TYPES)
	public IMap<String, Object> getInputTypes() { return inputTypes; }

	/**
	 * Returns the element type of each output.
	 *
	 * @return a map from output name to element type
	 */
	@getter (OUTPUT_TYPES)
	public IMap<String, Object> getOutputTypes() { return outputTypes; }

	/**
	 * Returns the metadata embedded in the file.
	 *
	 * @return a map of metadata
	 */
	@getter (METADATA)
	public IMap<String, Object> getMetadata() { return metadata; }

	/**
	 * Returns a human-readable summary of the signature of the model.
	 *
	 * @return the summary
	 */
	@getter (INFO)
	public String getInfo() {
		final StringBuilder sb = new StringBuilder();
		sb.append("ONNX model '").append(path).append("'");
		if (closed) { sb.append(" [closed]"); }
		sb.append("\n  file: ").append(absolutePath);
		if (!optionsSignature.isEmpty()) { sb.append("\n  options: ").append(optionsSignature); }
		sb.append("\n  inputs:");
		appendNodes(sb, inputNames, inputShapes, inputTypes);
		sb.append("\n  outputs:");
		appendNodes(sb, outputNames, outputShapes, outputTypes);
		metadata.forEach((k, v) -> {
			if (v != null && !v.toString().isBlank()) { sb.append("\n  ").append(k).append(": ").append(v); }
		});
		return sb.toString();
	}

	/**
	 * Appends the description of a set of nodes to the summary.
	 *
	 * @param sb
	 *            the builder
	 * @param names
	 *            the node names
	 * @param shapes
	 *            the shapes
	 * @param types
	 *            the element types
	 */
	private void appendNodes(final StringBuilder sb, final IList<String> names, final IMap<String, Object> shapes,
			final IMap<String, Object> types) {
		if (names.isEmpty()) { sb.append(" none"); }
		for (final String name : names) {
			sb.append("\n    ").append(name).append(": ").append(types.get(name)).append(" ").append(shapes.get(name));
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// Session access
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Returns the underlying session.
	 *
	 * @param scope
	 *            the scope, used to report the use of a closed model
	 * @return the session
	 */
	public OrtSession getSession(final IScope scope) {
		if (closed) throw GamaRuntimeException.error(
				"The ONNX model '" + path + "' has been freed and cannot be used any more", scope);
		return session;
	}

	/**
	 * Returns the tensor info of an input, or null if that input does not exist or is not a tensor.
	 *
	 * @param name
	 *            the name of the input
	 * @return the tensor info, or null
	 */
	public TensorInfo getInputTensorInfo(final String name) {
		try {
			final NodeInfo info = session.getInputInfo().get(name);
			return info != null && info.getInfo() instanceof TensorInfo tensor ? tensor : null;
		} catch (final OrtException e) {
			return null;
		}
	}

	/**
	 * Whether the session has been closed.
	 *
	 * @return true if closed
	 */
	public boolean isClosed() { return closed; }

	/**
	 * Closes the session. Further uses of the model raise an error.
	 */
	void close() {
		if (closed) return;
		closed = true;
		try {
			session.close();
		} catch (final OrtException e) {
			// nothing sensible to do at this point: the model is unusable either way
		}
	}

	// ---------------------------------------------------------------------------------------------------------------
	// IValue
	// ---------------------------------------------------------------------------------------------------------------

	@Override
	public IContainerType<?> getGamlType() { return (IContainerType<?>) Types.get(OnnxConstants.MODEL_ID); }

	@Override
	public String stringValue(final IScope scope) {
		return getInfo();
	}

	@Override
	public String serializeToGaml(final boolean includingBuiltIn) {
		return OnnxConstants.MODEL + "('" + path + "')";
	}

	@Override
	public IJsonValue serializeToJson(final IJson json) {
		return json.typedObject(getGamlType(), PATH, path, INPUTS, inputNames, OUTPUTS, outputNames);
	}

	/**
	 * Returns this model. Sessions are shared and immutable, so copying one would only waste memory and time.
	 */
	@Override
	public IContainer<String, Object> copy(final IScope scope) {
		return this;
	}

	// ---------------------------------------------------------------------------------------------------------------
	// IContainer: the model is a read-only container of its nodes, keyed by name
	// ---------------------------------------------------------------------------------------------------------------

	@Override
	public Object get(final IScope scope, final String index) throws GamaRuntimeException {
		return nodes.get(index);
	}

	@Override
	public Object getFromIndicesList(final IScope scope, final IList<String> indices) throws GamaRuntimeException {
		if (indices == null || indices.isEmpty()) return null;
		return get(scope, indices.get(0));
	}

	@Override
	public boolean contains(final IScope scope, final Object o) {
		return nodes.containsValue(o);
	}

	@Override
	public boolean containsKey(final IScope scope, final Object o) {
		return nodes.containsKey(o);
	}

	@Override
	public Object firstValue(final IScope scope) {
		return nodes.isEmpty() ? null : nodes.values().iterator().next();
	}

	@Override
	public Object lastValue(final IScope scope) {
		Object last = null;
		for (final Object value : nodes.values()) { last = value; }
		return last;
	}

	@Override
	public Object anyValue(final IScope scope) {
		return firstValue(scope);
	}

	@Override
	public int length(final IScope scope) {
		return nodes.size();
	}

	@Override
	public boolean isEmpty(final IScope scope) {
		return nodes.isEmpty();
	}

	@Override
	public IContainer<?, ?> reverse(final IScope scope) {
		return nodes.reverse(scope);
	}

	@Override
	public IList<Object> listValue(final IScope scope, final IType<?> contentType, final boolean copy) {
		return nodes.listValue(scope, contentType, copy);
	}

	@Override
	public IMatrix<?> matrixValue(final IScope scope, final IType<?> contentType, final boolean copy) {
		return nodes.matrixValue(scope, contentType, copy);
	}

	@Override
	public IMatrix<?> matrixValue(final IScope scope, final IType<?> contentType, final IPoint size,
			final boolean copy) {
		return nodes.matrixValue(scope, contentType, size, copy);
	}

	@Override
	public <D, C> IMap<C, D> mapValue(final IScope scope, final IType<C> keyType, final IType<D> contentType,
			final boolean copy) {
		return nodes.mapValue(scope, keyType, contentType, copy);
	}

	@Override
	public Iterable<Object> iterable(final IScope scope) {
		return nodes.values();
	}

	// ---------------------------------------------------------------------------------------------------------------
	// IContainer.Modifiable: refused, a model describes a file on disk
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Raises the error shared by every modification operation.
	 *
	 * @param scope
	 *            the scope
	 */
	private void readOnly(final IScope scope) {
		throw GamaRuntimeException.error("An " + OnnxConstants.MODEL
				+ " describes a file on disk and cannot be modified. Load another file instead.", scope);
	}

	@Override
	public void addValue(final IScope scope, final Object value) {
		readOnly(scope);
	}

	@Override
	public void addValueAtIndex(final IScope scope, final Object index, final Object value) {
		readOnly(scope);
	}

	@Override
	public void setValueAtIndex(final IScope scope, final Object index, final Object value) {
		readOnly(scope);
	}

	@Override
	public void addValues(final IScope scope, final Object index, final IContainer<?, ?> values) {
		readOnly(scope);
	}

	@Override
	public void setAllValues(final IScope scope, final Object value) {
		readOnly(scope);
	}

	@Override
	public void removeValue(final IScope scope, final Object value) {
		readOnly(scope);
	}

	@Override
	public void removeIndex(final IScope scope, final Object index) {
		readOnly(scope);
	}

	@Override
	public void removeIndexes(final IScope scope, final IContainer<?, ?> index) {
		readOnly(scope);
	}

	@Override
	public void removeValues(final IScope scope, final IContainer<?, ?> values) {
		readOnly(scope);
	}

	@Override
	public void removeAllOccurrencesOfValue(final IScope scope, final Object value) {
		readOnly(scope);
	}

}
