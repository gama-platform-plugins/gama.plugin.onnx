/*******************************************************************************************************
 *
 * GamaOnnxFile.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import gama.annotations.doc;
import gama.annotations.example;
import gama.annotations.file;
import gama.annotations.support.IConcept;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.gaml.symbols.Facets;
import gama.api.gaml.types.IType;
import gama.api.gaml.types.Types;
import gama.api.runtime.scope.IScope;
import gama.api.types.file.GamaFile;
import gama.api.types.list.GamaListFactory;
import gama.api.types.list.IList;
import gama.api.types.map.IMap;
import gama.api.utils.geometry.IEnvelope;

/**
 * A {@code .onnx} file, whose contents is the loaded {@link GamaOnnxModel}.
 *
 * <p>
 * This is the idiomatic way of referring to a model in GAML, since path resolution, the generic {@code file(...)}
 * resolver and the navigator all come for free. The {@code onnx_model} operator is the direct shortcut. Both share the
 * session cache of the current experiment, so referring to the same file twice never loads it twice, and nothing
 * outlives the experiment.
 * </p>
 */
@file (
		name = "onnx",
		extensions = { "onnx" },
		buffer_type = OnnxConstants.MODEL_ID,
		buffer_index = IType.STRING,
		concept = { IConcept.FILE, OnnxConstants.CONCEPT },
		doc = @doc ("A file containing a computation graph in the ONNX format. Its contents is an onnx_model, ready to be run with onnx_predict."))
public class GamaOnnxFile extends GamaFile<GamaOnnxModel, Object> {

	/** The session options, or null. */
	private IMap<String, Object> options;

	/**
	 * Refers to an ONNX model on disk. The file is only read the first time its contents is needed.
	 *
	 * @param scope
	 *            the scope
	 * @param pathName
	 *            the path of the file
	 */
	@doc (
			value = "Refers to an ONNX model stored on disk. The model itself is only loaded the first time it is used.",
			examples = { @example (
					value = "onnx_file f <- onnx_file(\"../includes/mnist.onnx\");",
					isExecutable = false) })
	public GamaOnnxFile(final IScope scope, final String pathName) throws GamaRuntimeException {
		super(scope, pathName);
	}

	/**
	 * Refers to an ONNX model on disk, to be loaded with the given session options.
	 *
	 * @param scope
	 *            the scope
	 * @param pathName
	 *            the path of the file
	 * @param options
	 *            the session options, see the onnx_model operator
	 */
	@doc (
			value = "Refers to an ONNX model to be loaded with session options. Supported keys are 'intra_op_threads', 'inter_op_threads', 'memory_pattern' and 'optimization_level'.",
			examples = { @example (
					value = "onnx_file f <- onnx_file(\"../includes/mnist.onnx\", [\"intra_op_threads\"::1]);",
					isExecutable = false) })
	public GamaOnnxFile(final IScope scope, final String pathName, final IMap<String, Object> options) {
		super(scope, pathName);
		this.options = options;
	}

	@Override
	protected void fillBuffer(final IScope scope) throws GamaRuntimeException {
		if (getBuffer() != null) return;
		setBuffer(OnnxRuntimeManager.load(scope, getOriginalPath(), options));
	}

	/**
	 * ONNX files are produced by training toolchains, not by GAMA.
	 */
	@Override
	protected void flushBuffer(final IScope scope, final Facets facets) throws GamaRuntimeException {
		throw GamaRuntimeException.error(
				"An onnx_file cannot be saved: GAMA runs ONNX models but does not produce them.", scope);
	}

	/**
	 * Returns the names of the nodes of the graph, so that the file can be inspected like any other GAMA file.
	 */
	@Override
	public IList<String> getAttributes(final IScope scope) {
		final GamaOnnxModel model = getContents(scope);
		if (model == null) return GamaListFactory.getEmptyList();
		final IList<String> names = GamaListFactory.create(Types.STRING);
		names.addAll(model.getInputs());
		names.addAll(model.getOutputs());
		return names;
	}

	@Override
	public IEnvelope computeEnvelope(final IScope scope) {
		return null;
	}

}
