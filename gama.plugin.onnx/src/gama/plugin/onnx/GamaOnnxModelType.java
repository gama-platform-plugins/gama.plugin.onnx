/*******************************************************************************************************
 *
 * GamaOnnxModelType.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import gama.annotations.doc;
import gama.annotations.type;
import gama.annotations.support.IConcept;
import gama.annotations.support.ISymbolKind;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.gaml.expressions.IExpression;
import gama.api.gaml.types.GamaContainerType;
import gama.api.gaml.types.IType;
import gama.api.gaml.types.ITypesManager;
import gama.api.gaml.types.Types;
import gama.api.runtime.scope.IScope;

/**
 * The GAML type of a loaded ONNX model.
 *
 * <p>
 * A model behaves as a read-only container of the nodes of its graph, keyed by node name, which is what allows
 * {@code onnx_file} to expose it as its contents. Its signature is read through its attributes ({@code inputs},
 * {@code input_shapes}, {@code metadata}, {@code info}) and inferences are run with {@code onnx_predict}.
 * </p>
 */
@type (
		name = OnnxConstants.MODEL,
		id = OnnxConstants.MODEL_ID,
		wraps = { GamaOnnxModel.class },
		kind = ISymbolKind.REGULAR,
		concept = { IConcept.TYPE, IConcept.FILE, OnnxConstants.CONCEPT },
		doc = @doc ("A neural network or any other computation graph stored in the ONNX format, loaded and ready to run. "
				+ "Obtained with the onnx_model operator or as the contents of an onnx_file, and run with onnx_predict."))
public class GamaOnnxModelType extends GamaContainerType<GamaOnnxModel> {

	/**
	 * Instantiates the type.
	 *
	 * @param typesManager
	 *            the types manager that owns this type
	 */
	public GamaOnnxModelType(final ITypesManager typesManager) {
		super(typesManager);
	}

	@Override
	@doc ("Casts the operand into an ONNX model. A string is understood as the path of a file to load, and an onnx_file as the model it contains.")
	public GamaOnnxModel cast(final IScope scope, final Object obj, final Object param, final IType<?> keyType,
			final IType<?> contentType, final boolean copy) throws GamaRuntimeException {
		return staticCast(scope, obj);
	}

	/**
	 * Casts any object into a model.
	 *
	 * @param scope
	 *            the scope
	 * @param obj
	 *            the object to cast
	 * @return the model, or null if the object cannot denote one
	 */
	public static GamaOnnxModel staticCast(final IScope scope, final Object obj) {
		return switch (obj) {
			case null -> null;
			case GamaOnnxModel model -> model;
			case GamaOnnxFile file -> file.getContents(scope);
			case String path -> OnnxRuntimeManager.load(scope, path, null);
			default -> null;
		};
	}

	@Override
	public IType<?> getKeyType() { return Types.STRING; }

	@Override
	public IType<?> keyTypeIfCasting(final IExpression exp) {
		return Types.STRING;
	}

	@Override
	public IType<?> contentsTypeIfCasting(final IExpression exp) {
		return Types.MAP;
	}

	@Override
	public GamaOnnxModel getDefault() { return null; }

	/**
	 * A model holds a native session, so it can never be turned into a compile-time constant.
	 */
	@Override
	public boolean canCastToConst() {
		return false;
	}

}
