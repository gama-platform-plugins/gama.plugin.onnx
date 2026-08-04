/*******************************************************************************************************
 *
 * OnnxConverters.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import java.awt.image.BufferedImage;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.TensorInfo;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.gaml.types.IType;
import gama.api.gaml.types.Types;
import gama.api.runtime.scope.IScope;
import gama.api.types.list.GamaListFactory;
import gama.api.types.list.IList;
import gama.api.types.map.GamaMapFactory;
import gama.api.types.map.IMap;
import gama.api.types.matrix.IMatrix;
import gama.api.types.misc.IContainer;

/**
 * Converts GAMA values into ONNX tensors and back.
 *
 * <p>
 * The guiding rule is that <em>the model decides</em>. A GAML value passed to {@code onnx_predict} is flattened in
 * row-major order and then reshaped into the shape and element type declared by the corresponding input of the graph.
 * Dimensions declared as dynamic (-1) are deduced from the amount of data actually provided. This is what lets a
 * modeller write {@code onnx_predict(m, [1.0, 2.0, 3.0])} without ever building a tensor by hand.
 * </p>
 *
 * <p>
 * Outputs are converted back into nested lists whose nesting matches the shape of the tensor exactly. A {@code [1,10]}
 * output becomes a list containing one list of ten floats, not a flat list of ten. Sequences and maps, produced for
 * instance by scikit-learn models converted to ONNX, become GAMA lists and maps.
 * </p>
 */
public class OnnxConverters {

	private OnnxConverters() {}

	/**
	 * Returns the GAML-facing name of an ONNX element type.
	 *
	 * @param type
	 *            the ONNX type
	 * @return a name such as "float32" or "int64"
	 */
	public static String nameOf(final OnnxJavaType type) {
		return switch (type) {
			case FLOAT -> "float32";
			case DOUBLE -> "float64";
			case INT8 -> "int8";
			case INT16 -> "int16";
			case INT32 -> "int32";
			case INT64 -> "int64";
			case BOOL -> "bool";
			case STRING -> "string";
			case UINT8 -> "uint8";
			case FLOAT16 -> "float16";
			case BFLOAT16 -> "bfloat16";
			default -> "unknown";
		};
	}

	// ---------------------------------------------------------------------------------------------------------------
	// GAMA -> ONNX
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Builds the tensor to feed to an input of the graph.
	 *
	 * @param scope
	 *            the scope
	 * @param env
	 *            the ONNX environment
	 * @param nodeName
	 *            the name of the input, used in error messages
	 * @param info
	 *            the declared type and shape of the input
	 * @param value
	 *            the GAML value provided by the modeller
	 * @return a tensor the caller is responsible for closing
	 */
	public static OnnxTensor toTensor(final IScope scope, final OrtEnvironment env, final String nodeName,
			final TensorInfo info, final Object value) {
		if (value == null)
			throw GamaRuntimeException.error("The input '" + nodeName + "' of the ONNX model received nil", scope);
		if (info.type == OnnxJavaType.STRING) return toStringTensor(scope, env, nodeName, info, value);
		final Flat flat = new Flat();
		final long[] shape;
		if (value instanceof BufferedImage image) {
			// an image carries its own height, width and channel count, so the shape is built from it
			// rather than deduced from the number of values: a fully dynamic vision input works too
			shape = flattenImage(scope, nodeName, info, image, flat);
		} else {
			flattenNumbers(scope, nodeName, value, flat);
			shape = resolveShape(scope, nodeName, info.getShape(), flat.size, value);
		}
		try {
			return build(scope, env, nodeName, info.type, flat, shape);
		} catch (final OrtException e) {
			throw GamaRuntimeException.error(
					"Cannot build the tensor for the input '" + nodeName + "': " + e.getMessage(), scope);
		}
	}

	/**
	 * Builds a tensor of the requested element type from the flattened values.
	 *
	 * @param scope
	 *            the scope
	 * @param env
	 *            the environment
	 * @param nodeName
	 *            the input name
	 * @param type
	 *            the element type declared by the graph
	 * @param flat
	 *            the flattened values
	 * @param shape
	 *            the resolved shape
	 * @return the tensor
	 */
	private static OnnxTensor build(final IScope scope, final OrtEnvironment env, final String nodeName,
			final OnnxJavaType type, final Flat flat, final long[] shape) throws OrtException {
		final int n = flat.size;
		switch (type) {
			case FLOAT: {
				final FloatBuffer buffer = FloatBuffer.allocate(n);
				for (int i = 0; i < n; i++) { buffer.put((float) flat.values[i]); }
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape);
			}
			case DOUBLE: {
				final DoubleBuffer buffer = DoubleBuffer.allocate(n);
				buffer.put(flat.values, 0, n);
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape);
			}
			case INT32: {
				final IntBuffer buffer = IntBuffer.allocate(n);
				for (int i = 0; i < n; i++) { buffer.put((int) flat.values[i]); }
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape);
			}
			case INT64: {
				final LongBuffer buffer = LongBuffer.allocate(n);
				for (int i = 0; i < n; i++) { buffer.put((long) flat.values[i]); }
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape);
			}
			case INT16: {
				final ShortBuffer buffer = ShortBuffer.allocate(n);
				for (int i = 0; i < n; i++) { buffer.put((short) flat.values[i]); }
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape);
			}
			case INT8:
			case UINT8: {
				final ByteBuffer buffer = ByteBuffer.allocate(n);
				for (int i = 0; i < n; i++) { buffer.put((byte) flat.values[i]); }
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape, type);
			}
			case BOOL: {
				// booleans travel as one byte per element
				final ByteBuffer buffer = ByteBuffer.allocate(n);
				for (int i = 0; i < n; i++) { buffer.put((byte) (flat.values[i] != 0 ? 1 : 0)); }
				buffer.rewind();
				return OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.BOOL);
			}
			default:
				throw GamaRuntimeException.error("The input '" + nodeName + "' has the element type " + nameOf(type)
						+ ", which this plugin cannot build from GAMA values yet.", scope);
		}
	}

	/**
	 * Builds a tensor of strings.
	 *
	 * @param scope
	 *            the scope
	 * @param env
	 *            the environment
	 * @param nodeName
	 *            the input name
	 * @param info
	 *            the declared info
	 * @param value
	 *            the GAML value
	 * @return the tensor
	 */
	private static OnnxTensor toStringTensor(final IScope scope, final OrtEnvironment env, final String nodeName,
			final TensorInfo info, final Object value) {
		final List<String> strings = new ArrayList<>();
		flattenStrings(scope, value, strings);
		final long[] shape = resolveShape(scope, nodeName, info.getShape(), strings.size(), value);
		try {
			return OnnxTensor.createTensor(env, strings.toArray(new String[0]), shape);
		} catch (final OrtException e) {
			throw GamaRuntimeException.error(
					"Cannot build the string tensor for the input '" + nodeName + "': " + e.getMessage(), scope);
		}
	}

	/**
	 * Flattens a GAML value into a sequence of strings.
	 *
	 * @param scope
	 *            the scope
	 * @param value
	 *            the value
	 * @param out
	 *            the accumulator
	 */
	private static void flattenStrings(final IScope scope, final Object value, final List<String> out) {
		switch (value) {
			case null -> out.add(null);
			case IContainer<?, ?> container -> {
				for (final Object item : container.iterable(scope)) { flattenStrings(scope, item, out); }
			}
			case Iterable<?> iterable -> {
				for (final Object item : iterable) { flattenStrings(scope, item, out); }
			}
			default -> out.add(value.toString());
		}
	}

	/**
	 * Flattens a GAML value into a sequence of doubles, in row-major order.
	 *
	 * <p>
	 * Matrices are read column by column within each row so that a {@code matrix} reads like the nested list a
	 * modeller would have written by hand.
	 * </p>
	 *
	 * @param scope
	 *            the scope
	 * @param nodeName
	 *            the input name, for error messages
	 * @param value
	 *            the value to flatten
	 * @param out
	 *            the accumulator
	 */
	private static void flattenNumbers(final IScope scope, final String nodeName, final Object value, final Flat out) {
		switch (value) {
			case null -> throw GamaRuntimeException
					.error("The input '" + nodeName + "' of the ONNX model contains a nil value", scope);
			case Number number -> out.add(number.doubleValue());
			case Boolean bool -> out.add(bool ? 1 : 0);
			case IMatrix<?> matrix -> {
				final int rows = matrix.getRows(scope);
				final int cols = matrix.getCols(scope);
				for (int row = 0; row < rows; row++) {
					for (int col = 0; col < cols; col++) {
						flattenNumbers(scope, nodeName, matrix.get(scope, col, row), out);
					}
				}
			}
			case IContainer<?, ?> container -> {
				for (final Object item : container.iterable(scope)) { flattenNumbers(scope, nodeName, item, out); }
			}
			case Iterable<?> iterable -> {
				for (final Object item : iterable) { flattenNumbers(scope, nodeName, item, out); }
			}
			default -> throw GamaRuntimeException.error("The input '" + nodeName + "' of the ONNX model cannot accept a "
					+ value.getClass().getSimpleName() + ". Expected a number, a list, a matrix, a field or an image.",
					scope);
		}
	}

	/**
	 * Flattens an image into the layout expected by the input of the graph.
	 *
	 * <p>
	 * The layout is deduced from the declared shape: a rank-4 shape is read as NCHW unless its last dimension is a
	 * plausible channel count (1, 3 or 4) and its second one is not, in which case it is read as NHWC. Rank 3 works the
	 * same way without the batch dimension, and rank 2 is grayscale. Channel values are normalised to [0,1]; one
	 * channel means luminance, three mean RGB and four mean RGBA.
	 * </p>
	 *
	 * @param scope
	 *            the scope
	 * @param nodeName
	 *            the input name
	 * @param info
	 *            the declared info
	 * @param image
	 *            the image
	 * @param out
	 *            the accumulator
	 * @return the concrete shape of the tensor to build
	 */
	private static long[] flattenImage(final IScope scope, final String nodeName, final TensorInfo info,
			final BufferedImage image, final Flat out) {
		final long[] declared = info.getShape();
		final int rank = declared.length;
		if (rank < 2 || rank > 4) throw GamaRuntimeException.error("The input '" + nodeName + "' has a rank-" + rank
				+ " shape " + Arrays.toString(declared) + ", which an image cannot feed. Expected rank 2, 3 or 4.",
				scope);

		final int channelAxis = channelAxisOf(declared, rank);
		final int channels = channelAxis < 0 ? 1 : declared[channelAxis] < 0 ? 3 : (int) declared[channelAxis];
		if (channels != 1 && channels != 3 && channels != 4)
			throw GamaRuntimeException.error("The input '" + nodeName + "' expects " + channels
					+ " channels, which cannot be filled from an image (expected 1, 3 or 4).", scope);

		// height and width are the two remaining spatial dimensions, in that order
		final int[] spatial = spatialAxesOf(declared, rank, channelAxis);
		final int expectedHeight = (int) declared[spatial[0]];
		final int expectedWidth = (int) declared[spatial[1]];
		final int height = image.getHeight();
		final int width = image.getWidth();
		if (expectedHeight > 0 && expectedHeight != height || expectedWidth > 0 && expectedWidth != width)
			throw GamaRuntimeException.error("The input '" + nodeName + "' expects an image of "
					+ dimension(expectedWidth) + "x" + dimension(expectedHeight) + " but received one of " + width + "x"
					+ height + ". Resize it first, for instance with the 'with_size' operator of the image extension.",
					scope);

		// every dimension is now known, including the dynamic ones
		final long[] shape = declared.clone();
		if (channelAxis >= 0) { shape[channelAxis] = channels; }
		shape[spatial[0]] = height;
		shape[spatial[1]] = width;
		if (rank == 4) { shape[0] = 1; }

		final boolean channelsLast = channelAxis == rank - 1;
		if (channelsLast) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					final int pixel = image.getRGB(x, y);
					for (int c = 0; c < channels; c++) { out.add(channelOf(pixel, c, channels)); }
				}
			}
		} else {
			for (int c = 0; c < channels; c++) {
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) { out.add(channelOf(image.getRGB(x, y), c, channels)); }
				}
			}
		}
		return shape;
	}

	/**
	 * Renders a declared dimension, showing dynamic ones as a question mark rather than as -1.
	 *
	 * @param dimension
	 *            the dimension
	 * @return its textual form
	 */
	private static String dimension(final int dimension) {
		return dimension < 0 ? "?" : String.valueOf(dimension);
	}

	/**
	 * Finds the axis holding the channels, or -1 for a rank-2 (grayscale) shape.
	 *
	 * @param shape
	 *            the declared shape
	 * @param rank
	 *            its rank
	 * @return the index of the channel axis, or -1
	 */
	private static int channelAxisOf(final long[] shape, final int rank) {
		if (rank == 2) return -1;
		final int first = rank == 4 ? 1 : 0;
		final int last = rank - 1;
		final boolean firstPlausible = isChannelCount(shape[first]);
		final boolean lastPlausible = isChannelCount(shape[last]);
		// NCHW is by far the most common convention, so it wins whenever both readings are plausible
		if (firstPlausible) return first;
		if (lastPlausible) return last;
		return first;
	}

	/**
	 * Whether a dimension is a plausible number of image channels.
	 *
	 * @param dimension
	 *            the dimension
	 * @return true if 1, 3 or 4
	 */
	private static boolean isChannelCount(final long dimension) {
		return dimension == 1 || dimension == 3 || dimension == 4;
	}

	/**
	 * Returns the axes holding the height and the width, in that order.
	 *
	 * @param shape
	 *            the shape
	 * @param rank
	 *            its rank
	 * @param channelAxis
	 *            the channel axis, or -1
	 * @return the two spatial axes
	 */
	private static int[] spatialAxesOf(final long[] shape, final int rank, final int channelAxis) {
		final int[] axes = new int[2];
		int found = 0;
		final int start = rank == 4 ? 1 : 0;
		for (int axis = start; axis < rank && found < 2; axis++) {
			if (axis != channelAxis) { axes[found++] = axis; }
		}
		return axes;
	}

	/**
	 * Extracts one normalised channel from a packed ARGB pixel.
	 *
	 * @param pixel
	 *            the ARGB pixel
	 * @param channel
	 *            the channel index
	 * @param channels
	 *            the total number of channels
	 * @return a value in [0,1]
	 */
	private static double channelOf(final int pixel, final int channel, final int channels) {
		final int red = pixel >> 16 & 0xFF;
		final int green = pixel >> 8 & 0xFF;
		final int blue = pixel & 0xFF;
		if (channels == 1) return (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0;
		return switch (channel) {
			case 0 -> red / 255.0;
			case 1 -> green / 255.0;
			case 2 -> blue / 255.0;
			default -> (pixel >> 24 & 0xFF) / 255.0;
		};
	}

	/**
	 * Turns the shape declared by the graph into a concrete one, deducing the dynamic dimensions from the amount of
	 * data provided.
	 *
	 * @param scope
	 *            the scope
	 * @param nodeName
	 *            the input name
	 * @param declared
	 *            the declared shape, possibly containing -1
	 * @param count
	 *            the number of values provided
	 * @param value
	 *            the original value, for the error message
	 * @return a shape with no dynamic dimension left
	 */
	private static long[] resolveShape(final IScope scope, final String nodeName, final long[] declared,
			final int count, final Object value) {
		if (declared.length == 0) {
			if (count != 1) throw GamaRuntimeException.error("The input '" + nodeName + "' is a scalar but " + count
					+ " values were provided.", scope);
			return new long[0];
		}
		final long[] shape = declared.clone();
		long fixed = 1;
		int dynamic = 0;
		int firstDynamic = -1;
		for (int i = 0; i < shape.length; i++) {
			if (shape[i] < 0) {
				dynamic++;
				if (firstDynamic < 0) { firstDynamic = i; }
			} else {
				fixed *= shape[i];
			}
		}
		if (dynamic == 0) {
			if (fixed != count) throw GamaRuntimeException.error("The input '" + nodeName + "' expects "
					+ fixed + " values (shape " + Arrays.toString(declared) + ") but " + count + " were provided"
					+ describe(value) + ".", scope);
			return shape;
		}
		// every dynamic dimension but the first is assumed to be 1, which covers the usual "dynamic batch" case
		for (int i = 0; i < shape.length; i++) {
			if (shape[i] < 0 && i != firstDynamic) { shape[i] = 1; }
		}
		if (fixed == 0 || count % fixed != 0)
			throw GamaRuntimeException.error("The input '" + nodeName + "' has shape " + Arrays.toString(declared)
					+ ", so the number of values must be a multiple of " + fixed + ", but " + count + " were provided"
					+ describe(value) + ".", scope);
		shape[firstDynamic] = count / fixed;
		return shape;
	}

	/**
	 * Describes the origin of a value, to make the shape errors actionable.
	 *
	 * @param value
	 *            the value
	 * @return a suffix such as " (from a matrix)"
	 */
	private static String describe(final Object value) {
		return switch (value) {
			case null -> "";
			case BufferedImage image -> " (from a " + image.getWidth() + "x" + image.getHeight() + " image)";
			case IMatrix<?> matrix -> " (from a matrix)";
			case IContainer<?, ?> container -> " (from a container)";
			default -> "";
		};
	}

	// ---------------------------------------------------------------------------------------------------------------
	// ONNX -> GAMA
	// ---------------------------------------------------------------------------------------------------------------

	/**
	 * Converts a value produced by the graph into GAMA values.
	 *
	 * @param scope
	 *            the scope
	 * @param value
	 *            the ONNX value
	 * @return nested lists for a tensor, a list for a sequence, a map for a map
	 */
	public static Object fromOnnx(final IScope scope, final OnnxValue value) {
		try {
			return switch (value) {
				case null -> null;
				case OnnxTensor tensor -> convert(scope, tensor.getValue());
				case OnnxSequence sequence -> {
					final IList<Object> list = GamaListFactory.create(Types.NO_TYPE);
					for (final OnnxValue item : sequence.getValue()) { list.add(fromOnnx(scope, item)); }
					yield list;
				}
				case OnnxMap map -> {
					final IMap<Object, Object> result = GamaMapFactory.create(Types.NO_TYPE, Types.NO_TYPE);
					for (final Map.Entry<?, ?> entry : map.getValue().entrySet()) {
						result.put(convert(scope, entry.getKey()), convert(scope, entry.getValue()));
					}
					yield result;
				}
				default -> convert(scope, value.getValue());
			};
		} catch (final OrtException e) {
			throw GamaRuntimeException.error("Cannot read the output of the ONNX model: " + e.getMessage(), scope);
		}
	}

	/**
	 * Converts a Java value produced by the runtime, either a scalar or a (possibly nested) array, into GAMA values.
	 *
	 * @param scope
	 *            the scope
	 * @param value
	 *            the java value
	 * @return the GAMA value
	 */
	private static Object convert(final IScope scope, final Object value) {
		if (value == null) return null;
		if (value.getClass().isArray()) {
			final int length = Array.getLength(value);
			final IList<Object> list = GamaListFactory.create(contentTypeOf(value.getClass().getComponentType()));
			for (int i = 0; i < length; i++) { list.add(convert(scope, Array.get(value, i))); }
			return list;
		}
		return switch (value) {
			case Long l -> toGamlInt(scope, l);
			case Integer i -> i;
			case Short s -> (int) s;
			case Byte b -> (int) b;
			case Float f -> Double.valueOf(f);
			case Double d -> d;
			case Boolean b -> b;
			case String s -> s;
			default -> value;
		};
	}

	/**
	 * Converts a 64-bit integer produced by the model into a GAML {@code int}, which is 32 bits.
	 *
	 * @param scope
	 *            the scope
	 * @param value
	 *            the value
	 * @return the value as an Integer
	 */
	private static Integer toGamlInt(final IScope scope, final long value) {
		if (value < Integer.MIN_VALUE || value > Integer.MAX_VALUE)
			throw GamaRuntimeException.error("The ONNX model returned the int64 value " + value
					+ ", which does not fit in a GAML int (32 bits).", scope);
		return (int) value;
	}

	/**
	 * Maps the component type of a Java array to the GAML type of the list that will hold it.
	 *
	 * @param component
	 *            the component type
	 * @return the GAML content type
	 */
	private static IType<?> contentTypeOf(final Class<?> component) {
		if (component.isArray()) return Types.LIST;
		if (component == float.class || component == double.class || component == Float.class
				|| component == Double.class)
			return Types.FLOAT;
		if (component == long.class || component == int.class || component == short.class || component == byte.class
				|| component == Long.class || component == Integer.class)
			return Types.INT;
		if (component == boolean.class || component == Boolean.class) return Types.BOOL;
		if (component == String.class) return Types.STRING;
		return Types.NO_TYPE;
	}

	/**
	 * A growable array of doubles, used to flatten GAML values without boxing every element.
	 */
	private static final class Flat {

		/** The values. */
		private double[] values = new double[64];

		/** The number of values actually held. */
		private int size;

		/**
		 * Appends a value.
		 *
		 * @param value
		 *            the value
		 */
		void add(final double value) {
			if (size == values.length) { values = Arrays.copyOf(values, size * 2); }
			values[size++] = value;
		}
	}

}
