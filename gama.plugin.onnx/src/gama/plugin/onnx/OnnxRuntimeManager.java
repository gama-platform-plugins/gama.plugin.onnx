/*******************************************************************************************************
 *
 * OnnxRuntimeManager.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation
 * platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import java.io.File;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.kernel.simulation.IExperimentAgent;
import gama.api.runtime.scope.IScope;
import gama.api.types.map.IMap;
import gama.api.utils.files.FileUtils;

/**
 * Owns the ONNX Runtime environment and the sessions loaded by an experiment.
 *
 * <p>
 * Building an {@link OrtSession} means parsing the model and letting the runtime optimise its graph, which is orders of
 * magnitude more expensive than running an inference. Since GAML expressions can be re-evaluated at every step, sessions
 * have to be cached rather than rebuilt. But a session is a native resource, so <em>where</em> that cache lives
 * decides who owns it.
 * </p>
 *
 * <p>
 * The cache is held by the <b>experiment agent</b>, and closed when that agent is disposed. Nothing survives an
 * experiment. This is the widest scope a native resource may have in GAMA: a static cache would leak sessions for the
 * lifetime of the JVM, and would let one simulation close sessions still in use by the others running beside it in a
 * batch. Within one experiment, the simulations of a batch do share their sessions, which is what makes exploring a
 * thousand parameter sets against the same model affordable. Running them in parallel is safe as well, since
 * {@code OrtSession.run} is thread-safe.
 * </p>
 */
public class OnnxRuntimeManager {

	/** The key under which the sessions of an experiment are stored on its agent. */
	private static final String SESSIONS = "__gama_plugin_onnx_sessions__";

	/**
	 * The environment. Genuinely process-wide: {@code OrtEnvironment.getEnvironment()} is a singleton in the runtime
	 * itself, and it holds no model, only the thread pools and the logger. It is never closed, since another
	 * experiment may start at any time.
	 */
	private static volatile OrtEnvironment ENVIRONMENT;

	private OnnxRuntimeManager() {}

	/**
	 * Returns the ONNX Runtime environment, creating it on first use.
	 *
	 * <p>
	 * The runtime extracts its native libraries from the jar using its own classloader. Under OSGi the thread context
	 * classloader is usually unrelated to the bundle, so it is temporarily swapped for the one that loaded this class.
	 * The jar sits on this bundle's {@code Bundle-ClassPath}, so that classloader can always see the natives.
	 * </p>
	 *
	 * @param scope
	 *            the scope, used to report a failure to load the native libraries
	 * @return the environment
	 */
	public static OrtEnvironment getEnvironment(final IScope scope) {
		if (ENVIRONMENT == null) {
			synchronized (OnnxRuntimeManager.class) {
				if (ENVIRONMENT == null) {
					final Thread current = Thread.currentThread();
					final ClassLoader previous = current.getContextClassLoader();
					try {
						current.setContextClassLoader(OnnxRuntimeManager.class.getClassLoader());
						ENVIRONMENT = OrtEnvironment.getEnvironment();
					} catch (final Throwable t) {
						throw GamaRuntimeException.error("The ONNX Runtime native libraries could not be loaded: "
								+ t.getMessage() + ". Supported platforms are win-x64, linux-x64, linux-aarch64 and "
								+ "macos-aarch64.", scope);
					} finally {
						current.setContextClassLoader(previous);
					}
				}
			}
		}
		return ENVIRONMENT;
	}

	/**
	 * Returns the sessions of the current experiment, creating the cache and arming its disposal on first use.
	 *
	 * @param scope
	 *            the scope
	 * @return the mutable cache, keyed by absolute path and options
	 */
	@SuppressWarnings ("unchecked")
	private static Map<String, GamaOnnxModel> sessionsOf(final IScope scope) {
		final IExperimentAgent experiment = scope == null ? null : scope.getExperiment();
		if (experiment == null) throw GamaRuntimeException.error(
				"ONNX models can only be loaded from within an experiment, since their sessions belong to it.", scope);
		Map<String, GamaOnnxModel> sessions = (Map<String, GamaOnnxModel>) experiment.getAttribute(SESSIONS);
		if (sessions != null) return sessions;
		// two agents of the same experiment may ask for the first model at the same time
		synchronized (experiment) {
			sessions = (Map<String, GamaOnnxModel>) experiment.getAttribute(SESSIONS);
			if (sessions == null) {
				final Map<String, GamaOnnxModel> created = new ConcurrentHashMap<>();
				experiment.setAttribute(SESSIONS, created);
				experiment.postDisposeAction(s -> {
					closeAll(created.values());
					created.clear();
					return null;
				});
				sessions = created;
			}
		}
		return sessions;
	}

	/**
	 * Loads a model, reusing the session if this experiment already loaded the same file with the same options.
	 *
	 * @param scope
	 *            the scope
	 * @param path
	 *            the path of the model, as written in the model (relative paths are resolved against it)
	 * @param options
	 *            the session options, or null
	 * @return the model, never null
	 */
	public static GamaOnnxModel load(final IScope scope, final String path, final IMap<String, Object> options) {
		if (path == null) throw GamaRuntimeException.error("Cannot load an ONNX model from a nil path", scope);
		final Map<String, GamaOnnxModel> sessions = sessionsOf(scope);
		final String absolute = resolve(scope, path);
		final String signature = signatureOf(options);
		final String key = absolute + " " + signature;
		final GamaOnnxModel cached = sessions.get(key);
		if (cached != null && !cached.isClosed()) return cached;
		final GamaOnnxModel model = build(scope, path, absolute, options, signature);
		// putIfAbsent so that two threads racing on the same model share the winner and the loser is closed
		final GamaOnnxModel concurrent = sessions.putIfAbsent(key, model);
		if (concurrent != null && !concurrent.isClosed()) {
			model.close();
			return concurrent;
		}
		sessions.put(key, model);
		return model;
	}

	/**
	 * Resolves a possibly relative path against the model file that contains the current experiment.
	 *
	 * @param scope
	 *            the scope
	 * @param path
	 *            the path
	 * @return the absolute path of an existing readable file
	 */
	private static String resolve(final IScope scope, final String path) {
		final File file = new File(FileUtils.constructAbsoluteFilePath(scope, path, true));
		if (!file.exists())
			throw GamaRuntimeException.error("The ONNX model '" + path + "' does not exist (looked for "
					+ file.getAbsolutePath() + ")", scope);
		if (!file.canRead())
			throw GamaRuntimeException.error("The ONNX model '" + file.getAbsolutePath() + "' cannot be read", scope);
		return file.getAbsolutePath();
	}

	/**
	 * Builds a new session.
	 *
	 * @param scope
	 *            the scope
	 * @param path
	 *            the path as written by the modeller, kept for error messages
	 * @param absolute
	 *            the resolved absolute path
	 * @param options
	 *            the options, or null
	 * @param signature
	 *            the canonical form of the options
	 * @return the model
	 */
	private static GamaOnnxModel build(final IScope scope, final String path, final String absolute,
			final IMap<String, Object> options, final String signature) {
		final OrtEnvironment env = getEnvironment(scope);
		try (SessionOptions sessionOptions = new SessionOptions()) {
			apply(scope, sessionOptions, options);
			final OrtSession session = env.createSession(absolute, sessionOptions);
			return new GamaOnnxModel(scope, path, absolute, signature, session);
		} catch (final OrtException e) {
			throw GamaRuntimeException.error("Cannot load the ONNX model '" + path + "': " + e.getMessage(), scope);
		}
	}

	/**
	 * Applies the GAML options to the session options.
	 *
	 * @param scope
	 *            the scope
	 * @param target
	 *            the session options to configure
	 * @param options
	 *            the GAML options, or null
	 */
	private static void apply(final IScope scope, final SessionOptions target, final IMap<String, Object> options)
			throws OrtException {
		if (options == null || options.isEmpty()) return;
		for (final Map.Entry<String, Object> entry : options.entrySet()) {
			final String key = entry.getKey();
			final Object value = entry.getValue();
			switch (key) {
				case OnnxConstants.OPT_INTRA_OP_THREADS -> target.setIntraOpNumThreads(intOf(scope, key, value));
				case OnnxConstants.OPT_INTER_OP_THREADS -> target.setInterOpNumThreads(intOf(scope, key, value));
				case OnnxConstants.OPT_MEMORY_PATTERN -> target.setMemoryPatternOptimization(Boolean.TRUE.equals(value));
				case OnnxConstants.OPT_OPTIMIZATION_LEVEL -> target.setOptimizationLevel(levelOf(scope, value));
				default -> throw GamaRuntimeException.error("Unknown ONNX session option '" + key + "'. Expected one of "
						+ OnnxConstants.OPT_INTRA_OP_THREADS + ", " + OnnxConstants.OPT_INTER_OP_THREADS + ", "
						+ OnnxConstants.OPT_MEMORY_PATTERN + ", " + OnnxConstants.OPT_OPTIMIZATION_LEVEL, scope);
			}
		}
	}

	/**
	 * Reads an integer option.
	 *
	 * @param scope
	 *            the scope
	 * @param key
	 *            the option name, for the error message
	 * @param value
	 *            the value
	 * @return the integer
	 */
	private static int intOf(final IScope scope, final String key, final Object value) {
		if (value instanceof Number n) return n.intValue();
		throw GamaRuntimeException.error("The ONNX session option '" + key + "' expects an int, got " + value, scope);
	}

	/**
	 * Reads the optimisation level option.
	 *
	 * @param scope
	 *            the scope
	 * @param value
	 *            the value
	 * @return the level
	 */
	private static OptLevel levelOf(final IScope scope, final Object value) {
		final String level = value == null ? "" : value.toString().toLowerCase();
		return switch (level) {
			case "none" -> OptLevel.NO_OPT;
			case "basic" -> OptLevel.BASIC_OPT;
			case "extended" -> OptLevel.EXTENDED_OPT;
			case "layout" -> OptLevel.LAYOUT_OPT;
			case "all" -> OptLevel.ALL_OPT;
			default -> throw GamaRuntimeException.error("Unknown ONNX optimization_level '" + value
					+ "'. Expected 'none', 'basic', 'extended', 'layout' or 'all'.", scope);
		};
	}

	/**
	 * Builds a canonical representation of the options, so that two equivalent maps share the same session.
	 *
	 * @param options
	 *            the options, or null
	 * @return a stable string
	 */
	private static String signatureOf(final IMap<String, Object> options) {
		if (options == null || options.isEmpty()) return "";
		final List<String> entries = new ArrayList<>();
		options.forEach((k, v) -> entries.add(k + "=" + v));
		entries.sort(null);
		return String.join(",", entries);
	}

	/**
	 * Removes a model from the cache of the current experiment and closes its session.
	 *
	 * @param scope
	 *            the scope
	 * @param model
	 *            the model to free, may be null
	 * @return true if a session was actually closed
	 */
	public static boolean free(final IScope scope, final GamaOnnxModel model) {
		if (model == null || model.isClosed()) return false;
		sessionsOf(scope).values().removeIf(m -> m == model);
		model.close();
		return true;
	}

	/**
	 * Closes the sessions of the current experiment whose path contains the given filter.
	 *
	 * @param scope
	 *            the scope
	 * @param pathFilter
	 *            a substring the path must contain, or null/empty to close every session of the experiment
	 * @return the number of sessions that were closed
	 */
	public static int freeAll(final IScope scope, final String pathFilter) {
		final boolean all = pathFilter == null || pathFilter.isEmpty();
		final Collection<GamaOnnxModel> selected = new ArrayList<>();
		sessionsOf(scope).values().removeIf(model -> {
			if (!all && !model.getPath().contains(pathFilter)) return false;
			selected.add(model);
			return true;
		});
		return closeAll(selected);
	}

	/**
	 * Closes a set of models, ignoring the ones already closed.
	 *
	 * @param models
	 *            the models
	 * @return the number of sessions that were closed
	 */
	private static int closeAll(final Collection<GamaOnnxModel> models) {
		int closed = 0;
		for (final GamaOnnxModel model : new ArrayList<>(models)) {
			if (!model.isClosed()) {
				model.close();
				closed++;
			}
		}
		return closed;
	}

}
