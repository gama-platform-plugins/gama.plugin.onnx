/*******************************************************************************************************
 *
 * OnnxSkill.java, in gama.plugin.onnx, is part of the source code of the GAMA modeling and simulation platform.
 *
 * (c) 2007-2026 UMI 209 UMMISCO IRD/SU & Partners (IRIT, MIAT, ESPACE-DEV, CTU)
 *
 * Visit https://github.com/gama-platform/gama.plugin.onnx for license information and contacts.
 *
 ********************************************************************************************************/
package gama.plugin.onnx;

import gama.annotations.action;
import gama.annotations.arg;
import gama.annotations.doc;
import gama.annotations.example;
import gama.annotations.getter;
import gama.annotations.setter;
import gama.annotations.skill;
import gama.annotations.variable;
import gama.annotations.vars;
import gama.annotations.support.IConcept;
import gama.api.exceptions.GamaRuntimeException;
import gama.api.gaml.types.IType;
import gama.api.kernel.agent.IAgent;
import gama.api.kernel.skill.Skill;
import gama.api.runtime.scope.IScope;
import gama.api.types.dataframe.IDataFrame;
import gama.api.types.map.IMap;

/**
 * Gives an agent its own ONNX model and lets it run inferences with statements rather than operators.
 *
 * <p>
 * This is syntactic sugar over {@link OnnxOperators}: everything it does can be written with {@code onnx_model} and
 * {@code onnx_predict}. It is worth using when a species carries a model of its own, such as a learned policy or a
 * surrogate of a costly submodel, because {@code do predict} then reads better than threading a model attribute
 * through every expression.
 * </p>
 *
 * <pre>
 * species walker skills: [onnx] {
 *     init { do load_model(path: "../includes/policy.onnx"); }
 *     reflex act {
 *         dataframe out &lt;- predict([speed, heading, distance_to_target]);
 *         list&lt;float&gt; action &lt;- out["action"][0];
 *     }
 * }
 * </pre>
 */
@skill (
		name = OnnxSkill.NAME,
		concept = { OnnxConstants.CONCEPT, IConcept.SKILL },
		doc = @doc ("Allows an agent to hold an ONNX model and run inferences on it. Every agent of the experiment that loads the same file shares one session, so giving this skill to thousands of agents does not load thousands of models. The sessions are released when the experiment is disposed."))
@vars ({ @variable (
		name = OnnxSkill.MODEL,
		type = OnnxConstants.MODEL_ID,
		doc = @doc ("The model this agent runs its inferences on. Can be set directly, or through the load_model action.")) })
public class OnnxSkill extends Skill {

	/** The name of the skill. */
	public static final String NAME = "onnx";

	/** The name of the attribute holding the model. */
	public static final String MODEL = "inner_model";

	/**
	 * Returns the model held by the agent.
	 *
	 * @param agent
	 *            the agent
	 * @return the model, or null if none was loaded
	 */
	@getter (MODEL)
	public GamaOnnxModel getModel(final IAgent agent) {
		return (GamaOnnxModel) agent.getAttribute(MODEL);
	}

	/**
	 * Sets the model held by the agent.
	 *
	 * @param agent
	 *            the agent
	 * @param model
	 *            the model
	 */
	@setter (MODEL)
	public void setModel(final IAgent agent, final GamaOnnxModel model) {
		agent.setAttribute(MODEL, model);
	}

	/**
	 * Loads a model and stores it in the agent.
	 *
	 * @param scope
	 *            the scope
	 * @return the loaded model
	 */
	@action (
			name = "load_model",
			args = { @arg (
					name = "path",
					type = IType.STRING,
					optional = false,
					doc = @doc ("The path of the .onnx file to load")),
					@arg (
							name = "options",
							type = IType.MAP,
							optional = true,
							doc = @doc ("The session options, as in the onnx_model operator")) })
	@doc (
			value = "Loads an ONNX model and stores it in the '" + MODEL + "' attribute of the agent. Returns the model.",
			examples = { @example (
					value = "do load_model(path: \"../includes/policy.onnx\");",
					isExecutable = false) })
	public GamaOnnxModel primLoadModel(final IScope scope) throws GamaRuntimeException {
		final IAgent agent = scope.getAgent();
		final String path = scope.getStringArg("path");
		@SuppressWarnings ("unchecked")
		final IMap<String, Object> options = (IMap<String, Object>) scope.getArgIfExists("options", IType.MAP);
		final GamaOnnxModel model = OnnxRuntimeManager.load(scope, path, options);
		agent.setAttribute(MODEL, model);
		return model;
	}

	/**
	 * Runs the model held by the agent.
	 *
	 * <p>
	 * Always a dataframe, like the two-operand form of the operator. Picking a single output is done on the result
	 * ({@code result["logits"]}) rather than through an argument, so that this action has one return type and only one.
	 * </p>
	 *
	 * @param scope
	 *            the scope
	 * @return a dataframe with one column per output of the model and one row per element of the batch
	 */
	@action (
			name = "predict",
			args = { @arg (
					name = "input",
					type = IType.NONE,
					optional = false,
					doc = @doc ("The input data, or a map from input name to data if the model has several inputs")) })
	@doc (
			value = "Runs the model held by the agent on the given input and returns a dataframe with one column per output and one row per element of the batch. Same conversion rules as the onnx_predict operator.",
			examples = { @example (
					value = "dataframe out <- predict([0.2, 0.9, 0.1]);",
					isExecutable = false),
					@example (
							value = "list<float> logits <- out[\"logits\"][0];",
							isExecutable = false) })
	public IDataFrame primPredict(final IScope scope) throws GamaRuntimeException {
		final GamaOnnxModel model = getModel(scope.getAgent());
		if (model == null) throw GamaRuntimeException.error(
				"This agent has no ONNX model. Call load_model or set its 'model' attribute first.", scope);
		return OnnxOperators.predict(scope, model, scope.getArg("input", IType.NONE));
	}

}
