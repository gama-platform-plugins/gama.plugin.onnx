/**
* Name: ONNX Walkers
* Author: Baptiste Lesquoy
* Description: Agents driven by a model, with a display. Each walker perceives three values, hands them
*              to the model through the onnx skill, and reads back a speed and a turn. This is what the
*              skill is for: 'do predict' inside a species reads better than threading a model attribute
*              through every expression.
*
*              The weights of the model are hand-set, not trained -- the point is the wiring, not the
*              behaviour. For the pattern that matters to performance (one batched inference for the
*              whole population instead of one call per agent) see 03 Classification.
* Tags: onnx, machine learning, display
*/
model onnx_walkers

global {

	int nb_walkers <- 80 min: 1 max: 600;
	float world_size <- 100.0;

	geometry shape <- square(world_size);

	// One model, shared by every walker: the session cache means 80 walkers hold 1 session, not 80
	onnx_model brain <- onnx_model("../includes/steering.onnx");

	init {
		write brain.info;
		create walker number: nb_walkers;
	}
}

species walker skills: [moving, onnx] {

	// An individual trait, the only thing that differs between walkers
	float drive <- rnd(0.2, 1.0);

	init {
		// Sharing the loaded model rather than calling load_model in every agent
		inner_model <- brain;
		location <- any_location_in(world.shape);
		heading <- rnd(360.0);
	}

	reflex decide {
		float distance_to_centre <- location distance_to world.location;

		// The three perceptions the model was built to read, all roughly in [0,1]
		list<float> perception <- [drive, heading / 360.0, distance_to_centre / world_size];

		// The action returns a dataframe: column "action" (the name of the output), one row here
		list<float> action_vec <- predict(perception)["action"][0];

		speed <- max(0.2, action_vec[0]);
		heading <- heading + action_vec[1];
	}

	reflex act {
		do move;
	}

	// Keeping the population inside the world by wrapping around its edges
	reflex wrap {
		float x <- location.x;
		float y <- location.y;
		if (x < 0.0) { x <- x + world_size; }
		if (x > world_size) { x <- x - world_size; }
		if (y < 0.0) { y <- y + world_size; }
		if (y > world_size) { y <- y - world_size; }
		location <- {x, y};
	}

	// Fast walkers are red, slow ones blue
	rgb colour {
		float ratio <- min(1.0, max(0.0, speed / 1.5));
		return rgb(int(255 * ratio), 60, int(255 * (1 - ratio)));
	}

	aspect default {
		rgb c <- colour();
		draw circle(1.0) color: c;
		draw line([location, location + {cos(heading) * 3.0, sin(heading) * 3.0}]) color: c;
	}
}

experiment walkers type: gui {

	parameter "Number of walkers" var: nb_walkers category: "Population";

	output {
		display "Walkers" type: 2d {
			graphics "background" {
				draw world.shape color: #white border: #lightgray;
			}
			species walker aspect: default;
		}

		display "Speed decided by the model" type: 2d {
			chart "Speed" type: series {
				data "mean" value: mean(walker collect each.speed) color: #black;
				data "max" value: max(walker collect each.speed) color: #red;
				data "min" value: min(walker collect each.speed) color: #blue;
			}
		}
	}
}
