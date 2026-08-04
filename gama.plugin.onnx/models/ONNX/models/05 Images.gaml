/**
* Name: ONNX Images
* Author: Baptiste Lesquoy
* Description: Images can be fed straight to a vision model. The layout (NCHW or NHWC), the channel
*              count and the normalisation are all read from the shape the graph declares, so nothing
*              has to be prepared by hand. The model here multiplies a grayscale image by 255, which
*              means it hands back the gray levels the pixels came from -- easy to check by eye.
*
*              Requires the image extension of GAMA for the 'image' type itself; the plugin does not
*              depend on it.
* Tags: onnx, machine learning, image
*/
model onnx_images

global {

	// image [1,1,height,width] -> levels [1,1,height,width], with height and width left free
	onnx_model gray <- onnx_model("../includes/image_gray.onnx");

	image_file source <- image_file("../includes/gradient.png");

	init {
		write gray.info;
		// Both spatial dimensions are dynamic: the shape is built from the image, not from the
		// number of values, so any size goes through
		write "declared shape : " + gray.input_shapes["image"];   // [1,1,-1,-1]

		image picture <- image(source);
		write "image          : " + picture.width + "x" + picture.height;

		write "\n===== feeding the image =====";
		// A single-channel input means luminance, normalised to [0,1]; multiplying by 255 gives the
		// original gray levels back. The result keeps the rank of the tensor: [1][1][h][w].
		dataframe result <- onnx_predict(gray, picture);
		// The output is [1,1,height,width]. Its first dimension becomes the rows of the dataframe,
		// so there is one row, and its cell holds what is left: [1][height][width].
		list cell <- result["levels"][0];
		list rows <- first(cell);
		write "rows returned  : " + length(rows);
		write "first row      : " + first(rows);
		write "last row       : " + last(rows);

		write "\n===== the conventions the plugin applies =====";
		write "- rank 4 is read as NCHW, unless only the last dimension is a plausible channel";
		write "  count (1, 3 or 4), in which case NHWC";
		write "- rank 3 is the same without the batch, rank 2 is grayscale";
		write "- 1 channel means luminance, 3 means RGB, 4 means RGBA";
		write "- channels are normalised to [0,1]";
		write "- an image whose size does not match a FIXED declared size is refused rather than";
		write "  silently resized: resize it yourself with 'with_size' beforehand";
	}
}

experiment images type: gui {
	output {
		display "Source image" type: 2d {
			image source position: {0, 0} size: {1, 1} refresh:false;
		}
	}
}
