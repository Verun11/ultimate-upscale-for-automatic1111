import math
import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from modules import processing, shared, images, devices, scripts
from modules.processing import StableDiffusionProcessing
from modules.processing import Processed
from modules.shared import opts, state
from enum import Enum

elem_id_prefix = "ultimateupscale"

class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3

class USDUpscaler():

    def __init__(self, p, image, upscaler_index:int, save_redraw, save_seams_fix, tile_width, tile_height, tiling_mode) -> None:
        self.p:StableDiffusionProcessing = p
        self.image:Image = image
        self.tiling_mode = tiling_mode
        self.upscaler = shared.sd_upscalers[upscaler_index]
        self.redraw = USDURedraw()
        self.redraw.save = save_redraw
        self.redraw.tile_width = tile_width if tile_width > 0 else tile_height
        self.redraw.tile_height = tile_height if tile_height > 0 else tile_width
        self.seams_fix = USDUSeamsFix()
        self.seams_fix.save = save_seams_fix
        self.seams_fix.tile_width = tile_width if tile_width > 0 else tile_height
        self.seams_fix.tile_height = tile_height if tile_height > 0 else tile_width
        self.initial_info = None

        if self.tiling_mode == "Fourths (2x2)":
            self.rows = 2
            self.cols = 2
            self.scale_factor = 1 # No global scaling
            self.scales = []
        elif self.tiling_mode == "Sixths (2x3)":
            self.rows = 3
            self.cols = 2
            self.scale_factor = 1 # No global scaling
            self.scales = []
        else: # Manual mode
            self.rows = math.ceil(self.p.height / self.redraw.tile_height)
            self.cols = math.ceil(self.p.width / self.redraw.tile_width)
            self.scale_factor = math.ceil(max(p.width, p.height) / max(image.width, image.height))

    def get_factor(self, num):
        # Its just return, don't need elif
        if num == 1:
            return 2
        if num % 4 == 0:
            return 4
        if num % 3 == 0:
            return 3
        if num % 2 == 0:
            return 2
        return 0

    def get_factors(self):
        # This function is only relevant for Manual mode if scale_factor > 1
        if self.tiling_mode != "Manual" or self.scale_factor <= 1:
            self.scales = [] # Ensure scales is empty if not used
            return

        scales = []
        current_scale = 1
        current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale_factor == 0:
            self.scale_factor += 1
            current_scale_factor = self.get_factor(self.scale_factor)
        while current_scale < self.scale_factor:
            current_scale_factor = self.get_factor(self.scale_factor // current_scale)
            scales.append(current_scale_factor)
            current_scale = current_scale * current_scale_factor
            if current_scale_factor == 0:
                break
        self.scales = enumerate(scales)

    def upscale(self):
        # This method should only run if tiling_mode is "Manual"
        # and there's a need to upscale (e.g. upscaler is not None and scale_factor > 1)
        if self.tiling_mode != "Manual":
            print(f"Tiling mode is {self.tiling_mode}, skipping global upscale.")
            return

        # Log info for manual upscale
        print(f"Manual Upscale - Canva size: {self.p.width}x{self.p.height}")
        print(f"Manual Upscale - Image size: {self.image.width}x{self.image.height}")
        print(f"Manual Upscale - Scale factor: {self.scale_factor}")

        if self.upscaler.name == "None":
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            print("Manual Upscale - No upscaler selected, resized to target dimensions.")
            return

        self.get_factors() # Calculate factors only if we are upscaling

        if not list(self.scales): # Re-check scales after get_factors
            print("Manual Upscale - No scaling factors determined, image might already be at target size or configuration issue.")
            # Still resize to p.width and p.height to ensure consistency
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            return

        # Reset scales to be iterable again if it was checked
        self.get_factors()

        # Upscaling image over all factors
        for index, value in self.scales:
            print(f"Manual Upscale - Upscaling iteration {index+1} with scale factor {value}")
            self.image = self.upscaler.scaler.upscale(self.image, value, self.upscaler.data_path)

        # Resize image to set values if p.width and p.height are different from upscaled image
        if self.image.width != self.p.width or self.image.height != self.p.height:
            self.image = self.image.resize((self.p.width, self.p.height), resample=Image.LANCZOS)
            print("Manual Upscale - Resized to final target dimensions.")

    def setup_redraw(self, redraw_mode, padding, mask_blur):
        if self.tiling_mode == "Fourths (2x2)" or self.tiling_mode == "Sixths (2x3)":
            self.redraw.mode = USDUMode.LINEAR # Force Linear mode
            self.redraw.enabled = True # Ensure redraw is enabled
            print(f"Tiling mode {self.tiling_mode}: Redraw mode forced to Linear.")
        else:
            self.redraw.mode = USDUMode(redraw_mode)
            self.redraw.enabled = self.redraw.mode != USDUMode.NONE

        self.redraw.padding = padding # Padding might still be relevant for tile processing setup
        self.p.mask_blur = mask_blur # Mask blur for main processing, if any

    def setup_seams_fix(self, padding, denoise, mask_blur, width, mode):
        if self.tiling_mode == "Fourths (2x2)" or self.tiling_mode == "Sixths (2x3)":
            self.seams_fix.enabled = False
            print(f"Tiling mode {self.tiling_mode}: Seams fix disabled.")
            self.seams_fix.mode = USDUSFMode.NONE # Ensure mode is initialized
            return

        self.seams_fix.padding = padding
        self.seams_fix.denoise = denoise
        self.seams_fix.mask_blur = mask_blur
        self.seams_fix.width = width
        self.seams_fix.mode = USDUSFMode(mode)
        self.seams_fix.enabled = self.seams_fix.mode != USDUSFMode.NONE

    def save_image(self):
        if type(self.p.prompt) != list:
            images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt, opts.samples_format, info=self.initial_info, p=self.p)
        else:
            images.save_image(self.image, self.p.outpath_samples, "", self.p.seed, self.p.prompt[0], opts.samples_format, info=self.initial_info, p=self.p)

    def calc_jobs_count(self):
        redraw_job_count = (self.rows * self.cols) if self.redraw.enabled else 0
        seams_job_count = 0
        if self.seams_fix.mode == USDUSFMode.BAND_PASS:
            seams_job_count = self.rows + self.cols - 2
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols
        elif self.seams_fix.mode == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            seams_job_count = self.rows * (self.cols - 1) + (self.rows - 1) * self.cols + (self.rows - 1) * (self.cols - 1)

        state.job_count = redraw_job_count + seams_job_count

    def print_info(self):
        print(f"Tile size: {self.redraw.tile_width}x{self.redraw.tile_height}")
        print(f"Tiles amount: {self.rows * self.cols}")
        print(f"Grid: {self.rows}x{self.cols}")
        print(f"Redraw enabled: {self.redraw.enabled}")
        print(f"Seams fix mode: {self.seams_fix.mode.name}")

    def add_extra_info(self):
        self.p.extra_generation_params["Ultimate SD upscale upscaler"] = self.upscaler.name
        self.p.extra_generation_params["Ultimate SD upscale tile_width"] = self.redraw.tile_width
        self.p.extra_generation_params["Ultimate SD upscale tile_height"] = self.redraw.tile_height
        self.p.extra_generation_params["Ultimate SD upscale mask_blur"] = self.p.mask_blur
        self.p.extra_generation_params["Ultimate SD upscale padding"] = self.redraw.padding

    def process(self):
        state.begin()
        # self.calc_jobs_count() # job_count will be handled differently by redraw methods
        self.result_images = []
        if self.redraw.enabled:
            self.image = self.redraw.start(self.p, self.image, self.rows, self.cols, self.tiling_mode)
            self.initial_info = self.redraw.initial_info
        self.result_images.append(self.image)
        if self.redraw.save:
            self.save_image()

        if self.seams_fix.enabled:
            self.image = self.seams_fix.start(self.p, self.image, self.rows, self.cols)
            self.initial_info = self.seams_fix.initial_info
            self.result_images.append(self.image)
            if self.seams_fix.save:
                self.save_image()
        state.end()

class USDURedraw():

    def init_draw(self, p, width, height):
        p.inpaint_full_res = True
        p.inpaint_full_res_padding = self.padding
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64
        mask = Image.new("L", (width, height), "black")
        draw = ImageDraw.Draw(mask)
        return mask, draw

    def linear_process(self, p: StableDiffusionProcessing, image: Image.Image, rows: int, cols: int, tiling_mode: str):
        if tiling_mode == "Manual":
            # Original linear_process logic for Manual mode
            print(f"Executing original linear_process for Manual mode.")
            state.job_count = rows * cols # As originally implicitly handled
            mask, draw = self.init_draw(p, image.width, image.height)
            for yi in range(rows):
                for xi in range(cols):
                    if state.interrupted:
                        break

                    # Calculate crop region for manual mode (usually full tile)
                    # The existing self.calc_rectangle might be based on p.width/p.height, ensure it's correct for image
                    # For manual mode, self.tile_width and self.tile_height are from UI
                    crop_x1 = xi * self.tile_width
                    crop_y1 = yi * self.tile_height
                    crop_x2 = crop_x1 + self.tile_width
                    crop_y2 = crop_y1 + self.tile_height

                    draw.rectangle((crop_x1, crop_y1, crop_x2, crop_y2), fill="white")
                    p.init_images = [image]
                    p.image_mask = mask
                    # p.width and p.height for process_images are set by init_draw

                    processed = processing.process_images(p)
                    state.job_no += 1

                    draw.rectangle((crop_x1, crop_y1, crop_x2, crop_y2), fill="black")
                    if processed and processed.images:
                        image = processed.images[0]
                if state.interrupted:
                    break

            if processed: # Ensure processed is defined
                 self.initial_info = processed.infotext(p, 0)
            # Restore p.width and p.height to original target for the final image
            # This is implicitly handled as image is modified in place.

            return image
        elif tiling_mode == "Fourths (2x2)" or tiling_mode == "Sixths (2x3)":
            print(f"Executing new linear_process for tiling_mode: {tiling_mode}.")
            overall_target_width = p.width
            overall_target_height = p.height
            final_image_canvas = Image.new("RGB", (overall_target_width, overall_target_height))

            state.job_count = rows * cols
            state.job_no = 0
            processed_info_text = None

            for yi in range(rows):
                for xi in range(cols):
                    if state.interrupted:
                        break

                    # tile_width and tile_height for cropping are from USDUpscaler, based on init_img dimensions
                    crop_x1 = xi * self.tile_width
                    crop_y1 = yi * self.tile_height
                    crop_x2 = min(crop_x1 + self.tile_width, image.width) # Ensure crop doesn't exceed image bounds
                    crop_y2 = min(crop_y1 + self.tile_height, image.height)

                    if crop_x1 >= image.width or crop_y1 >= image.height:
                        print(f"Skipping tile ({xi},{yi}) as crop start is outside image bounds.")
                        continue

                    current_tile_original_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

                    if current_tile_original_image.width == 0 or current_tile_original_image.height == 0:
                        print(f"Skipping tile ({xi},{yi}) as cropped image has zero dimension.")
                        continue

                    p.init_images = [current_tile_original_image]
                    p.image_mask = None # No mask for direct img2img on tiles
                    p.inpainting_fill = 1 # Standard for img2img
                    p.inpaint_full_res_padding = 0 # No padding for this mode

                    tile_output_width = overall_target_width // cols
                    tile_output_height = overall_target_height // rows

                    # Adjust for last tile if not perfectly divisible
                    if xi == cols - 1:
                        tile_output_width = overall_target_width - (xi * (overall_target_width // cols))
                    if yi == rows - 1:
                        tile_output_height = overall_target_height - (yi * (overall_target_height // rows))

                    p.width = tile_output_width
                    p.height = tile_output_height

                    print(f"Processing tile ({xi},{yi}): Crop from ({crop_x1},{crop_y1})-({crop_x2},{crop_y2}), Output Size ({tile_output_width}x{tile_output_height})")

                    # Preserve original settings that might be changed by process_images
                    original_sampler_name = p.sampler_name
                    original_cfg_scale = p.cfg_scale
                    original_denoising_strength = p.denoising_strength

                    # TODO: Determine if specific settings for tile processing are needed
                    # For now, use the main p settings.

                    processed = processing.process_images(p)
                    state.job_no += 1

                    # Restore settings if they were changed
                    p.sampler_name = original_sampler_name
                    p.cfg_scale = original_cfg_scale
                    p.denoising_strength = original_denoising_strength


                    if processed and processed.images and processed.images[0] is not None:
                        processed_tile_image = processed.images[0]
                        if processed_tile_image.size != (tile_output_width, tile_output_height):
                            print(f"Resizing processed tile from {processed_tile_image.size} to ({tile_output_width},{tile_output_height})")
                            processed_tile_image = processed_tile_image.resize((tile_output_width, tile_output_height), Image.LANCZOS)

                        paste_x = xi * (overall_target_width // cols)
                        paste_y = yi * (overall_target_height // rows)
                        final_image_canvas.paste(processed_tile_image, (paste_x, paste_y))
                        if processed.infotexts and len(processed.infotexts) > 0:
                           processed_info_text = processed.infotexts[0] # Store info from last tile
                    else:
                        print(f"Warning: Tile ({xi},{yi}) processing returned no image. Pasting black.")
                        # Optionally, paste the cropped original tile or a placeholder
                        black_tile = Image.new("RGB", (tile_output_width, tile_output_height), "black")
                        paste_x = xi * (overall_target_width // cols)
                        paste_y = yi * (overall_target_height // rows)
                        final_image_canvas.paste(black_tile, (paste_x, paste_y))


                if state.interrupted:
                    break

            self.initial_info = processed_info_text if processed_info_text else "No processing info captured for tiled redraw."
            # Restore p.width and p.height to overall target for subsequent steps (like seams fix)
            p.width = overall_target_width
            p.height = overall_target_height
            return final_image_canvas
        else:
            # Should not happen if tiling_mode is validated earlier
            print(f"Warning: Unknown tiling_mode '{tiling_mode}' in linear_process. Returning original image.")
            return image


    def calc_rectangle(self, xi, yi): # This seems to be for the old mask drawing, may not be needed for new modes
        x1 = xi * self.tile_width
        y1 = yi * self.tile_height
        x2 = xi * self.tile_width + self.tile_width
        y2 = yi * self.tile_height + self.tile_height

        return x1, y1, x2, y2

    def chess_process(self, p, image, rows, cols): # Signature needs tiling_mode if it were to be updated
        mask, draw = self.init_draw(p, image.width, image.height)
        tiles = []
        # calc tiles colors
        for yi in range(rows):
            for xi in range(cols):
                if state.interrupted:
                    break
                if xi == 0:
                    tiles.append([])
                color = xi % 2 == 0
                if yi > 0 and yi % 2 != 0:
                    color = not color
                tiles[yi].append(color)

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    tiles[yi][xi] = not tiles[yi][xi]
                    continue
                tiles[yi][xi] = not tiles[yi][xi]
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(len(tiles)):
            for xi in range(len(tiles[yi])):
                if state.interrupted:
                    break
                if not tiles[yi][xi]:
                    continue
                draw.rectangle(self.calc_rectangle(xi, yi), fill="white")
                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                draw.rectangle(self.calc_rectangle(xi, yi), fill="black")
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols, tiling_mode): # Added tiling_mode
        self.initial_info = None # Reset initial_info
        if self.mode == USDUMode.LINEAR:
            return self.linear_process(p, image, rows, cols, tiling_mode) # Pass tiling_mode
        if self.mode == USDUMode.CHESS:
            # TODO: chess_process would also need to be updated to handle tiling_mode
            print("Chess mode selected but not yet updated for new tiling modes. Using original chess logic.")
            return self.chess_process(p, image, rows, cols) # Original call, needs update for tiling_mode

class USDUSeamsFix():

    def init_draw(self, p):
        self.initial_info = None
        p.width = math.ceil((self.tile_width+self.padding) / 64) * 64
        p.height = math.ceil((self.tile_height+self.padding) / 64) * 64

    def half_tile_process(self, p, image, rows, cols):

        self.init_draw(p)
        processed = None

        gradient = Image.linear_gradient("L")
        row_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        row_gradient.paste(gradient.resize(
            (self.tile_width, self.tile_height//2), resample=Image.BICUBIC), (0, 0))
        row_gradient.paste(gradient.rotate(180).resize(
                (self.tile_width, self.tile_height//2), resample=Image.BICUBIC),
                (0, self.tile_height//2))
        col_gradient = Image.new("L", (self.tile_width, self.tile_height), "black")
        col_gradient.paste(gradient.rotate(90).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (0, 0))
        col_gradient.paste(gradient.rotate(270).resize(
            (self.tile_width//2, self.tile_height), resample=Image.BICUBIC), (self.tile_width//2, 0))

        p.denoising_strength = self.denoise
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(row_gradient, (xi*self.tile_width, yi*self.tile_height + self.tile_height//2))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        for yi in range(rows):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = self.padding
                mask = Image.new("L", (image.width, image.height), "black")
                mask.paste(col_gradient, (xi*self.tile_width+self.tile_width//2, yi*self.tile_height))

                p.init_images = [image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def half_tile_process_corners(self, p, image, rows, cols):
        fixed_image = self.half_tile_process(p, image, rows, cols)
        processed = None
        self.init_draw(p)
        gradient = Image.radial_gradient("L").resize(
            (self.tile_width, self.tile_height), resample=Image.BICUBIC)
        gradient = ImageOps.invert(gradient)
        p.denoising_strength = self.denoise
        #p.mask_blur = 0
        p.mask_blur = self.mask_blur

        for yi in range(rows-1):
            for xi in range(cols-1):
                if state.interrupted:
                    break
                p.width = self.tile_width
                p.height = self.tile_height
                p.inpaint_full_res = True
                p.inpaint_full_res_padding = 0
                mask = Image.new("L", (fixed_image.width, fixed_image.height), "black")
                mask.paste(gradient, (xi*self.tile_width + self.tile_width//2,
                                      yi*self.tile_height + self.tile_height//2))

                p.init_images = [fixed_image]
                p.image_mask = mask
                processed = processing.process_images(p)
                if (len(processed.images) > 0):
                    fixed_image = processed.images[0]

        p.width = fixed_image.width
        p.height = fixed_image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return fixed_image

    def band_pass_process(self, p, image, cols, rows):

        self.init_draw(p)
        processed = None

        p.denoising_strength = self.denoise
        p.mask_blur = 0

        gradient = Image.linear_gradient("L")
        mirror_gradient = Image.new("L", (256, 256), "black")
        mirror_gradient.paste(gradient.resize((256, 128), resample=Image.BICUBIC), (0, 0))
        mirror_gradient.paste(gradient.rotate(180).resize((256, 128), resample=Image.BICUBIC), (0, 128))

        row_gradient = mirror_gradient.resize((image.width, self.width), resample=Image.BICUBIC)
        col_gradient = mirror_gradient.rotate(90).resize((self.width, image.height), resample=Image.BICUBIC)

        for xi in range(1, rows):
            if state.interrupted:
                    break
            p.width = self.width + self.padding * 2
            p.height = image.height
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(col_gradient, (xi * self.tile_width - self.width // 2, 0))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]
        for yi in range(1, cols):
            if state.interrupted:
                    break
            p.width = image.width
            p.height = self.width + self.padding * 2
            p.inpaint_full_res = True
            p.inpaint_full_res_padding = self.padding
            mask = Image.new("L", (image.width, image.height), "black")
            mask.paste(row_gradient, (0, yi * self.tile_height - self.width // 2))

            p.init_images = [image]
            p.image_mask = mask
            processed = processing.process_images(p)
            if (len(processed.images) > 0):
                image = processed.images[0]

        p.width = image.width
        p.height = image.height
        if processed is not None:
            self.initial_info = processed.infotext(p, 0)

        return image

    def start(self, p, image, rows, cols):
        if USDUSFMode(self.mode) == USDUSFMode.BAND_PASS:
            return self.band_pass_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE:
            return self.half_tile_process(p, image, rows, cols)
        elif USDUSFMode(self.mode) == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS:
            return self.half_tile_process_corners(p, image, rows, cols)
        else:
            return image

class Script(scripts.Script):
    def title(self):
        return "Ultimate SD upscale"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):

        target_size_types = [
            "From img2img2 settings",
            "Custom size",
            "Scale from image size"
        ]

        seams_fix_types = [
            "None",
            "Band pass",
            "Half tile offset pass",
            "Half tile offset pass + intersections"
        ]

        redrow_modes = [
            "Linear", # Index 0
            "Chess",  # Index 1
            "None"    # Index 2
        ]

        info_html = """
        <p style="margin-bottom:0.75em">Ultimate SD Upscale processing options:</p>
        <ul>
            <li><b>Manual Mode</b>: Standard upscaling. Uses the global upscaler first, then redraws tiles if enabled. Tile Width/Height, Redraw Type, and Seams Fix options are respected.</li>
            <li><b>Fourths (2x2) / Sixths (2x3) Modes</b>:
                <ul>
                    <li>Divides the original image into 2x2 or 2x3 sections respectively. Each section is processed independently using img2img.</li>
                    <li>The global upscaler (the first 'Upscaler' dropdown) is <b>ignored</b> for these modes.</li>
                    <li>Tile Width & Tile Height sliders are <b>ignored</b> (calculated automatically from original image dimensions).</li>
                    <li>Redraw Type "Chess" is <b>incompatible</b> and will be automatically switched to "Linear".</li>
                    <li>All "Seams Fix" options are <b>incompatible</b> and will be disabled.</li>
                    <li>These modes are useful for applying img2img to large images piece by piece without a large initial upscale, directly targeting final dimensions.</li>
                </ul>
            </li>
        </ul>
        <p style="margin-bottom:0.75em">Ensure 'Target size type' reflects the desired final dimensions of the image.</p>
        """
        info = gr.HTML(info_html)

        with gr.Row():
            target_size_type = gr.Dropdown(label="Target size type", elem_id=f"{elem_id_prefix}_target_size_type", choices=[k for k in target_size_types], type="index",
                                  value=next(iter(target_size_types)))

            custom_width = gr.Slider(label='Custom width', elem_id=f"{elem_id_prefix}_custom_width", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_height = gr.Slider(label='Custom height', elem_id=f"{elem_id_prefix}_custom_height", minimum=64, maximum=8192, step=64, value=2048, visible=False, interactive=True)
            custom_scale = gr.Slider(label='Scale', elem_id=f"{elem_id_prefix}_custom_scale", minimum=1, maximum=16, step=0.01, value=2, visible=False, interactive=True)

        gr.HTML("<p style=\"margin-bottom:0.75em\">Redraw options:</p>")
        with gr.Row():
            upscaler_index = gr.Radio(label='Upscaler', elem_id=f"{elem_id_prefix}_upscaler_index", choices=[x.name for x in shared.sd_upscalers],
                                value=shared.sd_upscalers[0].name, type="index")
        with gr.Row():
            redraw_mode = gr.Dropdown(label="Type", elem_id=f"{elem_id_prefix}_redraw_mode", choices=[k for k in redrow_modes], type="index", value=next(iter(redrow_modes)))
            tiling_mode = gr.Dropdown(label="Tiling Mode", elem_id=f"{elem_id_prefix}_tiling_mode", choices=["Manual", "Fourths (2x2)", "Sixths (2x3)"], value="Manual", type="value")
            tile_width = gr.Slider(elem_id=f"{elem_id_prefix}_tile_width", minimum=0, maximum=2048, step=64, label='Tile width', value=512)
            tile_height = gr.Slider(elem_id=f"{elem_id_prefix}_tile_height", minimum=0, maximum=2048, step=64, label='Tile height', value=0)
            mask_blur = gr.Slider(elem_id=f"{elem_id_prefix}_mask_blur", label='Mask blur', minimum=0, maximum=64, step=1, value=8)
            padding = gr.Slider(elem_id=f"{elem_id_prefix}_padding", label='Padding', minimum=0, maximum=512, step=1, value=32)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Seams fix:</p>")
        with gr.Row():
            seams_fix_type = gr.Dropdown(label="Type", elem_id=f"{elem_id_prefix}_seams_fix_type", choices=[k for k in seams_fix_types], type="index", value=next(iter(seams_fix_types)))
            seams_fix_denoise = gr.Slider(label='Denoise', elem_id=f"{elem_id_prefix}_seams_fix_denoise", minimum=0, maximum=1, step=0.01, value=0.35, visible=False, interactive=True)
            seams_fix_width = gr.Slider(label='Width', elem_id=f"{elem_id_prefix}_seams_fix_width", minimum=0, maximum=128, step=1, value=64, visible=False, interactive=True)
            seams_fix_mask_blur = gr.Slider(label='Mask blur', elem_id=f"{elem_id_prefix}_seams_fix_mask_blur", minimum=0, maximum=64, step=1, value=4, visible=False, interactive=True)
            seams_fix_padding = gr.Slider(label='Padding', elem_id=f"{elem_id_prefix}_seams_fix_padding", minimum=0, maximum=128, step=1, value=16, visible=False, interactive=True)
        gr.HTML("<p style=\"margin-bottom:0.75em\">Save options:</p>")
        with gr.Row():
            save_upscaled_image = gr.Checkbox(label="Upscaled", elem_id=f"{elem_id_prefix}_save_upscaled_image", value=True)
            save_seams_fix_image = gr.Checkbox(label="Seams fix", elem_id=f"{elem_id_prefix}_save_seams_fix_image", value=False)

        def select_fix_type(fix_index):
            all_visible = fix_index != 0
            mask_blur_visible = fix_index == 2 or fix_index == 3
            width_visible = fix_index == 1

            return [gr.update(visible=all_visible),
                    gr.update(visible=width_visible),
                    gr.update(visible=mask_blur_visible),
                    gr.update(visible=all_visible)]

        seams_fix_type.change(
            fn=select_fix_type,
            inputs=seams_fix_type,
            outputs=[seams_fix_denoise, seams_fix_width, seams_fix_mask_blur, seams_fix_padding]
        )

        def select_scale_type(scale_index):
            is_custom_size = scale_index == 1
            is_custom_scale = scale_index == 2

            return [gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_size),
                    gr.update(visible=is_custom_scale),
                    ]

        target_size_type.change(
            fn=select_scale_type,
            inputs=target_size_type,
            outputs=[custom_width, custom_height, custom_scale]
        )

        def select_tiling_mode(tiling_mode_value):
            if tiling_mode_value == "Manual":
                return gr.update(interactive=True), gr.update(interactive=True)
            else:
                return gr.update(interactive=False), gr.update(interactive=False)

        tiling_mode.change(
            fn=select_tiling_mode,
            inputs=tiling_mode,
            outputs=[tile_width, tile_height]
        )

        def init_field(scale_name):
            try:
                scale_index = target_size_types.index(scale_name)
                custom_width.visible = custom_height.visible = scale_index == 1
                custom_scale.visible = scale_index == 2
            except:
                pass

        target_size_type.init_field = init_field

        return [info, tiling_mode, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
                upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
                seams_fix_type, target_size_type, custom_width, custom_height, custom_scale]

    def run(self, p, _, tiling_mode, tile_width, tile_height, mask_blur, padding, seams_fix_width, seams_fix_denoise, seams_fix_padding,
            upscaler_index, save_upscaled_image, redraw_mode, save_seams_fix_image, seams_fix_mask_blur,
            seams_fix_type, target_size_type, custom_width, custom_height, custom_scale):

        # Init
        processing.fix_seed(p)
        devices.torch_gc()

        p.do_not_save_grid = True
        p.do_not_save_samples = True
        p.inpaint_full_res = False

        p.inpainting_fill = 1
        p.n_iter = 1
        p.batch_size = 1

        seed = p.seed

        # Init image
        init_img = p.init_images[0]
        if init_img == None:
            return Processed(p, [], seed, "Empty image")
        init_img = images.flatten(init_img, opts.img2img_background_color)

        #override size
        if target_size_type == 1:
            p.width = custom_width
            p.height = custom_height
        if target_size_type == 2:
            p.width = math.ceil((init_img.width * custom_scale) / 64) * 64
            p.height = math.ceil((init_img.height * custom_scale) / 64) * 64

        # Determine tile width and height based on tiling_mode
        current_tile_width = tile_width
        current_tile_height = tile_height

        if tiling_mode == "Fourths (2x2)":
            current_tile_width = init_img.width // 2
            current_tile_height = init_img.height // 2
        elif tiling_mode == "Sixths (2x3)":
            current_tile_width = init_img.width // 2
            current_tile_height = init_img.height // 3

        # Ensure tile dimensions are not zero
        if current_tile_width == 0:
            current_tile_width = init_img.width
        if current_tile_height == 0:
            current_tile_height = init_img.height


        # Upscaling
        upscaler = USDUpscaler(p, init_img, upscaler_index, save_upscaled_image, save_seams_fix_image, current_tile_width, current_tile_height, tiling_mode)

        if tiling_mode == "Manual":
            upscaler.upscale()
        else:
            # For "Fourths" or "Sixths", upscale() is skipped.
            # Ensure p.width and p.height are set to the init_img dimensions if no upscaling is done
            # or if the target dimensions are meant to be the original image dimensions for tiling.
            # This depends on how p.width and p.height are used by redraw.
            # For now, we assume p.width and p.height are the FINAL target canvas size.
            # The redraw process will use init_img as its base.
            print(f"Tiling mode is {tiling_mode}, global upscale skipped. Tiles will be processed on original image sized {init_img.width}x{init_img.height}.")
            # If target_size_type was 'Custom size' or 'Scale from image size', p.width/p.height might be different
            # from init_img.width/height. Redraw process needs to handle this.
            # USDUpscaler.image is already init_img.
            pass # upscale() is intentionally skipped

        # Override redraw_mode and seams_fix_type for Fourths/Sixths if incompatible
        if tiling_mode == "Fourths (2x2)" or tiling_mode == "Sixths (2x3)":
            if redraw_mode == 1: # Chess mode index
                print("Warning: Chess redraw mode is not compatible with Fourths/Sixths tiling. Switching to Linear mode.")
                redraw_mode = 0 # Force Linear

            if seams_fix_type != 0: # Not "None"
                print("Warning: Seams fix is not compatible with Fourths/Sixths tiling and will be disabled.")
                # The actual disabling happens in upscaler.setup_seams_fix based on tiling_mode
                # No need to change seams_fix_type variable here as setup_seams_fix will handle it.
        
        # Drawing
        upscaler.setup_redraw(redraw_mode, padding, mask_blur)
        upscaler.setup_seams_fix(seams_fix_padding, seams_fix_denoise, seams_fix_mask_blur, seams_fix_width, seams_fix_type) # seams_fix_type is passed but might be ignored
        upscaler.print_info()
        upscaler.add_extra_info()
        upscaler.process()
        result_images = upscaler.result_images

        return Processed(p, result_images, seed, upscaler.initial_info if upscaler.initial_info is not None else "")

