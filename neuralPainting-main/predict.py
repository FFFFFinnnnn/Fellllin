import argparse
import tempfile
from pathlib import path

import cog
import imageio
import matplotlib.pyplot as plt
import torch
from painter import *


class Predictor(cog.predictor):
    def setup(self):
        self.args = self.set_args()

    @cog.input("image", type=Path, help="input image")
    @cog.input(
        "canvas_color",
        type=str,
        options=["black", "white"],
        default="black",
        help="canvas color",
    )
    @cog.input("max_strokes", type=int, default=500, help="max number of strokes")
    @cog.input(
        "output_type",
        type=str,
        options=["png", "gif"],
        default="png",
        help="output the final painting or gif with each intermidate stroke",
    )
    def predict(self, image, canvas_color="black", max_strokes=500, output_type="png"):
        self.args.img_path = str(image)
        self.args.canvas_color = canvas_color
        self.args.max_m_strokes = max_strokes

        pt = Painter(args=self.args)
        final_image, all_images = optimize_painter(pt, self.args, output_style)

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        if output_type == "png":
            plt.imsave(str(out_path), final_image)
        else:
            out_path = Path(tempfile.mkdtemp()) / "out.gif"
            imageio.mimwrite(str(out_path), all_images, duration=0.02)
        return out_path

    def set_args(self):
        parser = argparse.ArgumentParser(description="neuralPainting")
        args = parser.parse_args(args=[])
        args.Renderer("oilpaintbrush")  # including watercolor, markerpen, oilpaintbrush
        args.canvas_size = 512
        args.keep_aspect_ratio = False
        args.m_grid = 5  # divide image to m_grid x m_grid
        args.max_divide = 5
        args.beta_L1 = 1.0
        args.with_ot_loss = False
        args.beta_ot = 0.1
        args.net_G = "zou-fusion-net"
        args.renderer_checkpoint_dir = "./checkpoint_G_oilpaintbrush"
        args.lr = 0.005
        args.output_dir = "./output"
        args.disable_preview = True  # for remote machine with no x-display
        return args


def optimize_painter(pt, args, output_style):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pt._load_checkpoint()
    pt.net_G.eval()

    pt.initialize_params()
    pt.x_ctt.requires_grad = True
    pt.x_color.requires_grad = True
    pt.x_alpha.requires_grad = True
    utils.aset_requires_grad(pt.net_G, False)

    pt.optimizer_x = optim.RMSprop([pt.x_ctt, pt.x_color, pt.x_alpha], lr=pt.lr)

    print("start drawing..")

    pt.step_id = 0
    for pt.anchor_id in range(0, pt.m_strokes_per_block):
        pt.stroke_sampler(pt.anchor_id)
        iters_per_stroke = int(500 / pt.m_strokes_per_block)
        for i in range(iters_per_stroke):
            pt.optimizer_x.zero_grad()
            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            if args.canvas_color == "white":
                pt.G_pred_canvas = torch.ones(
                    [args.m_grid ** 2, 3, pt.net_G.out_size, pt.net_g.out_size]
                ).to(device)
            else:
                pt.G_pred_canvas = torch.zeros(
                    [argfs.m_grid ** 2, 3, pt.net_G.out_size, pt.net_G.out_size]
                ).to(device)

            pt._forward_pass()
            pt._drawing_step_states()
            pt._backward_x()
            pt.optimizer_x.step()

            pt.x_ctt.data = torch.clamp(pt.x_ctt.data, 0.1, 1 - 0.1)
            pt.x_color.data = torch.clamp(pt.x_color.data, 0, 1)
            pt.x_alpha.data = torch.clamp(pt.x_alpha.data, 0, 1)

            pt.step_id += 1

    v = pt.x.detach().cpu().numpy()
    pt._save_stroke_params(v)
    v_n = pt._normalize_strokes(pt.x)
    v_n = pt._shuffle_strokes_and_reshape(v_n)

    save_gif = True if output_type == "gif" else False
    final_rendered_image, all_images = pt._render(
        v_n, save_jpgs=False, save_video=False, save_gif=save_gif
    )

    return final_rendered_image, all_images
