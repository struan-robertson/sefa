# python 3.7
# pyright: basic

"""Demo."""

import argparse

import numpy as np
import PIL
import streamlit as st
import torch

import dnnlib
import legacy
import SessionState
from models import parse_gan_type
from utils import factorize_weight, load_generator, postprocess, to_tensor

torch.cuda.empty_cache()


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    return load_generator(model_name)


# @st.cache(allow_output_mutation=True, show_spinner=False)
def factorize_model(model, size, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(model, size, layer_idx)


def sample(model, gan_type, num=1):
    """Samples latent codes."""
    # device = torch.device('cuda')
    # codes = torch.from_numpy(torch.randn(1, model.z_dim)).to(device).cpu().detach().numpy()
    codes = torch.randn(1, model.z_dim).cuda()
    if gan_type == "pggan":
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == "stylegan":
        codes = model.mapping(codes)["w"]
        codes = model.truncation(codes, trunc_psi=0.7, trunc_layers=8)
    elif gan_type == "stylegan2":
        codes = model.mapping(codes, None)
        # codes = model.truncation(codes,
        #                          trunc_psi=0.5,
        #                          trunc_layers=17)
    codes = codes.detach().cpu().numpy()
    return codes


# @st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == "pggan":
        image = model(to_tensor(code))["image"]
    elif gan_type in ["stylegan", "stylegan2"]:
        # image = model.synthesis(to_tensor(code))#['image']
        print(code)
        image = model(to_tensor(code[0]), None, truncation_psi=1)
        # image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    image = postprocess(image)
    # print(image.shape)
    return image[0]


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description="Discover semantics from the pre-trained weight."
    )
    parser.add_argument("model", type=str, help="Name to the pre-trained model.")
    parser.add_argument("name", type=str, help="Name to the pre-trained model.")
    parser.add_argument("size", type=int, help="Name to the pre-trained model.")
    parser.add_argument(
        "--type",
        type=str,
        default="stylegan2",
        help="gan type " "(default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # os.makedirs(args.save_dir, exist_ok=True)
    """Main function (loop for StreamLit)."""
    st.title("Closed-Form Factorization of Latent Semantics in GANs")
    st.sidebar.title("Options")
    reset = st.sidebar.button("Reset")

    model_name = st.sidebar.selectbox(
        "Model to Interpret",
        # ['stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256', 'pggan_celebahq1024'])
        # ['stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256', 'pggan_celebahq1024', 'custom_stylegan2'])
        [args.name],
    )

    # model = get_model(model_name)
    # print("stage 0")
    # gan_type = parse_gan_type(model)
    gan_type = args.type
    layer_idx = st.sidebar.selectbox(
        "Layers to Interpret", ["all", "0-1", "2-5", "6-17"]
    )
    # print("stage 1")
    device = torch.device("cuda")
    print(args.model)
    with dnnlib.util.open_url(args.model) as f:
        model = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
    layers, boundaries, eigen_values = factorize_model(model, args.size, layer_idx)
    # print("stage 2")
    num_semantics = st.sidebar.number_input(
        "Number of semantics", value=10, min_value=0, max_value=None, step=1
    )
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    if gan_type == "pggan":
        max_step = 5.0
    elif gan_type == "stylegan":
        max_step = 2.0
    elif gan_type == "stylegan2" or gan_type == "stylegan3":
        max_step = 15.0
    for sem_idx in steps:
        eigen_value = eigen_values[sem_idx]
        steps[sem_idx] = st.sidebar.slider(
            f"Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})",
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step,
        )
        # if not reset else 0.0)

    image_placeholder = st.empty()
    button_placeholder = st.empty()

    # try:
    #     base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
    # except FileNotFoundError:
    base_codes = sample(model, gan_type)

    state = SessionState.get(model_name=model_name, code_idx=0, codes=base_codes[0:1])
    if state.model_name != model_name:
        state.model_name = model_name
        state.code_idx = 0
        state.codes = base_codes[0:1]

    if button_placeholder.button("Random", key=0):
        state.code_idx += 1
        if state.code_idx < base_codes.shape[0]:
            state.codes = base_codes[state.code_idx][np.newaxis]
        else:
            state.codes = sample(model, gan_type)
    # print("hoi")
    code = state.codes.copy()
    for sem_idx, step in steps.items():
        if gan_type == "pggan":
            code += boundaries[sem_idx : sem_idx + 1] * step
        elif gan_type in ["stylegan", "stylegan2", "stylegan3"]:
            code[:, layers, :] += boundaries[sem_idx : sem_idx + 1] * step
    image = synthesize(model, gan_type, code)
    image_placeholder.image(image / 255.0)


if __name__ == "__main__":
    main()
