import re

import gradio as gr
from modules import script_callbacks
from modules import generation_parameters_copypaste as params_copypaste

import scripts.t2p.prompt_generator as pgen

wd_like = pgen.WDLike()

# brought from modules/deepbooru.py
re_special = re.compile(r'([\\()])')

def get_conversion(choice: int):
    if choice == 0: return pgen.ProbabilityConversion.CUTOFF_AND_POWER
    elif choice == 1: return pgen.ProbabilityConversion.SOFTMAX
    else: raise NotImplementedError()

def get_sampling(choice: int):
    if choice == 0: return pgen.SamplingMethod.NONE
    elif choice == 1: return pgen.SamplingMethod.TOP_K
    elif choice == 2: return pgen.SamplingMethod.TOP_P
    else: raise NotImplementedError()

def get_tag_range_txt(tag_range: int):
    if wd_like.database is None:
        return 'Tag range: NONE'
    maxval = len(wd_like.database.tag_idx) - 1
    i = max(0, min(tag_range, maxval))
    r = wd_like.database.tag_idx[i]
    return f'Tag range: <b> &gt; {r[0]} tagged</b> ({r[1] + 1} tags total)'


def dd_database_changed(database_name: str, tag_range: int):
    wd_like.load_data(database_name)
    return [
        gr.Slider.update(tag_range, 0, len(wd_like.database.tag_idx) - 1),
        get_tag_range_txt(tag_range)
    ]


def sl_tag_range_changed(tag_range: int):
    return get_tag_range_txt(tag_range)


def generate_prompt(text: str, tag_range: int, conversion: int, power: float, sampling: int, n: int, k: int, p: float, weighted: bool, replace_underscore: bool, excape_brackets: bool):
    wd_like.load_model() #skip loading if not needed
    tags = wd_like(text, pgen.GenerationSettings(tag_range, get_conversion(conversion), power, get_sampling(sampling), n, k, p, weighted))
    if replace_underscore: tags = [t.replace('_', ' ') for t in tags]
    if excape_brackets: tags = [re.sub(re_special, r'\\\1', t) for t in tags]
    return ', '.join(tags)


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as text2prompt_interface:
        with gr.Row():
            with gr.Column():
                tb_input = gr.Textbox(label='Input Theme', interactive=True)
                cb_replace_underscore = gr.Checkbox(value=True, label='Replace underscore in tag with whitespace', interactive=True)
                cb_escape_brackets = gr.Checkbox(value=True, label='Escape brackets in tag', interactive=True)
                btn_generate = gr.Button(value='Generate', variant='primary')
                tb_output = gr.Textbox(label='Output', interactive=True)
                with gr.Row():
                    buttons = params_copypaste.create_buttons(["txt2img", "img2img", "inpaint", "extras"])
                params_copypaste.bind_buttons(buttons, None, tb_output)
                
            with gr.Column(variant='panel'):
                gr.HTML(value='Generation Settings')
                choices = wd_like.get_model_names()
                with gr.Column():
                    if choices: wd_like.load_data(choices[0])
                    dd_database = gr.Dropdown(choices=choices, value=choices[0] if choices else None, interactive=True, label='Database')
                    sl_tag_range = gr.Slider(0, 8, 0, step=1, interactive=True, label='Tag count filter')
                    txt_tag_range = gr.HTML(get_tag_range_txt(0))
                with gr.Column():
                    rb_prob_conversion_method = gr.Radio(choices=['Cutoff and Power', 'Softmax'], value='Cutoff and Power', type='index', label='Method to convert similarity into probability')
                    sl_power = gr.Slider(0, 5, value=2, step=0.1, label='Power', interactive=True)
                    rb_sampling_method = gr.Radio(choices=['NONE', 'Top-k', 'Top-p (Nucleus)'], value='Top-k', type='index', label='Sampling method')
                    nb_max_tag_num = gr.Number(value=20, label='Max number of tags', precision=0, interactive=True)
                    nb_k_value = gr.Number(value=50, label='k value', precision=0, interactive=True)
                    sl_p_value = gr.Slider(0, 1, label='p value', value=0.1, step=0.01, interactive=True)
                    cb_weighted = gr.Checkbox(value=True, label='Use weighted choice', interactive=True)

        dd_database.change(
            fn=dd_database_changed,
            inputs=[dd_database, sl_tag_range],
            outputs=[sl_tag_range, txt_tag_range]
        )

        sl_tag_range.change(
            fn=sl_tag_range_changed,
            inputs=sl_tag_range,
            outputs=txt_tag_range
        )

        nb_max_tag_num.change(
            fn=lambda x: max(0, x),
            inputs=nb_max_tag_num,
            outputs=nb_max_tag_num
        )

        nb_k_value.change(
            fn=lambda x: max(1, x),
            inputs=nb_k_value,
            outputs=nb_k_value
        )

        btn_generate.click(
            fn=generate_prompt,
            inputs=[
                tb_input,
                sl_tag_range,
                rb_prob_conversion_method,
                sl_power,
                rb_sampling_method,
                nb_max_tag_num,
                nb_k_value,
                sl_p_value,
                cb_weighted,
                cb_replace_underscore,
                cb_escape_brackets
            ],
            outputs=tb_output
        )
        
    return [(text2prompt_interface, "Text2Prompt", "text2prompt_interface")]


script_callbacks.on_ui_tabs(on_ui_tabs)