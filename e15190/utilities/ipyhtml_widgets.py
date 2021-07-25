import base64
import copy
import io
import pathlib

import IPython
from mako.template import Template
import numpy as np
import PIL

from e15190 import PROJECT_DIR

if 'IPYTHON_HTML_WIDGET_TAG' not in globals():
    IPYTHON_HTML_WIDGET_TAG = 0

class ImageDisplayer:
    def __init__(self):
        self.template_directory = pathlib.Path(PROJECT_DIR, 'database/utilities/ipyhtml_widgets')

    @staticmethod
    def mpl_to_pil(mpl_fig):
        buf = mpl_fig.canvas.buffer_rgba()
        return PIL.Image.fromarray(np.asarray(buf))

    @staticmethod
    def pil_to_str64(pil_img):
        with io.BytesIO() as stream:
            pil_img.save(stream, format='png')
            string64 = 'data:image/png;base64,'
            string64 += base64.b64encode(stream.getvalue()).decode('utf-8')
        return string64

    @staticmethod
    def mpl_to_str64(mpl_fig):
        return ImageDisplayer.pil_to_str64(ImageDisplayer.mpl_to_pil(mpl_fig))
    
    @staticmethod
    def render(template, subs):
        global IPYTHON_HTML_WIDGET_TAG
        IPYTHON_HTML_WIDGET_TAG += 1
        return template.render(**subs)

    def _get_html(
        self,
        figures,
        method,
        first,
        image_style,
        slider_style,
        init_animation_interval,
    ):
        # parse the figures
        if isinstance(figures, dict):
            figures = copy.copy(figures)
        elif isinstance(figures, list):
            figures = {ii: fig for ii, fig in enumerate(figures)}
        else: # assume single image
            image = ImageDisplayer.mpl_to_str64(figures)
            return r'<img src="%s" style="%s">' % (image, image_style)

        # convert matplotlib figures into str64
        for label, image in figures.items():
            figures[label] = ImageDisplayer.mpl_to_str64(image)

        # decide which to show first
        if isinstance(first, int): # specified by simple index in figures
            pass
        elif isinstance(first, str): # specified by label (key) in figures
            first = list(figures).index(first)
        
        # format image style
        if isinstance(image_style, str):
            pass
        elif isinstance(image_style, dict):
            image_style = ' '.join([f'{key}: {val};' for key, val in image_style.items()])
        
        # render template and return
        template_filename = f'image_displayer_{method}.html.mako'
        template_path = pathlib.Path(self.template_directory, template_filename)
        template = Template(filename=str(template_path))
        subs = dict(
            uniqtag=f'ipywg{IPYTHON_HTML_WIDGET_TAG}',
            figures=figures,
            first=first,
            image_style=image_style,
            slider_style=slider_style,
            init_animation_interval=init_animation_interval,
        )
        return ImageDisplayer.render(template, subs)

    def display(
        self,
        figures,
        method='button',
        first=0,
        image_style='height: 300px;',
        slider_style='',
        init_animation_interval=1000,
        debug=False,
    ):
        fkwargs = copy.copy(locals())
        for key in ['self', 'debug']:
            fkwargs.pop(key)
        html = self._get_html(**fkwargs)
        return html if debug else IPython.display.HTML(html)

    def write_to_file(
        self,
        figures,
        outpath,
        method='button',
        first=0,
        image_style='height: 300px;',
        slider_style='',
        init_animation_interval=1000,
    ):
        fkwargs = copy.copy(locals())
        for key in ['self', 'outpath']:
            fkwargs.pop(key)
        html = self._get_html(**fkwargs)
        with open(outpath, 'w') as file:
            file.write(html)

_imager_displayer = ImageDisplayer()

def display(
    figures,
    method='button',
    first=0,
    image_style='height: 300px;',
    slider_style='',
    init_animation_interval=1000,
    debug=False,
):
    global _imager_displayer
    return _imager_displayer.display(**locals())

def write_to_file(
    figures,
    outpath,
    method='button',
    first=0,
    image_style='height: 300px;',
    slider_style='',
    init_animation_interval=1000,
):
    global _imager_displayer
    return _imager_displayer.write_to_file(**locals())
