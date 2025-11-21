# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import datetime
import importlib
import inspect
import os
import posixpath

from sphinx.application import Sphinx
from sphinx.writers.html5 import HTML5Translator

# -- Project information -----------------------------------------------------

project = "Sentence Transformers"
copyright = str(datetime.datetime.now().year)
author = "Nils Reimers, Tom Aarsen"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "myst_parser",
    "sphinx_markdown_tables",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx_inline_tabs",
    "sphinxcontrib.mermaid",
    "sphinx_toolbox.collapse",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to include when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
include_patterns = [
    "docs/**",
    "sentence_transformers/**/.py",
    "examples/**",
    "index.rst",
]

intersphinx_mapping = {
    "datasets": ("https://huggingface.co/docs/datasets/main/en/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en/", None),
    "huggingface_hub": ("https://huggingface.co/docs/huggingface_hub/main/en/", None),
    "optimum": ("https://huggingface.co/docs/optimum/main/en/", None),
    "peft": ("https://huggingface.co/docs/peft/main/en/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": True,
    "canonical_url": "https://www.sbert.net",
    "collapse_navigation": False,
    "navigation_depth": 3,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static", "img/hf-logo.svg"]

# Add any paths that contain "extra" files, such as .htaccess or
# robots.txt.
html_extra_path = [".htaccess"]

html_css_files = [
    "css/custom.css",
]

html_js_files = [
    "js/custom.js",
]

html_show_sourcelink = False
html_context = {
    "display_github": True,
    "github_user": "huggingface",
    "github_repo": "sentence-transformers",
    "github_version": "main/",
}

html_logo = "img/logo.png"
html_favicon = "img/favicon.ico"

autoclass_content = "both"

# Required to get rid of some myst.xref_missing warnings
myst_heading_anchors = 3


# https://github.com/readthedocs/sphinx-autoapi/issues/202#issuecomment-907582382
def linkcode_resolve(domain, info):
    # Non-linkable objects from the starter kit in the tutorial.
    if domain == "js" or info["module"] == "connect4":
        return

    assert domain == "py", "expected only Python objects"

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])
    obj = inspect.unwrap(obj)

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    file = os.path.relpath(file, os.path.abspath(".."))
    if not file.startswith("sentence_transformers"):
        # e.g. object is a typing.NewType
        return None
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"https://github.com/huggingface/sentence-transformers/blob/main/{file}#L{start}-L{end}"


def visit_download_reference(self, node):
    root = "https://github.com/huggingface/sentence-transformers/tree/main"
    atts = {"class": "reference download", "download": ""}

    if not self.builder.download_support:
        self.context.append("")
    elif "refuri" in node:
        atts["class"] += " external"
        atts["href"] = node["refuri"]
        self.body.append(self.starttag(node, "a", "", **atts))
        self.context.append("</a>")
    elif "reftarget" in node and "refdoc" in node:
        atts["class"] += " external"
        atts["href"] = posixpath.join(root, os.path.dirname(node["refdoc"]), node["reftarget"])
        self.body.append(self.starttag(node, "a", "", **atts))
        self.context.append("</a>")
    else:
        self.context.append("")


HTML5Translator.visit_download_reference = visit_download_reference


def setup(app: Sphinx):
    pass
