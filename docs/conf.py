# -- Project information -----------------------------------------------------

project = 'PIPPIN-NACO'
author  = 'S. de Regt'

# -- General configuration ---------------------------------------------------

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
#source_suffix = '.rst'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autosectionlabel',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
                      'titles_only': False,
                      'logo_only': True,
                     }

#html_logo = 'figures/logo.png'
html_logo = 'figures/PIPPIN_logo.png'
