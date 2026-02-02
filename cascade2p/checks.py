

"""
General checks to simplify jupyter notebooks and GUI
"""

def check_packages():
    """ Wrapper for check_yaml and check_keras_version """
    check_yaml()
    check_pytorch_version()


def check_yaml():
    """ Check if ruamel.yaml is installed, otherwise notify user with instructions """

    try:
        import ruamel.yaml
        yaml = ruamel.yaml.YAML(typ='rt')
    except ModuleNotFoundError:
        print('\nModuleNotFoundError: The package "ruamel.yaml" does not seem to be installed on this PC.',
              'This package is necessary to load the configuration files of the models.\n',
              'Please install it with "pip install ruamel.yaml"')
        return

    print('\tYAML reader installed (version {}).'.format(ruamel.yaml.__version__))

def check_pytorch_version():
    """ Import torch and check versions """
    try:
        import torch
    except ModuleNotFoundError:
        print('\nModuleNotFoundError: The package "torch" does not seem to be installed on this PC.',
              'It is not possible to train models or predict neural activity without torch.\n',
              'Please install torch.')
        return


    print('\tTorch installed (version {}).'.format(torch.__version__) )
