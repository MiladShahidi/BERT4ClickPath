import tensorflow as tf
import argparse
import collections


def load_vocabulary(vocab_file):
    if tf.io.gfile.isdir(vocab_file):
        # Strangely enough GFile does not raise an error when it is given a directory to read from.
        # Reported this on Github: https://github.com/tensorflow/tensorflow/issues/46282#issue-782000566
        raise IsADirectoryError(f'{vocab_file} is a directory.')

    with tf.io.gfile.GFile(vocab_file, 'r') as f:
        return tf.strings.strip(f.readlines())


def parse_cmd_line_arguments(args):
    """
    Parses command line arguments and returns parsed values in a dict.

    Args:
        args: A dictionary specifying the name-specification pairs for expected arguments. Specification should be
        either a default value (which will also determine the type of argument to be expected) or a type

        Example:

        {
            'arg_name_1': 1,
            'arge_name_2: float,
            'arg_name_3': 'Some String'
            'arg_name_4': True
        }

        1) integer with default value 1
        2) float and required
        3) string with default value
        4) boolean argument. see below

        Non-boolean arguments are expected to have two leading hyphens as "--filename". Booleans need a single leading
        hyphen: "-silent".

    Returns:
        A dictionary of argument names and parsed argument values.
    """
    parser = argparse.ArgumentParser()
    for param_name, arg_spec in args.items():
        if isinstance(arg_spec, type):  # example: {'arg_1': str}
            parser.add_argument('--%s' % param_name, type=arg_spec, required=True)
        else:  # If the spec provides a value
            if arg_spec is None:
                parser.add_argument('--%s' % param_name, type=type(arg_spec),
                                    default=arg_spec)
            else:
                if isinstance(arg_spec, bool):  # Switch arguments that don't require values
                    actions = {True: 'store_true', False: 'store_false'}
                    parser.add_argument('-%s' % param_name, action=actions[arg_spec])
                else:
                    parser.add_argument('--%s' % param_name, type=type(arg_spec),
                                        default=arg_spec)

    parsed_args = parser.parse_args().__dict__

    return parsed_args


# if __name__ == '__main__':
#     ArgSpec = collections.namedtuple('ArgSpec', ('type', 'default', 'required'), defaults=())
#
#     arg_spec = {
#         'arg_1': 12,
#         'arg_2': 'some_string',
#         'arg_3':
#     }