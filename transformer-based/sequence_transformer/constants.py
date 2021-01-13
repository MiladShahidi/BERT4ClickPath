MASK_EVERY_N_ITEMS = 5
SYNTHETIC_POSITIVE_SAMPLE_RATE = 0.5

LABEL_PAD = -1.0  # Labels should be padded with -1, since 0 indicates class 0

# # Reserved tokens for vocabulary
NUM_RESERVED_TOKENS = 10

INPUT_PADDING_TOKEN = '[PAD]'
UNKNOWN_TOKEN = '[UNK]'
INPUT_MASKING_TOKEN = '[MASK]'  # This is for unsupervised pre-is_training. Not used in the supervised is_training
CLASSIFICATION_TOKEN = '[CLS]'
SEPARATOR_TOKEN = '[SEP]'
MISSING_EVENT_OR_ITEM_TOKEN = '[NA]'

# It's important that these match the top rows of the vocabulary file exactly.
RESERVED_TOKENS = [
    INPUT_PADDING_TOKEN,
    INPUT_MASKING_TOKEN,
    UNKNOWN_TOKEN,
    CLASSIFICATION_TOKEN,
    SEPARATOR_TOKEN,
    MISSING_EVENT_OR_ITEM_TOKEN,
]

# Append [RESERVED_i] tokens to the list of reserved tokens to make it NUM_RESERVED_TOKENS long
RESERVED_TOKENS += [f'[RESERVED_{i}]' for i in range(len(RESERVED_TOKENS), NUM_RESERVED_TOKENS)]
INPUT_PAD = RESERVED_TOKENS.index(INPUT_PADDING_TOKEN)
UNKNOWN_INPUT = RESERVED_TOKENS.index(UNKNOWN_TOKEN)
INPUT_MASK = RESERVED_TOKENS.index(UNKNOWN_TOKEN)
CLS = RESERVED_TOKENS.index(CLASSIFICATION_TOKEN)
SEP = RESERVED_TOKENS.index(SEPARATOR_TOKEN)
MISSING_EVENT_OR_ITEM = RESERVED_TOKENS.index(MISSING_EVENT_OR_ITEM_TOKEN)

# ITEM_EMBEDDING_DIM = 20

# Warning, the following name is used to save and restore item embeddings in checkpoints. The name in the checkpoint
# should be the same as the one that the loading script expects. Otherwise it won't find it. So, if this name is changed
# we won't be able to load previously saved checkpoints.
# Hint: You can use tf.train.list_variables to see all variables in a checkpoint file
ITEM_EMBEDDING_LAYER_NAME = 'item_embedding_layer'
