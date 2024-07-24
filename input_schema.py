INPUT_SCHEMA = {
    "prompt_value": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["A male speaker with a low-pitched voice delivering his words at a fast pace in a small, confined space with a very clear audio and an animated tone."]
    },
    "input_value": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Remember - this is only the first iteration of the model! To improve the prosody and naturalness of the speech further, we're scaling up the amount of training data by a factor of five times."]
    }
}
IS_STREAMING_OUTPUT = True
