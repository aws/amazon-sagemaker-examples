import json

class TrainingChannels:

    def __init__(self, name, description, supported_content_types = [], is_required=True, supported_compression_types = ["None"], supported_input_modes = ["File"]):
        self.Name = name
        self.Description = description
        self.IsRequired = is_required
        self.SupportedContentTypes = supported_content_types
        self.SupportedCompressionTypes = supported_compression_types
        self.SupportedInputModes = supported_input_modes

    def to_json(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True)